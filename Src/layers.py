import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log
from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Reduce

class VIB(nn.Module):
    def __init__(self, dim=128, encoding_dim=64, output_dim=1, deterministic_eval=False, deterministic_train=False, std_offset=-5.0):
        super().__init__()
        self.K = encoding_dim
        self.dim = dim
        self.output_dim = output_dim
        self.deterministic_eval = deterministic_eval
        self.deterministic_train = deterministic_train
        self.std_offset = float(std_offset)

        self.encode = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 2 * self.K)
        )

        self.decode = nn.Linear(self.K, output_dim)

    def forward(self, x, num_sample=1, prior_mu=None, prior_std=None, valid_mask=None, std_offset=None):
        # x shape: [bsz, len, dim]
        statistics = self.encode(x)  # [bsz, len, 2*K]
        mu = statistics[..., :self.K]  # [bsz, len, K]
        active_std_offset = self.std_offset if std_offset is None else float(std_offset)
        std = F.softplus(statistics[..., self.K:] + active_std_offset, beta=1).clamp_min(1e-6)

        encoding = self.reparametrize_n(mu, std, num_sample)  # [n, bsz, len, K] if n>1 else [bsz, len, K]
        logits = self.decode(encoding)  # [n, bsz, len, output_dim] or [bsz, len, output_dim]

        if num_sample > 1:
            logits = F.softmax(logits, dim=-1).mean(dim=0)  # [bsz, len, output_dim]

        kl_loss = self.compute_kl_loss(
            mu, std, prior_mu=prior_mu, prior_std=prior_std, valid_mask=valid_mask
        )

        return (mu, std), encoding, logits, kl_loss

    def reparametrize_n(self, mu, std, n=1):
        if n != 1:
            # Add sample dimension and expand to n samples
            mu = mu.unsqueeze(0).expand(n, *mu.size())
            std = std.unsqueeze(0).expand(n, *std.size())

        if (self.deterministic_eval and not self.training) or (self.deterministic_train and self.training):
            return mu
        
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl_loss(self, mu, std, prior_mu=None, prior_std=None, valid_mask=None):
        std = std.clamp_min(1e-6)
        if prior_mu is None or prior_std is None:
            kl = 0.5 * (mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
            return self._masked_kl_mean(kl, valid_mask)

        while prior_mu.dim() < mu.dim():
            prior_mu = prior_mu.unsqueeze(0)
        while prior_std.dim() < std.dim():
            prior_std = prior_std.unsqueeze(0)
        prior_std = prior_std.clamp_min(1e-6)
        kl = torch.log(prior_std / std) + (std.pow(2) + (mu - prior_mu).pow(2)) / (2 * prior_std.pow(2)) - 0.5
        return self._masked_kl_mean(kl, valid_mask)

    @staticmethod
    def _masked_kl_mean(kl, valid_mask):
        if valid_mask is None:
            return kl.mean()
        mask = valid_mask.to(device=kl.device, dtype=kl.dtype)
        while mask.dim() < kl.dim():
            mask = mask.unsqueeze(-1)
        denom = mask.sum() * kl.size(-1)
        if denom.item() == 0:
            return kl.sum() * 0.0
        return (kl * mask).sum() / denom.clamp_min(1.0)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('gelu'))
                if m.bias is not None:
                    m.bias.data.zero_()

# Perceiver cross-attention utilities.
def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_all(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )
        
    def forward(self, x):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)
        x_out = x + x_out

        return x_out


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )
        
    def forward(self, x, data, valid_mask=None):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(data).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(data).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q.float() @ k.float().transpose(-2, -1)  # [B, heads, s, s]
        if valid_mask is not None:
            key_mask = valid_mask.to(device=attn.device, dtype=torch.bool)[:, None, None, :]
            attn = attn.masked_fill(~key_mask, torch.finfo(attn.dtype).min)
        # Normalize attention weights over valid keys.
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)
        x_out = x + x_out
        if valid_mask is not None:
            x_out = x_out * valid_mask.to(x_out.dtype).unsqueeze(-1)

        return x_out
