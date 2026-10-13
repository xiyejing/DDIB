import numpy as np
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).with_name('Src')
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from Adapters import DDIBConditionMixin
from layers import *

class DDIBModel(DDIBConditionMixin, nn.Module):
    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(DDIBModel, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate
        self.depth = depth
        self.test_condition = args.test_condition
        self.ablation = getattr(args, 'ablation', 'none')
        self.fusion_aib = getattr(args, 'fusion_aib', False)
        self.DDIB_uni = getattr(args, 'DDIB_uni', False)
        self.DDIB_fusion = getattr(args, 'DDIB_fusion', False)
        self.DDIB_fusion_residual_alpha = float(getattr(args, 'DDIB_fusion_residual_alpha', 1.0))
        self.DDIB_fusion_residual_mode = getattr(args, 'DDIB_fusion_residual_mode', 'blend')
        self.DDIB_fusion_gate_init = float(getattr(args, 'DDIB_fusion_gate_init', 0.05))
        self.DDIB_fusion_gate_reg_lambda = float(getattr(args, 'DDIB_fusion_gate_reg_lambda', 0.0))
        self.DDIB_fusion_deterministic_train = getattr(args, 'DDIB_fusion_deterministic_train', False)
        self.DDIB_fusion_alpha_scale = 1.0
        self.DDIB_fusion_deterministic_eval = getattr(args, 'DDIB_fusion_deterministic_eval', False)
        self.modality_prior = getattr(args, 'modality_prior', False)
        self.condition_prior = getattr(args, 'condition_prior', False)
        self.deterministic_eval_aib = getattr(args, 'deterministic_eval_aib', False)
        self.deterministic_train_aib = getattr(args, 'deterministic_train_aib', False)
        self.ib_std_offset = float(getattr(args, 'ib_std_offset', -5.0))
        def resolve_std_offset(name):
            value = getattr(args, f'ib_std_offset_{name}', None)
            return self.ib_std_offset if value is None else float(value)
        self.ib_std_offsets = {
            name: resolve_std_offset(name) for name in ('a', 't', 'v', 'common', 'fusion')
        }
        self.adaptive_multi_ib = bool(getattr(args, 'adaptive_multi_ib', False))
        self.multi_ib_std_offsets = {
            name: getattr(args, f'multi_ib_std_offset_{name}', None)
            for name in ('a', 't', 'v', 'common', 'fusion')
        }
        self.normalize_pair_fusion = bool(getattr(args, 'normalize_pair_fusion', False))
        self.uni_aib_residual_alpha = getattr(args, 'uni_aib_residual_alpha', 1.0)
        self.second_stage_uni_decoder = getattr(args, 'second_stage_uni_decoder', False)
        self.single_condition_heads = getattr(args, 'single_condition_heads', False)
        self.condition_denoise_gates = getattr(args, 'condition_denoise_gates', False)
        self.condition_gate_reg_lambda = getattr(args, 'condition_gate_reg_lambda', 0.0)
        self.condition_adapter = getattr(args, 'condition_adapter', False)
        self.condition_adapter_reg_lambda = getattr(args, 'condition_adapter_reg_lambda', 0.0)
        self.condition_generated_adapter = getattr(args, 'condition_generated_adapter', False)
        self.adapter_bank_reg_lambda = getattr(args, 'adapter_bank_reg_lambda', 0.0)
        self.scrb_aux_aib = getattr(args, 'scrb_aux_aib', False)
        self.last_fusion_residual_gate = None
        if self.condition_adapter and self.condition_generated_adapter:
            raise ValueError('--condition-adapter and --condition-generated-adapter are mutually exclusive')
        self.condition_to_idx = {'a': 0, 't': 1, 'v': 2, 'at': 3, 'av': 4, 'tv': 5, 'atv': 6}
        self.modality_to_idx = {'a': 0, 't': 1, 'v': 2}
        self.condition_members = {
            condition: [self.modality_to_idx[modality] for modality in self.modality_to_idx if modality in condition]
            for condition in self.condition_to_idx
        }
        
        self.as_in_proj = nn.Conv1d(self.adim, D_e, kernel_size=1, padding=0, bias=False)
        self.ts_in_proj = nn.Conv1d(self.tdim, D_e, kernel_size=1, padding=0, bias=False)
        self.vs_in_proj = nn.Conv1d(self.vdim, D_e, kernel_size=1, padding=0, bias=False)
        
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        self.intra_attn_as = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)
        self.intra_attn_ts = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)
        self.intra_attn_vs = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)
        
        self.inter_attn_common = Attention(dim=self.D_e,num_heads=self.num_heads,mlp_ratio=mlp_ratio,attn_drop=attn_drop_rate)

        self.as_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes, deterministic_eval=self.deterministic_eval_aib, deterministic_train=self.deterministic_train_aib, std_offset=self.ib_std_offsets['a'])
        self.ts_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes, deterministic_eval=self.deterministic_eval_aib, deterministic_train=self.deterministic_train_aib, std_offset=self.ib_std_offsets['t'])
        self.vs_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes, deterministic_eval=self.deterministic_eval_aib, deterministic_train=self.deterministic_train_aib, std_offset=self.ib_std_offsets['v'])

        self.as_ts_vs_common_vib = VIB(dim=self.D_e, encoding_dim=self.D_e, output_dim=n_classes, deterministic_eval=self.deterministic_eval_aib, deterministic_train=self.deterministic_train_aib, std_offset=self.ib_std_offsets['common'])
        if self.modality_prior:
            self.modality_prior_mu = nn.Parameter(torch.zeros(3, D_e))
            self.modality_prior_logstd = nn.Parameter(torch.zeros(3, D_e))
        if self.fusion_aib:
            rng_state = torch.random.get_rng_state()
            self.fusion_vib = VIB(
                dim=D,
                encoding_dim=D,
                output_dim=n_classes,
                deterministic_eval=self.deterministic_eval_aib,
                deterministic_train=self.deterministic_train_aib or self.DDIB_fusion_deterministic_train,
                std_offset=self.ib_std_offsets['fusion'],
            )
            projected_modes = {
                'projected_add',
                'gated_projected_add',
                'gated_lowrank_add',
                'delta_projected_add',
                'gated_delta_projected_add',
                'gated_delta_lowrank_add',
            }
            gated_modes = {
                'gated_projected_add',
                'gated_lowrank_add',
                'gated_delta_projected_add',
                'gated_delta_lowrank_add',
            }
            lowrank_modes = {'gated_lowrank_add', 'gated_delta_lowrank_add'}
            if self.DDIB_fusion and self.DDIB_fusion_residual_mode in projected_modes:
                self.fusion_residual_norm = nn.LayerNorm(D)
                if self.DDIB_fusion_residual_mode in lowrank_modes:
                    rank = max(1, int(getattr(args, 'DDIB_fusion_residual_rank', max(4, D // 16))))
                    self.fusion_residual_down = nn.Linear(D, rank, bias=False)
                    self.fusion_residual_up = nn.Linear(rank, D, bias=False)
                    nn.init.normal_(self.fusion_residual_down.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(self.fusion_residual_up.weight)
                else:
                    self.fusion_residual_proj = nn.Linear(D, D)
                    nn.init.zeros_(self.fusion_residual_proj.weight)
                    nn.init.zeros_(self.fusion_residual_proj.bias)
            if self.DDIB_fusion and self.DDIB_fusion_residual_mode in gated_modes:
                self.fusion_gate_norm = nn.LayerNorm(2 * D)
                self.fusion_residual_gate = nn.Linear(2 * D, 1)
                nn.init.zeros_(self.fusion_residual_gate.weight)
                gate_init = min(1.0 - 1e-4, max(1e-4, self.DDIB_fusion_gate_init))
                nn.init.constant_(self.fusion_residual_gate.bias, np.log(gate_init / (1.0 - gate_init)))
            if self.DDIB_fusion and self.DDIB_fusion_deterministic_eval:
                self.fusion_vib.deterministic_eval = True
            if self.condition_prior:
                self.condition_prior_mu = nn.Parameter(torch.zeros(len(self.condition_to_idx), D))
                self.condition_prior_logstd = nn.Parameter(torch.zeros(len(self.condition_to_idx), D))
            torch.random.set_rng_state(rng_state)
        
        self.proj1 = nn.Linear(D, D)
        self.proj2 = nn.Linear(D, D)
        self.nlp_head = nn.Linear(D, n_classes)
        if self.condition_denoise_gates:
            # Gates are indexed by condition, modality, and representation branch.
            self.condition_gate_values = nn.Parameter(torch.ones(len(self.condition_to_idx), 3, 2))
        if self.condition_adapter:
            rank = max(1, int(getattr(args, 'condition_adapter_rank', 4)))
            self.condition_adapter_down = nn.ModuleList([nn.Linear(D, rank, bias=False) for _ in self.condition_to_idx])
            self.condition_adapter_up = nn.ModuleList([nn.Linear(rank, D, bias=False) for _ in self.condition_to_idx])
            for down, up in zip(self.condition_adapter_down, self.condition_adapter_up):
                nn.init.normal_(down.weight, mean=0.0, std=0.02)
                nn.init.zeros_(up.weight)
        if self.condition_generated_adapter:
            emb_dim = max(1, int(getattr(args, 'condition_embedding_dim', 32)))
            bank_size = max(1, int(getattr(args, 'adapter_bank_size', 4)))
            bank_rank = max(1, int(getattr(args, 'adapter_bank_rank', 8)))
            self.condition_modality_embedding = nn.Embedding(len(self.modality_to_idx), emb_dim)
            self.condition_cardinality_embedding = nn.Embedding(len(self.modality_to_idx) + 1, emb_dim)
            self.adapter_bank_pi = nn.Linear(emb_dim, bank_size)
            self.adapter_bank_eta = nn.Linear(emb_dim, 1)
            self.adapter_bank_down = nn.ModuleList([nn.Linear(D, bank_rank, bias=False) for _ in range(bank_size)])
            self.adapter_bank_up = nn.ModuleList([nn.Linear(bank_rank, D, bias=False) for _ in range(bank_size)])
            for down, up in zip(self.adapter_bank_down, self.adapter_bank_up):
                nn.init.normal_(down.weight, mean=0.0, std=0.02)
                nn.init.zeros_(up.weight)
            eta_init = float(getattr(args, 'adapter_eta_init', 0.05))
            eta_init = min(1.0 - 1e-4, max(1e-4, eta_init))
            nn.init.zeros_(self.adapter_bank_eta.weight)
            nn.init.constant_(self.adapter_bank_eta.bias, np.log(eta_init / (1.0 - eta_init)))
        if self.scrb_aux_aib:
            scrb_emb_dim = max(1, int(getattr(args, 'scrb_embedding_dim', getattr(args, 'condition_embedding_dim', 32))))
            self.scrb_modality_embedding = nn.Embedding(len(self.modality_to_idx), scrb_emb_dim)
            self.scrb_cardinality_embedding = nn.Embedding(len(self.modality_to_idx) + 1, scrb_emb_dim)
            self.scrb_context_proj = nn.Linear(scrb_emb_dim, D_e)
            self.scrb_vib = VIB(
                dim=self.D_e,
                encoding_dim=self.D_e,
                output_dim=n_classes,
                deterministic_eval=True,
                deterministic_train=False,
                std_offset=self.ib_std_offset,
            )
        if self.single_condition_heads:
            self.single_heads = nn.ModuleDict({
                'a': nn.Linear(D, n_classes),
                't': nn.Linear(D, n_classes),
                'v': nn.Linear(D, n_classes),
            })
        self.classifier_a = nn.Linear(D_e, n_classes)
        self.classifier_t = nn.Linear(D_e, n_classes)
        self.classifier_v = nn.Linear(D_e, n_classes)

        self.classifier_a_common = nn.Linear(D_e, n_classes)
        self.classifier_t_common = nn.Linear(D_e, n_classes)
        self.classifier_v_common = nn.Linear(D_e, n_classes)
        if self.second_stage_uni_decoder:
            self.second_decoder_a = nn.Linear(D_e, n_classes)
            self.second_decoder_t = nn.Linear(D_e, n_classes)
            self.second_decoder_v = nn.Linear(D_e, n_classes)
        if self.DDIB_uni:
            self.DDIB_aux_head_a = nn.Linear(D_e, n_classes)
            self.DDIB_aux_head_t = nn.Linear(D_e, n_classes)
            self.DDIB_aux_head_v = nn.Linear(D_e, n_classes)
            self.DDIB_aux_common_head = nn.Linear(D_e, n_classes)
            self.DDIB_z_aux_head_a = nn.Linear(D_e, n_classes)
            self.DDIB_z_aux_head_t = nn.Linear(D_e, n_classes)
            self.DDIB_z_aux_head_v = nn.Linear(D_e, n_classes)
            self.DDIB_z_aux_common_head = nn.Linear(D_e, n_classes)

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        """Run the model on sequence-first multimodal features and validity masks."""
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape
        
        # Project each modality to the shared hidden dimension.
        audio, text, video = audio.permute(1, 2, 0), text.permute(1, 2, 0), video.permute(1, 2, 0)
        proj_a = self.dropout_a(self.as_in_proj(audio))
        proj_t = self.dropout_t(self.ts_in_proj(text))
        proj_v = self.dropout_v(self.vs_in_proj(video))
        

        # Build modality-specific validity masks.
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        valid_a = input_mask[:, :, 0].bool()
        valid_t = input_mask[:, :, 1].bool()
        valid_v = input_mask[:, :, 2].bool()
        valid_utt = umask.bool()
        proj_a, proj_t, proj_v = proj_a.permute(0, 2, 1), proj_t.permute(0, 2, 1), proj_v.permute(0, 2, 1)

        disable_msd = self.ablation in ['wo-msd', 'wo-imd']
        disable_mcd = self.ablation in ['wo-mcd', 'wo-imd']
        zero_kls = proj_a.new_tensor(0.0)

        def active_std_offset(name):
            value = self.multi_ib_std_offsets[name]
            if first_stage or not self.adaptive_multi_ib or value is None:
                return self.ib_std_offsets[name]
            return float(value) if len(self._condition_modalities()) >= 2 else self.ib_std_offsets[name]

        # Modality-specific bottlenecks.
        if disable_msd:
            mu_a = std_a = mu_t = std_t = mu_v = std_v = None
            denoise_a, denoise_t, denoise_v = proj_a, proj_t, proj_v
            z_a, z_t, z_v = denoise_a, denoise_t, denoise_v
            out_a = proj_a.new_zeros(B, seq_len, self.n_classes)
            out_t = proj_t.new_zeros(B, seq_len, self.n_classes)
            out_v = proj_v.new_zeros(B, seq_len, self.n_classes)
            a_kls, t_kls, v_kls = zero_kls, zero_kls, zero_kls
        else:
            prior_mu_a, prior_std_a = self._modality_prior(0)
            prior_mu_t, prior_std_t = self._modality_prior(1)
            prior_mu_v, prior_std_v = self._modality_prior(2)
            (mu_a, std_a), denoise_a, out_a, a_kls = self.as_vib(proj_a, prior_mu=prior_mu_a, prior_std=prior_std_a, valid_mask=valid_a, std_offset=active_std_offset('a'))
            (mu_t, std_t), denoise_t, out_t, t_kls = self.ts_vib(proj_t, prior_mu=prior_mu_t, prior_std=prior_std_t, valid_mask=valid_t, std_offset=active_std_offset('t'))
            (mu_v, std_v), denoise_v, out_v, v_kls = self.vs_vib(proj_v, prior_mu=prior_mu_v, prior_std=prior_std_v, valid_mask=valid_v, std_offset=active_std_offset('v'))
            z_a, z_t, z_v = denoise_a, denoise_t, denoise_v
            denoise_a = self._residual_aib(proj_a, denoise_a)
            denoise_t = self._residual_aib(proj_t, denoise_t)
            denoise_v = self._residual_aib(proj_v, denoise_v)

        # Shared bottleneck.
        if disable_mcd:
            mu_a_common = std_a_common = mu_t_common = std_t_common = mu_v_common = std_v_common = None
            denoise_a_common, denoise_t_common, denoise_v_common = proj_a, proj_t, proj_v
            z_a_common, z_t_common, z_v_common = denoise_a_common, denoise_t_common, denoise_v_common
            out_a_common = proj_a.new_zeros(B, seq_len, self.n_classes)
            out_t_common = proj_t.new_zeros(B, seq_len, self.n_classes)
            out_v_common = proj_v.new_zeros(B, seq_len, self.n_classes)
            a_kls_common, t_kls_common, v_kls_common = zero_kls, zero_kls, zero_kls
        else:
            prior_mu_a, prior_std_a = self._modality_prior(0)
            prior_mu_t, prior_std_t = self._modality_prior(1)
            prior_mu_v, prior_std_v = self._modality_prior(2)
            common_std_offset = active_std_offset('common')
            (mu_a_common, std_a_common), denoise_a_common, out_a_common, a_kls_common = self.as_ts_vs_common_vib(proj_a, prior_mu=prior_mu_a, prior_std=prior_std_a, valid_mask=valid_a, std_offset=common_std_offset)
            (mu_t_common, std_t_common), denoise_t_common, out_t_common, t_kls_common = self.as_ts_vs_common_vib(proj_t, prior_mu=prior_mu_t, prior_std=prior_std_t, valid_mask=valid_t, std_offset=common_std_offset)
            (mu_v_common, std_v_common), denoise_v_common, out_v_common, v_kls_common = self.as_ts_vs_common_vib(proj_v, prior_mu=prior_mu_v, prior_std=prior_std_v, valid_mask=valid_v, std_offset=common_std_offset)
            z_a_common, z_t_common, z_v_common = denoise_a_common, denoise_t_common, denoise_v_common
            denoise_a_common = self._residual_aib(proj_a, denoise_a_common)
            denoise_t_common = self._residual_aib(proj_t, denoise_t_common)
            denoise_v_common = self._residual_aib(proj_v, denoise_v_common)
        
        # Modality-specific and shared interactions.
        intra_as = self.intra_attn_as(denoise_a, denoise_a, valid_mask=valid_a)
        intra_ts = self.intra_attn_ts(denoise_t, denoise_t, valid_mask=valid_t)
        intra_vs = self.intra_attn_vs(denoise_v, denoise_v, valid_mask=valid_v)

        inter_as = self.inter_attn_common(denoise_a_common, denoise_a_common, valid_mask=valid_a)
        inter_ts = self.inter_attn_common(denoise_t_common, denoise_t_common, valid_mask=valid_t)
        inter_vs = self.inter_attn_common(denoise_v_common, denoise_v_common, valid_mask=valid_v)

        outputs_intra_a = self.classifier_a(intra_as)
        outputs_intra_t = self.classifier_t(intra_ts)
        outputs_intra_v = self.classifier_v(intra_vs)

        outputs_inter_a = self.classifier_a_common(inter_as)
        outputs_inter_t = self.classifier_t_common(inter_ts)
        outputs_inter_v = self.classifier_v_common(inter_vs)

        def add_vib_stats(prefix, mu, std, kl):
            if mu is None or std is None:
                return {
                    f'{prefix}_kl': zero_kls.detach(),
                    f'{prefix}_post_var': zero_kls.detach(),
                    f'{prefix}_mu_abs': zero_kls.detach(),
                }
            return {
                f'{prefix}_kl': kl.detach(),
                f'{prefix}_post_var': std.detach().pow(2).mean(),
                f'{prefix}_mu_abs': mu.detach().abs().mean(),
            }

        self.last_vib_stats = {}
        self.last_vib_stats.update(add_vib_stats('a', mu_a, std_a, a_kls))
        self.last_vib_stats.update(add_vib_stats('t', mu_t, std_t, t_kls))
        self.last_vib_stats.update(add_vib_stats('v', mu_v, std_v, v_kls))
        self.last_vib_stats.update(add_vib_stats('a_common', mu_a_common, std_a_common, a_kls_common))
        self.last_vib_stats.update(add_vib_stats('t_common', mu_t_common, std_t_common, t_kls_common))
        self.last_vib_stats.update(add_vib_stats('v_common', mu_v_common, std_v_common, v_kls_common))
        self.last_unimodal_reps = {
            'a': denoise_a,
            't': denoise_t,
            'v': denoise_v,
        }
        vib_kls_for_loss = [a_kls, t_kls, v_kls, a_kls_common, t_kls_common, v_kls_common]
        (denoise_a, denoise_t, denoise_v), (denoise_a_common, denoise_t_common, denoise_v_common) = self._apply_condition_denoise_gates(
            [proj_a, proj_t, proj_v],
            [denoise_a, denoise_t, denoise_v],
            [denoise_a_common, denoise_t_common, denoise_v_common],
            first_stage,
        )
        self.last_DDIB_aux_logits = {}
        self.last_DDIB_z_aux_logits = {}
        if self.DDIB_uni and self.training and not first_stage:
            self.last_DDIB_aux_logits = {
                'a': (self.DDIB_aux_head_a(denoise_a), self.DDIB_aux_common_head(denoise_a_common)),
                't': (self.DDIB_aux_head_t(denoise_t), self.DDIB_aux_common_head(denoise_t_common)),
                'v': (self.DDIB_aux_head_v(denoise_v), self.DDIB_aux_common_head(denoise_v_common)),
            }
            self.last_DDIB_z_aux_logits = {
                'a': (self.DDIB_z_aux_head_a(z_a), self.DDIB_z_aux_common_head(z_a_common)),
                't': (self.DDIB_z_aux_head_t(z_t), self.DDIB_z_aux_common_head(z_t_common)),
                'v': (self.DDIB_z_aux_head_v(z_v), self.DDIB_z_aux_common_head(z_v_common)),
            }

        def build_second_stage_slots(spe_a, spe_t, spe_v, common_a, common_t, common_v):
            pair_scale = 0.5 if self.normalize_pair_fusion else 1.0
            if self.ablation == 'wo-imc':
                return torch.cat([spe_a, spe_t, spe_v], dim=1)
            if self.test_condition == 't':
                fusd_spe_ts = self.intra_attn_ts(spe_t, common_t, valid_mask=valid_utt)
                fusd_common_as = self.inter_attn_common(common_t, spe_t, valid_mask=valid_utt)
                fusd_common_vs = self.inter_attn_common(common_t, spe_t, valid_mask=valid_utt)
                return torch.cat([fusd_common_as, fusd_spe_ts, fusd_common_vs], dim=1)
            if self.test_condition == 'a':
                fusd_spe_as = self.intra_attn_as(spe_a, common_a, valid_mask=valid_utt)
                fusd_common_ts = self.inter_attn_common(common_a, spe_a, valid_mask=valid_utt)
                fusd_common_vs = self.inter_attn_common(common_a, spe_a, valid_mask=valid_utt)
                return torch.cat([fusd_spe_as, fusd_common_ts, fusd_common_vs], dim=1)
            if self.test_condition == 'v':
                fusd_spe_vs = self.intra_attn_vs(spe_v, common_v, valid_mask=valid_utt)
                fusd_common_as = self.inter_attn_common(common_v, spe_v, valid_mask=valid_utt)
                fusd_common_ts = self.inter_attn_common(common_v, spe_v, valid_mask=valid_utt)
                return torch.cat([fusd_common_as, fusd_common_ts, fusd_spe_vs], dim=1)
            if self.test_condition == 'at':
                fusd_spe_as = self.intra_attn_as(spe_a, common_a, valid_mask=valid_utt)
                fusd_spe_ts = self.intra_attn_ts(spe_t, common_t, valid_mask=valid_utt)
                fusd_common_vs = pair_scale * (self.inter_attn_common(common_t, spe_a, valid_mask=valid_utt) + self.inter_attn_common(common_a, spe_t, valid_mask=valid_utt))
                return torch.cat([fusd_spe_as, fusd_spe_ts, fusd_common_vs], dim=1)
            if self.test_condition == 'av':
                fusd_spe_as = self.intra_attn_as(spe_a, common_a, valid_mask=valid_utt)
                fusd_common_ts = pair_scale * (self.inter_attn_common(common_v, spe_a, valid_mask=valid_utt) + self.inter_attn_common(common_a, spe_v, valid_mask=valid_utt))
                fusd_spe_vs = self.intra_attn_vs(spe_v, common_v, valid_mask=valid_utt)
                return torch.cat([fusd_spe_as, fusd_common_ts, fusd_spe_vs], dim=1)
            if self.test_condition == 'tv':
                fusd_common_as = pair_scale * (self.inter_attn_common(common_t, spe_v, valid_mask=valid_utt) + self.inter_attn_common(common_v, spe_t, valid_mask=valid_utt))
                fusd_spe_ts = self.intra_attn_ts(spe_t, common_t, valid_mask=valid_utt)
                fusd_spe_vs = self.intra_attn_vs(spe_v, common_v, valid_mask=valid_utt)
                return torch.cat([fusd_common_as, fusd_spe_ts, fusd_spe_vs], dim=1)
            fusd_spe_ts = self.intra_attn_ts(spe_t, common_t, valid_mask=valid_utt)
            fusd_spe_as = self.intra_attn_as(spe_a, common_a, valid_mask=valid_utt)
            fusd_spe_vs = self.intra_attn_vs(spe_v, common_v, valid_mask=valid_utt)
            return torch.cat([fusd_spe_as, fusd_spe_ts, fusd_spe_vs], dim=1)
        
        if first_stage:
            x = torch.cat([denoise_a, denoise_t, denoise_v], dim=1)# [32, 189, 256]
        else:
            pair_scale = 0.5 if self.normalize_pair_fusion else 1.0
            out_a = out_t = out_v = out_a_common = out_t_common = out_v_common = proj_a.new_zeros(B, seq_len, self.n_classes)
            outputs_intra_a = outputs_intra_t = outputs_intra_v = outputs_inter_a = outputs_inter_t = outputs_inter_v = proj_a.new_zeros(B, seq_len, self.n_classes)

            if self.ablation == 'wo-imc':
                x = torch.cat([denoise_a, denoise_t, denoise_v], dim=1)
            elif self.test_condition == 't':
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common, valid_mask=valid_utt)
                fusd_common_as = self.inter_attn_common(denoise_t_common, denoise_t, valid_mask=valid_utt)
                fusd_common_vs = self.inter_attn_common(denoise_t_common, denoise_t, valid_mask=valid_utt)
                all_fused = torch.cat([fusd_common_as, fusd_spe_ts, fusd_common_vs], dim=1)
                x = all_fused
                
            elif self.test_condition == 'a':
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common, valid_mask=valid_utt)
                fusd_common_ts = self.inter_attn_common(denoise_a_common, denoise_a, valid_mask=valid_utt)
                fusd_common_vs = self.inter_attn_common(denoise_a_common, denoise_a, valid_mask=valid_utt)
                all_fused = torch.cat([fusd_spe_as, fusd_common_ts, fusd_common_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'v':
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common, valid_mask=valid_utt)
                fusd_common_as = self.inter_attn_common(denoise_v_common, denoise_v, valid_mask=valid_utt)
                fusd_common_ts = self.inter_attn_common(denoise_v_common, denoise_v, valid_mask=valid_utt)
                all_fused = torch.cat([fusd_common_as, fusd_common_ts, fusd_spe_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'at':
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common, valid_mask=valid_utt)
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common, valid_mask=valid_utt)
                fusd_common_vs = pair_scale * (self.inter_attn_common(denoise_t_common, denoise_a, valid_mask=valid_utt) + self.inter_attn_common(denoise_a_common, denoise_t, valid_mask=valid_utt))
                all_fused = torch.cat([fusd_spe_as, fusd_spe_ts, fusd_common_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'av':
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common, valid_mask=valid_utt)
                fusd_common_ts = pair_scale * (self.inter_attn_common(denoise_v_common, denoise_a, valid_mask=valid_utt) + self.inter_attn_common(denoise_a_common, denoise_v, valid_mask=valid_utt))
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common, valid_mask=valid_utt)
                all_fused = torch.cat([fusd_spe_as, fusd_common_ts, fusd_spe_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'tv':
                fusd_common_as = pair_scale * (self.inter_attn_common(denoise_t_common, denoise_v, valid_mask=valid_utt) + self.inter_attn_common(denoise_v_common, denoise_t, valid_mask=valid_utt))
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common, valid_mask=valid_utt)
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common, valid_mask=valid_utt)
                all_fused = torch.cat([fusd_common_as, fusd_spe_ts, fusd_spe_vs], dim=1)
                x = all_fused
            elif self.test_condition == 'atv':
                fusd_spe_ts = self.intra_attn_ts(denoise_t, denoise_t_common, valid_mask=valid_utt)
                fusd_spe_as = self.intra_attn_as(denoise_a, denoise_a_common, valid_mask=valid_utt)
                fusd_spe_vs = self.intra_attn_vs(denoise_v, denoise_v_common, valid_mask=valid_utt)
                all_fused = torch.cat([fusd_spe_as, fusd_spe_ts, fusd_spe_vs], dim=1)
                x = all_fused


        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
        self.last_second_stage_uni_logits = {}
        if self.second_stage_uni_decoder and not first_stage:
            self.last_second_stage_uni_logits = {
                'a': self.second_decoder_a(x_a),
                't': self.second_decoder_t(x_t),
                'v': self.second_decoder_v(x_v),
            }
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        res = x_joint
        u = self.proj2(F.dropout(F.relu(self.proj1(res)), p=self.out_dropout, training=self.training))
        hidden = u + res
        hidden = self._apply_condition_adapter(hidden, first_stage)
        hidden = self._apply_condition_generated_adapter(hidden, first_stage)
        out = self.nlp_head(hidden)
        if self.single_condition_heads and not first_stage and self.test_condition in self.single_heads:
            out = self.single_heads[self.test_condition](hidden)
        self.last_pre_fusion_hidden = hidden
        self.last_pre_fusion_out = out
        self.last_scrb_aux_hidden = None
        self.last_scrb_aux_out = None
        self.last_scrb_rates = {}
        if self.scrb_aux_aib and self.training and not first_stage:
            spe_reps = {'a': denoise_a, 't': denoise_t, 'v': denoise_v}
            common_reps = {'a': denoise_a_common, 't': denoise_t_common, 'v': denoise_v_common}
            aux_spe = dict(spe_reps)
            aux_common = dict(common_reps)
            for modality in self._condition_modalities():
                context = self._scrb_context(modality).view(1, 1, -1)
                (_, _), aux_spe[modality], _, spe_kl = self.scrb_vib(spe_reps[modality] + context)
                (_, _), aux_common[modality], _, common_kl = self.scrb_vib(common_reps[modality] + context)
                rate = 0.5 * (spe_kl + common_kl)
                self.last_scrb_rates[modality] = rate
                self.last_vib_stats[f'scrb_{modality}_kl'] = rate.detach()

            aux_x = build_second_stage_slots(
                aux_spe['a'], aux_spe['t'], aux_spe['v'],
                aux_common['a'], aux_common['t'], aux_common['v'],
            )
            aux_x_a, aux_x_t, aux_x_v = aux_x[:, :seq_len, :], aux_x[:, seq_len:2*seq_len, :], aux_x[:, 2*seq_len:, :]
            aux_res = torch.cat([aux_x_a, aux_x_t, aux_x_v], dim=-1)
            aux_u = self.proj2(F.dropout(F.relu(self.proj1(aux_res)), p=self.out_dropout, training=self.training))
            aux_hidden = aux_u + aux_res
            aux_hidden = self._apply_condition_adapter(aux_hidden, first_stage)
            aux_hidden = self._apply_condition_generated_adapter(aux_hidden, first_stage)
            aux_out = self.nlp_head(aux_hidden)
            if self.single_condition_heads and self.test_condition in self.single_heads:
                aux_out = self.single_heads[self.test_condition](aux_hidden)
            self.last_scrb_aux_hidden = aux_hidden
            self.last_scrb_aux_out = aux_out
        self.last_fusion_kl = hidden.new_tensor(0.0)
        if self.fusion_aib and not first_stage:
            prior_mu_fusion, prior_std_fusion = self._condition_prior()
            (mu_fusion, std_fusion), fusion_hidden, fusion_out, fusion_kls = self.fusion_vib(
                hidden, prior_mu=prior_mu_fusion, prior_std=prior_std_fusion, valid_mask=valid_utt,
                std_offset=active_std_offset('fusion'),
            )
            if self.DDIB_fusion:
                hidden = self._residual_fusion_aib(hidden, fusion_hidden)
                out = self.nlp_head(hidden)
                if self.single_condition_heads and self.test_condition in self.single_heads:
                    out = self.single_heads[self.test_condition](hidden)
                if self.last_fusion_residual_gate is not None:
                    self.last_vib_stats['fusion_gate'] = self.last_fusion_residual_gate
            else:
                hidden, out = fusion_hidden, fusion_out
            self.last_fusion_kl = fusion_kls
            self.last_vib_stats.update(add_vib_stats('fusion', mu_fusion, std_fusion, fusion_kls))
  
        return hidden, out, [outputs_intra_a, outputs_intra_t, outputs_intra_v], [outputs_inter_a, outputs_inter_t, outputs_inter_v], [out_a, out_t, out_v, out_a_common, out_t_common, out_v_common], vib_kls_for_loss
         


if __name__ == '__main__':
    pass
