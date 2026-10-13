import copy
import math
from contextlib import contextmanager

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            if name not in self.shadow or not torch.is_floating_point(param):
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def apply_to(self, model):
        self.backup = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if name in self.shadow
        }
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None


class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr = float(min_lr)

    def step(self, epoch):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            scale = float(epoch + 1) / float(self.warmup_epochs)
            lr = self.base_lr * scale
        else:
            span = max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, float(epoch - self.warmup_epochs + 1) / float(span)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr


def build_scheduler(args, optimizer, total_epochs=None):
    if getattr(args, 'lr_scheduler', 'cosine') == 'none':
        return None
    return WarmupCosineScheduler(
        optimizer,
        base_lr=args.lr,
        total_epochs=args.epochs if total_epochs is None else total_epochs,
        warmup_epochs=getattr(args, 'warmup_epochs', 5),
        min_lr=getattr(args, 'min_lr', 1e-6),
    )


def build_ema(args, model):
    if not getattr(args, 'ema_eval', False):
        return None
    return ModelEMA(model, decay=getattr(args, 'ema_decay', 0.999))
