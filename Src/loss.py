import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCELoss(nn.Module):

    def __init__(self, weight=None, label_smoothing=0.0):
        super(MaskedCELoss, self).__init__()
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, pred, target, umask, mask_m=None, first_stage=True):
        mask = umask.view(-1).float()
        if mask_m is not None:
            mask = mask * mask_m.reshape(-1).float()
        target = target.view(-1).long()
        log_prob = F.log_softmax(pred, dim=1)
        nll = -log_prob.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.label_smoothing > 0:
            smooth = -log_prob.mean(dim=1)
            loss = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            loss = nll

        if self.weight is not None:
            loss = loss * self.weight.to(pred.device).gather(0, target)
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom


# Masked regression loss.
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()

        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]

        loss = self.loss(pred*mask, target*mask) / torch.sum(mask)

        return loss
