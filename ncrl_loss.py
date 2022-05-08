import torch
import torch.nn as nn
import torch.nn.functional as F

class NCRLoss(nn.Module):
    def __init__(self, shift=0.0, isReg=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.isReg = isReg
        self.eps = eps
        self.reduction = reduction

    def compute_CE(self, x, y):
        """
        Adapted from "Asymmetric Loss For Multi-Label Classification"
        https://arxiv.org/abs/2009.14119
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Margin Shifting
        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()

    def forward(self, logits, labels):
        # Logit margin for pre-defined relations 
        rel_margin = logits[:,1:] - logits[:,0].unsqueeze(1)
        loss = self.compute_CE(rel_margin.float(), labels[:,1:].float())
        
        if self.isReg: # Enable margin regularization
            # Logit margin for the none class label
            na_margin = logits[:,0] - logits[:,1:].mean(-1) 
            loss += self.compute_CE(na_margin.float(), labels[:,0].float())
        
        if self.reduction == "mean":
            loss /= labels.shape[0]

        return loss

    def get_label(self, logits, num_labels=-1):
        """Copied from https://github.com/wzhouad/ATLOP/blob/main/losses.py#L32"""
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
