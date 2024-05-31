import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute Focal Loss.

        Args:
            inputs (Tensor): Predictions (logits), shape (N, C) where C is the number of classes.
            targets (Tensor): Ground truth, shape (N, C) in one-hot encoded form.

        Returns:
            loss (Tensor): Focal loss.
        """
        probs = F.softmax(inputs, dim=1)
        probs = (probs * targets).sum(dim=1)

        log_p = torch.log(probs)
        if self.alpha is not None:
            alpha = (self.alpha * targets).sum(dim=1)
            loss = -alpha * ((1 - probs) ** self.gamma) * log_p
        else:
            loss = -((1 - probs) ** self.gamma) * log_p
        return loss.mean()
