import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Compute Dice Loss.

        Args:
            pred (Tensor): Predictions (logits), shape (N, C) where C is the number of classes.
            target (Tensor): Ground truth, shape (N, C) in one-hot encoded form.

        Returns:
            loss (Tensor): Dice loss.
        """
        # Apply softmax to predictions to get probabilities
        pred = F.softmax(pred, dim=1)

        # Compute dice score
        intersection = (pred * target).sum(dim=1)
        dice_score = (2. * intersection + self.epsilon) / (pred.sum(dim=1) + target.sum(dim=1) + self.epsilon)

        # Compute dice loss
        loss = 1 - dice_score.mean()
        return loss
