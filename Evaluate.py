import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json


class DepthSensitiveLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-6):
        """
        alpha: weight between WBCE and CWL (0 = only CWL, 1 = only WBCE)
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def weighted_bce(self, y_pred, y_true, depth_weights):
        # log-likelihood terms
        log_preds = torch.log(y_pred + self.epsilon)
        log_one_minus = torch.log(1 - y_pred + self.epsilon)

        bce = - (y_true * log_preds + (1 - y_true) * log_one_minus)
        weighted_bce = depth_weights * bce  # shape (B, N)
        return weighted_bce.mean()

    def continuity_weighted_loss(self, y_pred, y_true):
        """
        Rewards longer correct subsequences.
        Normalized by N (total rounds).
        """
        B, N = y_true.shape
        y_pred_bin = (y_pred > 0.5).float()
        correct = (y_pred_bin == y_true).float()

        # Longest contiguous subsequence of correct predictions
        max_streak = torch.zeros(B, device=y_true.device)
        for b in range(B):
            streak, max_run = 0, 0
            for val in correct[b]:
                if val == 1:
                    streak += 1
                    max_run = max(max_run, streak)
                else:
                    streak = 0
            max_streak[b] = max_run

        cwl = 1 - (max_streak / N)  # Loss = 1 - (reward ratio)
        return cwl.mean()

    def forward(self, y_pred, y_true, depth_weights):
        wbce = self.weighted_bce(y_pred, y_true, depth_weights)
        cwl = self.continuity_weighted_loss(y_pred, y_true)
        return self.alpha * wbce + (1 - self.alpha) * cwl