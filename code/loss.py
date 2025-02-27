import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, similarity, labels):
        # For similar pairs (labels==1) we want similarity to be close to 1.
        loss_same = (1 - similarity) ** 2  
        # For dissimilar pairs (labels==0), we want similarity to be below the margin.
        loss_diff = (F.relu(self.margin - similarity)) ** 2  
        loss = labels * loss_same + (1 - labels) * loss_diff
        return loss.mean()
