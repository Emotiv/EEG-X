import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def get_loss_module():
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class NoFussCrossEntropyLoss_balanced(nn.CrossEntropyLoss):
    """
    A less fussy CrossEntropyLoss:
    - Automatically handles Long (int64) targets
    - Handles arbitrary-shaped inputs
    - Supports optional class-balanced weighting
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inp, target):
        # Apply class balancing if no weights are provided
        if self.weight is None:
            with torch.no_grad():
                # Flatten target and compute class frequencies
                target_flat = target.view(-1)
                class_counts = torch.bincount(target_flat, minlength=inp.size(1)).float()
                # Avoid division by zero
                class_counts[class_counts == 0] = 1
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum() * len(class_counts)
                self.weight = class_weights.to(inp.device)

        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

