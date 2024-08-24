from mmdet.models import LOSSES
from torch import nn
import torch
import torch.nn.functional as F

@LOSSES.register_module()
class ce_loss(nn.Module):
    def __init__(self, weight=1.0):
        super(ce_loss, self).__init__()
        self.weight = weight

    def forward(self, ce_inputs, ce_labels):
        return self.weight*F.cross_entropy(ce_inputs, ce_labels)