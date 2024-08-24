from mmdet.models import LOSSES
from torch import nn
import torch
import torch.nn.functional as F

@LOSSES.register_module()
class rec_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, labels):
        rec_loss = F.cross_entropy(x.permute(0, 5, 1, 2, 3, 4), labels)
        return rec_loss
