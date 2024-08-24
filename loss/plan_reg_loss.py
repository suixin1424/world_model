from mmdet.models import LOSSES
from torch import nn
import torch
import torch.nn.functional as F

@LOSSES.register_module()
class plan_reg_loss(nn.Module):
    def __init__(self, weight=1.0, num_modes=3, loss_type='l2'):
        super(plan_reg_loss, self).__init__()
        assert loss_type in ['l1', 'l2'], f'loss_type {loss_type} not supported'
        self.weight = weight
        self.num_modes = num_modes
        self.loss_type = loss_type
    
    def forward(self, pose_pred, pose_gt, mode_gt):
        bs, num_frames, num_modes, _ = pose_pred.shape #[1,9,3,2]
        
        mode_gt = mode_gt.transpose(1,2)
        pose_pred = pose_pred.transpose(1, 2) # B, M, F, 2
        pose_pred = torch.cumsum(pose_pred, -2)
        pose_gt = pose_gt.unsqueeze(1).repeat(1, num_modes, 1, 1) # B, M, F, 2
        pose_gt = torch.cumsum(pose_gt, -2)
            
        if self.loss_type == 'l1':
            weight = mode_gt[..., None].repeat(1, 1, 1, 2)
            loss = torch.abs(pose_pred - pose_gt) * weight
        elif self.loss_type == 'l2':
            weight = mode_gt # [..., None].repeat(1, 1, 1)
            loss = torch.sqrt(((pose_pred - pose_gt) ** 2).sum(-1)) * weight
        #loss = torch.abs(rel_pose - gt_rel_pose) * weight

        return loss.sum() / bs / num_frames