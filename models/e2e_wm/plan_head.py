from copy import deepcopy
import time
from mmdet.models import HEADS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from mmcv.runner import BaseModule
from mmdet.models import LOSSES

@HEADS.register_module()
class PlanHead(BaseModule):
    def __init__(self, idx_shape, in_channels, num_embeddings, embedding_dim):
        super().__init__()
        self.idx_shape = idx_shape
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.occ_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pose_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
    
    def forward(self, pose, occ_idx, occ_pred):
        '''
        pose: [1,9,512]
        occ_idx: [1,9,50,50]
        occ_pred: [1,9,512,50,50]
        '''
        occ_pred.requires_grad = True
        b = occ_idx.size(0)

        # spatial attention
        occ_idx = rearrange(occ_idx, 'b f h w -> (b f) (h w)') # [1*9, 50*50]
        occ_idx = self.occ_embedding(occ_idx.long()) # [9, 2500, 512]
        occ_pred = rearrange(occ_pred, 'b f c h w -> (b f) (h w) c') # [1*9, 50*50, 512]
        occ = self.spatial_attention(occ_idx, occ_pred, occ_pred)[0] # [9, 2500, 512]

        # pose attention
        pose = rearrange(pose, 'b f c -> (b f) c') # [1*9, 512]
        pose = pose.unsqueeze(1) # [9, 1, 512]
        query = torch.cat([pose, occ], dim=1) # [9, 2501, 512]
        pose = self.pose_attention(query, query, query)[0] #[9, 2501, 512]
        pose = query + pose
        pose = pose[:,0,:] # [9, 512]
        pose = rearrange(pose, '(b f) c -> b f c', b=b) # [1, 9, 512]
        return pose 
        





