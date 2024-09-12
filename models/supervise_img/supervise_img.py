from copy import deepcopy
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
class supervise_img(BaseModule):
    def __init__(self, img_backbone, img_neck, voxel_height):
        super().__init__()
        self.img_backbone = HEADS.build(img_backbone)
        self.img_neck = HEADS.build(img_neck)
        upsample_in_channel = img_neck['out_channels']
        self.querys = nn.Embedding(voxel_height, upsample_in_channel)
        self.attention = nn.MultiheadAttention(upsample_in_channel, 1)

    def extract_img_feat(self, imgs):
        b, v, h, w, c = imgs.shape
        imgs = imgs.float()
        imgs = imgs.reshape(b * v, h, w, c) # batch * view, h, w, c
        imgs = imgs.permute(0, 3, 1, 2) # batch * view, c, h, w
        img_feats = self.img_backbone(imgs)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = img_feat.reshape(b, v, *img_feat.shape[1:])
            img_feats_reshaped.append(img_feat)
        return img_feats_reshaped


            

    def train_step(self, data, optimizer, cfg):
        imgs = data['cams'] # batch, 1, view, h, w, c
        imgs = rearrange(imgs, 'b 1 v h w c -> (b 1) v h w c') #batch, view, h, w, c
        img_feats = self.extract_img_feat(imgs) # list of batch, view, c, h, w
        occ = []
        for img_feat in img_feats:
            query = self.querys.weight.unsqueeze(1).repeat(1, img_feat.shape[1], 1) # batch, view, c
        

    
    def val_step(self, data, optimizer, **kwargs):
        pass