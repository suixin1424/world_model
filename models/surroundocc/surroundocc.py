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
class surroundocc(BaseModule):
    def __init__(self, img_backbone, img_neck, pts_bbox_head):
        super(surroundocc, self).__init__()
        self.img_backbone = HEADS.build(img_backbone)
        self.img_neck = HEADS.build(img_neck)
        self.pts_bbox_head = HEADS.build(pts_bbox_head)
    
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
    
    def train_step(self, data, optimizer, **kwargs):
        output = {}
        occ_gt = data['occs'] # batch, 1, h, w, d
        lidar2img = data['lidar2img'] # batch, 1, view, 4, 4
        imgs = data['cams'] # batch, 1, view, h, w, c
        occ_gt = occ_gt.squeeze(1) # batch, h, w, d
        lidar2img = lidar2img.squeeze(1) # batch, view, 4, 4
        imgs = rearrange(imgs, 'b 1 v h w c -> (b 1) v h w c') #batch, view, h, w, c
        img_shape = imgs.shape[2:] # h, w, c
        img_feats = self.extract_img_feat(imgs)
        outs = self.pts_bbox_head(img_feats, lidar2img, img_shape)
        #occ = outs['occ_preds'][-1] # 1, 17, 200, 200, 16
        loss = self.pts_bbox_head.loss(occ_gt, outs)
        
        total_loss = 0
        for key in loss:
            total_loss += loss[key]
        output.update({
            'loss': total_loss
        })

        output.update(loss)

        return output
    
    def val_step(self, data, optimizer, **kwargs):
        output = {}
        occ_gt = data['occs'] # batch, 1, h, w, d
        lidar2img = data['lidar2img'] # batch, 1, view, 4, 4
        imgs = data['cams'] # batch, 1, view, h, w, c
        occ_gt = occ_gt.squeeze(1).to(torch.float32) # batch, h, w, d
        lidar2img = lidar2img.squeeze(1) # batch, view, 4, 4
        imgs = rearrange(imgs, 'b 1 v h w c -> (b 1) v h w c') #batch, view, h, w, c
        img_shape = imgs.shape[2:] # h, w, c
        img_feats = self.extract_img_feat(imgs)
        outs = self.pts_bbox_head(img_feats, lidar2img, img_shape)
        occ_pred = outs['occ_preds'][-1] # 1, 17, 200, 200, 16
        occ_pred = occ_pred.argmax(1) # 1, 200, 200, 16
        occ_pred = occ_pred.unsqueeze(1)
        occ_gt = occ_gt.unsqueeze(1)

        occ_m_gt = deepcopy(occ_gt)
        occ_m = deepcopy(occ_pred)
        occ_m_gt[occ_m_gt != 17] = 1
        occ_m_gt[occ_m_gt == 17] = 0
        occ_m[occ_m != 17] = 1
        occ_m[occ_m == 17] = 0
        output.update({
            'occ_rec': occ_pred,
            'occ_gt': occ_gt,
            'occ_m' : occ_m,
            'occ_m_gt' : occ_m_gt
        })
        return output
        

        