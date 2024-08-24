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
class TransVQVAE(BaseModule):
    def __init__(self, predict_frame_len, h_occ, h_occ_encodered, vqvae, transformer, pose_encoder, pose_decoder, loss):
        super(TransVQVAE, self).__init__()

        self.predict_frame_len = predict_frame_len
        self.vqvae = HEADS.build(vqvae)
        self.transformer = HEADS.build(transformer)
        self.pose_encoder = HEADS.build(pose_encoder)
        self.pose_decoder = HEADS.build(pose_decoder)
        self.plan_attention = nn.MultiheadAttention(embed_dim=h_occ_encodered, num_heads=4)
        self.modify_attention = nn.MultiheadAttention(embed_dim=h_occ, num_heads=1)
        if loss.type == 'multi_loss':
            self.losses = {
                key: LOSSES.build(value) for key, value in loss.items() if key != 'type'
            }
        else:
            self.loss = LOSSES.build(loss)
    
    def get_pose(self, pose, mode):
        pose_label = pose[:, self.predict_frame_len:] # [1,9,2]
        gt_mode = mode[:, self.predict_frame_len:] # [1,9,3]

        pose_input = torch.cat([pose, mode], dim=-1) # [1,10,5]
        pose_input = pose_input[:,:pose_input.shape[1]-self.predict_frame_len] # [1,9,5]

        #forward pose encoder
        pose = self.pose_encoder(pose_input.float())
        return pose, pose_label, gt_mode


    def forward_plan(self, rel_poses, occ, z_q_predict, decoder_shape, occ_shape):
        '''
        rel_poses: [1,9,128]
        occ: [1,9,128,50,50]
        z_q_predict: [1,9,512,50,50]
        '''
        shape = deepcopy(decoder_shape)

        b, f, d = rel_poses.shape
        rel_poses = rel_poses.flatten(0, 1) #[9,128]
        rel_poses = rel_poses.unsqueeze(1).repeat(1, 50*50, 1) #[9,2500,128]
        occ = rearrange(occ, 'b f d h w -> (b f) (h w) d') #[9,2500,128]
        attn_output, attn_output_weights = self.plan_attention(rel_poses, occ, occ)

        rel_poses = rel_poses + attn_output #[9,2500,128]
        # [9,2500,128] -> [9,40000,16]
        rel_poses = F.avg_pool1d(rel_poses, 8, 8) #[9,2500,16]
        rel_poses = rel_poses.repeat(1, 16, 1) #[9,40000,16]

        z_q_predict = z_q_predict.flatten(0, 1) #[9,512,50,50]
        z_q_predict = z_q_predict.argmax(dim=1)

        z_q_predict = self.vqvae.vector_quantizer.get_codebook_entry(z_q_predict)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vqvae.vector_quantizer.post_quant_conv(z_q_predict)
        z_q_predict = self.vqvae.forward_decoder(z_q_predict, shape, occ_shape)
        occ_rec = z_q_predict.argmax(dim=-1).detach() #[1,9,200,200,16]
        occ_rec = rearrange(occ_rec, 'b f h w d -> (b f) (h w) d') #[9,200*200,16]
        occ_rec = occ_rec.float()
        occ_rec.requires_grad = True

        pose, _ = self.modify_attention(rel_poses, occ_rec, occ_rec)
        pose = pose + rel_poses
        pose = torch.mean(pose, dim=1) #[9,16]
        pose = rearrange(pose, '(b f) d -> b f d', b=b) #[1,9,16]
        #upsample pose to [1,9,128]
        pose = F.interpolate(pose, size=d, mode='linear', align_corners=False) #[1,9,128]
        return pose


        

    def train_step(self, data, optimizer, **kwargs):
        occ = data['occs']
        occ_gt = occ[:, self.predict_frame_len:]
        b, f, h, w, d = occ.shape # [1,10,200,200,16]
        
        output = {}
        #forward vqvae encoder
        z, shape = self.vqvae.forward_encoder(occ) 
        z = self.vqvae.vector_quantizer.quant_conv(z)
        z, vq_loss, ids = self.vqvae.vector_quantizer.forward_quantizer(z) #[10,128,50,50] [25000]
        ids = rearrange(ids, '(b f h w) -> b f h w', b=b, f=f, h=z.shape[-2], w=z.shape[-1]) #[1,10,50,50]
        ids_label = ids[:, self.predict_frame_len:].detach().flatten(0,1) #[9,50,50]

        #forward transformer
        z = rearrange(z, '(b f) d h w -> b f d h w', b=b, f=f) #[1,10,128,50,50]
        rel_poses, pose_label, gt_mode = self.get_pose(data['pose'], data['gt_mode']) #[1,9,128]
        z_q_predict = self.transformer(z[:, :z.shape[1]-self.predict_frame_len]) #[1,9,512,50,50]
        
        
        #predict pose
        pred_pose = self.forward_plan(rel_poses, z[:, :z.shape[1]-self.predict_frame_len], z_q_predict.clone().detach(), shape, occ_gt.shape)

        #forward pose decoder
        pose_decoded = self.pose_decoder(pred_pose)

        #forward loss
        z_q_predict = z_q_predict.flatten(0, 1) #[9,512,50,50]
        ce_loss = self.losses['ce_loss'](z_q_predict, ids_label)
        plan_reg_loss = self.losses['plan_reg_loss'](pose_decoded, pose_label, gt_mode)
        loss = ce_loss + plan_reg_loss

        output.update({
            'loss': loss,
            'ce_loss': ce_loss,
            'plan_reg_loss': plan_reg_loss
        })

        return output
    
    def val_step(self, data, optimizer=None, **kwargs):
        occ = data['occs']
        occ_gt = occ[:, self.predict_frame_len:] # [1,9,200,200,16]
        b, f, h, w, d = occ.shape # [1,10,200,200,16]
        
        output = {}
        #forward vqvae encoder
        z, shape = self.vqvae.forward_encoder(occ) 
        z = self.vqvae.vector_quantizer.quant_conv(z)
        z, vq_loss, ids = self.vqvae.vector_quantizer.forward_quantizer(z) #[10,128,50,50] [25000]
        ids = rearrange(ids, '(b f h w) -> b f h w', b=b, f=f, h=z.shape[-2], w=z.shape[-1]) #[1,10,50,50]
        ids_label = ids[:, self.predict_frame_len:].detach().flatten(0,1) #[9,50,50]

        #forward transformer
        z = rearrange(z, '(b f) d h w -> b f d h w', b=b, f=f) #[1,10,128,50,50]
        rel_poses, pose_label, gt_mode = self.get_pose(data['pose'], data['gt_mode'])
        z_q_predict = self.transformer(z[:, :z.shape[1]-self.predict_frame_len]) #[1,9,512,50,50] [1,9,512]

        #predict pose
        pred_pose = self.forward_plan(rel_poses, z[:, :z.shape[1]-self.predict_frame_len], z_q_predict.clone().detach(), shape, occ_gt.shape)
        pose_decoded = self.pose_decoder(pred_pose) #[1,9,3,2]
        plan_reg_loss = self.losses['plan_reg_loss'](pose_decoded, pose_label, gt_mode)

        #forward vqvae decoder
        z_q_predict = z_q_predict.flatten(0, 1) #[9,512,50,50]
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vqvae.vector_quantizer.get_codebook_entry(z_q_predict)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vqvae.vector_quantizer.post_quant_conv(z_q_predict)

        z_q_predict = self.vqvae.forward_decoder(z_q_predict, shape, occ_gt.shape)
        occ_rec = z_q_predict.argmax(dim=-1).detach()

        occ_m_gt = deepcopy(occ_gt)
        occ_m = deepcopy(occ_rec)
        occ_m_gt[occ_m_gt != 17] = 1
        occ_m_gt[occ_m_gt == 17] = 0
        occ_m[occ_m != 17] = 1
        occ_m[occ_m == 17] = 0
        output.update({
            'occ_rec': occ_rec,
            'occ_gt': occ_gt,
            'occ_m' : occ_m,
            'occ_m_gt' : occ_m_gt,
            'L2': plan_reg_loss
        })

        return output
        

    def forward_test(self, data, input_times, predict_times, device='cuda'):
        occ = data['occs']
        assert input_times + predict_times <= len(occ[0])/2

        occ_input_ = occ[:, :input_times*2].to(device) #[1,6,200,200,16]
        pose_input_ = data['pose'][:, :input_times*2].to(device)
        plan_traj = []
        occ_pred = []
        for i in range(predict_times*2):
            frame_len = input_times*2 + i
            occ_input = torch.cat([occ_input_, occ_pred], dim=1) if len(occ_pred) > 0 else occ_input_
            pose = torch.cat([pose_input_, plan_traj], dim=1) if len(plan_traj) > 0 else pose_input_
            mode = data['gt_mode'][:, :frame_len].to(device)

            b, f, h, w, d = occ_input.shape # [1,10,200,200,16]
        
            #forward vqvae encoder
            z, shape = self.vqvae.forward_encoder(occ_input) 
            z = self.vqvae.vector_quantizer.quant_conv(z)
            z, vq_loss, ids = self.vqvae.vector_quantizer.forward_quantizer(z) #[10,128,50,50] [25000]

            #forward transformer
            z = rearrange(z, '(b f) d h w -> b f d h w', b=b, f=f) #[1,10,128,50,50]
            z_q_predict = self.transformer.forward_test(z)

            #predict pose
            pose_input = torch.cat([pose, mode], dim=-1) # [1,10,5]
            pose_input = self.pose_encoder(pose_input.float())
            pred_pose = self.forward_plan(pose_input, z, z_q_predict.clone().detach(), shape, occ_input.shape)
            pose_decoded = self.pose_decoder(pred_pose) #[1,9,3,2]
            pose_decoded = pose_decoded[mode==1].reshape(1, -1, 2) #[1,9,2]
            if isinstance(plan_traj, list):
                plan_traj.append(pose_decoded[:, -1])
                plan_traj = torch.stack(plan_traj, dim=1)
            else:
                pose_decoded = pose_decoded[:, -1].unsqueeze(1)
                plan_traj = torch.cat([plan_traj, pose_decoded], dim=1)


            #forward vqvae decoder
            z_q_predict = z_q_predict.flatten(0, 1) #[9,512,50,50]
            z_q_predict = z_q_predict.argmax(dim=1)
            z_q_predict = self.vqvae.vector_quantizer.get_codebook_entry(z_q_predict)
            z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
            z_q_predict = self.vqvae.vector_quantizer.post_quant_conv(z_q_predict)

            z_q_predict = self.vqvae.forward_decoder(z_q_predict, shape, occ_input.shape)
            occ_rec = z_q_predict.argmax(dim=-1).detach()
            if isinstance(occ_pred, list):
                occ_pred.append(occ_rec[:, -1])
                occ_pred = torch.stack(occ_pred, dim=1)
            else:
                occ_rec = occ_rec[:, -1].unsqueeze(1)
                occ_pred = torch.cat([occ_pred, occ_rec], dim=1)
        return {
            'plan_traj': plan_traj,
            'occ_pred': occ_pred
        }
