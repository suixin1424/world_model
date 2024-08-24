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
from models.mambapy import Mamba, MambaConfig

@HEADS.register_module()
class occmamba(BaseModule):
    def __init__(self, predict_frame_len, n_layers, vq_out_dim, d_model, vqvae, loss):
        super(occmamba, self).__init__()

        self.predict_frame_len = predict_frame_len
        self.vqvae = HEADS.build(vqvae)
        self.n_layers = n_layers
        self.d_model = d_model
        self.mamba = Mamba(MambaConfig(n_layers=self.n_layers, d_model=self.d_model))
        self.mamba_in_proj = nn.Embedding(d_model, d_model)
        self.mamba_out_proj = nn.Linear(d_model, d_model)
        self.mamba_out_proj.weight = self.mamba_in_proj.weight
        if loss.type == 'multi_loss':
            self.losses = {
                key: LOSSES.build(value) for key, value in loss.items() if key != 'type'
            }
        else:
            self.loss = LOSSES.build(loss)
    

    def train_step(self, data, optimizer, **kwargs):
        occ = data['occs']
        b, f, h, w, d = occ.shape
        
        output = {}
        #forward vqvae encoder
        z, shape = self.vqvae.forward_encoder(occ) 
        z = self.vqvae.vector_quantizer.quant_conv(z)
        z, vq_loss, ids = self.vqvae.vector_quantizer.forward_quantizer(z) #[10,128,50,50] [25000]
        ids = rearrange(ids, '(b f h w) -> b f h w', b=b, f=f, h=z.shape[-2], w=z.shape[-1]) #[1,10,50,50]
        ids_label = ids[:, self.predict_frame_len:].detach().flatten(0,1) #[9,50,50]

        #forward mamba
        mamba_input = ids[:, :ids.shape[1]-self.predict_frame_len] #[1,9,50,50]
        mamba_input = rearrange(mamba_input, 'b f h w -> (b f) (h w)') #[9,2500]
        mamba_input = self.mamba_in_proj(mamba_input) #[9,2500,512]
        mamba_out = self.mamba(mamba_input) #[9,2500,512]
        mamba_out = self.mamba_out_proj(mamba_out) #[9,2500,512]
        mamba_out = rearrange(mamba_out, '(b f) (h w) d -> (b f) d h w', b=b, h=z.shape[-2], w=z.shape[-1]) #[9,512,50,50]

        #forward loss
        loss = self.loss(mamba_out, ids_label)

        output['loss'] = loss

        return output
    
    def val_step(self, data, optimizer, **kwargs):
        occ = data['occs']
        b, f, h, w, d = occ.shape
        occ_gt = occ[:, self.predict_frame_len:]
        
        output = {}
        #forward vqvae encoder
        z, shape = self.vqvae.forward_encoder(occ) 
        z = self.vqvae.vector_quantizer.quant_conv(z)
        z, vq_loss, ids = self.vqvae.vector_quantizer.forward_quantizer(z) #[10,128,50,50] [25000]
        ids = rearrange(ids, '(b f h w) -> b f h w', b=b, f=f, h=z.shape[-2], w=z.shape[-1]) #[1,10,50,50]
        ids_label = ids[:, self.predict_frame_len:].detach().flatten(0,1) #[9,50,50]

        #forward mamba
        mamba_input = ids[:, :ids.shape[1]-self.predict_frame_len] #[1,9,50,50]
        mamba_input = rearrange(mamba_input, 'b f h w -> (b f) (h w)') #[9,2500]
        mamba_input = self.mamba_in_proj(mamba_input) #[9,2500,512]
        mamba_out = self.mamba(mamba_input) #[9,2500,512]
        mamba_out = self.mamba_out_proj(mamba_out) #[9,2500,512]
        mamba_out = rearrange(mamba_out, 'bf hw d -> hw bf d') #[2500,9,512]

        # forward vqvae decoder
        mamba_out = mamba_out.argmax(dim=-1) #[2500,9]
        mamba_out = rearrange(mamba_out, '(b h w) f -> (b f) h w', b=b, h=z.shape[-2], w=z.shape[-1]) #[9,50,50]
        z_q_predict = self.vqvae.vector_quantizer.get_codebook_entry(mamba_out)
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
            'occ_m_gt' : occ_m_gt
        })

        return output


        
        

