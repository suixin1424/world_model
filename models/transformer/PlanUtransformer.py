import torch
from torch.nn import functional as F
from torch import nn
from .modules import FFN


from mmdet.models import HEADS
from mmcv.runner import BaseModule
from einops import rearrange


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x
class IdentityUnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, 0)
        #self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        return output

@HEADS.register_module()
class PlanUAutoRegTransformer(BaseModule):
    def __init__(
        self,
        num_tokens,
        num_frames,
        predict_frames_len,
        num_layers,
        img_shape,
        tpe_dim=10,
        output_channel=1024,
        channels=[1,2,3],
        temporal_attn_layers=1,
        num_heads=8,
        conditional=True,
        add_aggregate=False,
        learnable_queries=True,
        without_multiscale=False,
        without_spatial_attn=False,
        without_temporal_attn=False) -> None:
        super().__init__()
        if without_multiscale:
            assert len(channels) == 2
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.channels = channels
        self.learnable_queries = learnable_queries
        if self.learnable_queries:
            self.queries = nn.Embedding(img_shape[1]*img_shape[2], img_shape[0])
        self.offset = predict_frames_len
        self.temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        
        self.temporal_attentions_en = nn.ModuleList([])
        self.temporal_attentions_de = nn.ModuleList([])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        up_down_sample_params = dict(kernel_size=2, stride=2, padding=0)
        self.up_down_sample_params = up_down_sample_params
        self.unfold_params = dict(kernel_size=[1, 2], stride=[1, 2], padding=[0, 0])
        
        C, H, W = img_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers-1):
            cH = (cH) // 2
            cW = (cW) // 2
            Hs.append(cH)
            Ws.append(cW)
        pre_c = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.temporal_attentions_en.append(temporal_attn_layer)
            if without_spatial_attn:
                self.encoders.append(nn.Sequential(IdentityUnetBlock((cH, cW), pre_c, channel), 
                                                   IdentityUnetBlock((cH, cW), channel, channel)))
            else:
                self.encoders.append(nn.Sequential(UnetBlock((cH, cW), pre_c, channel, True),
                                               UnetBlock((cH, cW), channel, channel, True)))
            if without_multiscale:
                self.downsamples.append(Identity())
            else:
                self.downsamples.append(nn.Conv2d(channel, channel, **up_down_sample_params))
            pre_c = channel

        channel = channels[-1]
        if without_multiscale:
            if without_spatial_attn:
                self.mid = nn.Sequential(
                    IdentityUnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    IdentityUnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
            else:
                self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                UnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
            )
        else:
            self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[-1], Ws[-1]), pre_c, channel, True),
                UnetBlock((channel, Hs[-1], Ws[-1]), channel, channel, True),
            )



        pre_c = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            channel_agg = channel if add_aggregate else channel * 2
            if without_multiscale:
                self.upsamples.append(Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(pre_c, channel, **up_down_sample_params))
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.temporal_attentions_de.append(temporal_attn_layer)
            if without_spatial_attn:
                self.decoders.append(nn.Sequential(
                    IdentityUnetBlock((channel_agg, cH,cW), channel_agg, channel, True),
                    IdentityUnetBlock((channel, cH,cW), channel, channel, True)
                ))
            else:
                self.decoders.append(
                    nn.Sequential(
                        UnetBlock((channel_agg, cH,cW),
                                channel_agg, channel, True),
                        UnetBlock((channel, cH,cW),
                                channel,
                                channel,
                                True)
                    )
                )
            pre_c = channel
        
        self.conv_out = nn.Conv2d(pre_c, output_channel, 3, 1, 1)
        
        
        attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
        for i_frame in range(num_frames):
            start1 = i_frame * num_tokens
            start2 = start1 + num_tokens if conditional else start1
            attn_mask[start1: (start1 + num_tokens), start2:] = True
        self.register_buffer('attn_mask', attn_mask)
    
    def forward(self, tokens):
        #import pdb;pdb.set_trace()
        # tokens: bs, f, c, h, w
        bs, F, C, H, W = tokens.shape
        assert F == self.num_frames
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[None, self.offset:, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :self.num_frames, None, None, :].expand(
            bs, -1, H, W, -1)
        
        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down in zip(self.temporal_attentions_en, self.encoders, self.downsamples):
            b, f, h, w, c = tokens.shape
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1]):
            queries = up(queries)
            tokens = up(tokens)
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = decoder(queries)
            tokens = decoder(tokens)
        queries = self.conv_out(queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        return queries
    def forward_test(self, tokens):
        #import pdb;pdb.set_trace()
        # tokens: bs, f, c, h, w
        bs, F, C, H, W = tokens.shape
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[None, self.offset:F+self.offset, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :F, None, None, :].expand(
            bs, -1, H, W, -1)
        
        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down in zip(self.temporal_attentions_en, self.encoders, self.downsamples):
            b, f, h, w, c = tokens.shape
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1]):
            queries = up(queries)
            tokens = up(tokens)
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]
                queries = cross_norm(queries)
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = decoder(queries)
            tokens = decoder(tokens)
        queries = self.conv_out(queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        return queries

class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.shortcut = nn.Identity()
            if in_c != out_c:
                self.shortcut = nn.Conv2d(in_c, out_c, 1, 1, 0)
    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        output = self.conv2(output)
        if self.residual:
            output = output + self.shortcut(input)
        output = self.act(output)
        return output
    