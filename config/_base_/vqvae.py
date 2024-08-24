expansion = 8
base_channel = 64
n_e_ = 512
_dim_ = 16
model = dict(
    type = 'vqvae',
    encoder=dict(
        type='Encoder2D',
        ch = base_channel, 
        out_ch = base_channel, 
        ch_mult = (1,2,4), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel * 2, 
        double_z = False,
    ), 
    decoder=dict(
        type='Decoder2D',
        ch = base_channel, 
        out_ch = _dim_ * expansion, 
        ch_mult = (1,2,4), 
        num_res_blocks = 2,
        attn_resolutions = (50,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        z_channels = base_channel * 2, 
        give_pre_end = False
    ),
    vector_quantizer=dict(
        type='VectorQuantizer',
        n_e = n_e_, 
        e_dim = base_channel * 2, 
        beta = 1., 
        z_channels = base_channel * 2),
    rec_loss = dict(
        type = 'rec_loss',
    ),
    lovasz_loss = dict(
        type = 'LovaszLoss'
    ),
    num_classes=18,
    expansion=expansion, 
)