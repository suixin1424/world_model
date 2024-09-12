base_channel = 64
_dim_ = 16
expansion = 8
n_e_ = 512
model = dict(
    type = 'e2e_wm',
    predict_frame_len=1,
    h_occ = 16,
    h_occ_encodered = 128,
    vqvae = dict(
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
        num_classes=18,
        expansion=expansion, 
        vector_quantizer=dict(
            type='VectorQuantizer',
            n_e = n_e_, 
            e_dim = base_channel * 2, 
            beta = 1., 
            z_channels = base_channel * 2, ),
        rec_loss = dict(
            type = 'rec_loss',
        )
    ),
    
    transformer=dict(
        type = 'OccTransformer',
        num_tokens=1,
        num_frames=9,
        predict_frames_len=1,
        num_layers=2,
        img_shape=(base_channel*2,50,50),
        tpe_dim=base_channel*2,
        channels=(base_channel*2, base_channel*4, base_channel*8),
        temporal_attn_layers=6,
        output_channel=n_e_,
        learnable_queries=False
    ),
    pose_encoder=dict(
        type = 'PoseEncoder',
        in_channels=5,
        out_channels=base_channel*8,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    pose_decoder=dict(
        type = 'PoseDecoder',
        in_channels=base_channel*8,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
    ),
    plan_head=dict(
        type = 'PlanHead',
        idx_shape=(50,50),
        in_channels=base_channel*2,
        num_embeddings=n_e_,
        embedding_dim=base_channel*8,
    ),
    loss=dict(
        type = 'multi_loss',
        ce_loss = dict(
            type = 'ce_loss',
            weight = 1.0,
        ),
        plan_reg_loss = dict(
            type = 'plan_reg_loss',
            weight = 0.1,
            num_modes = 3,
            loss_type = 'l2',
        )
    ),
)