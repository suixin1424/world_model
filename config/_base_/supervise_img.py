model = dict(
    img_backbone = dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck = dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),   
    voxel_height=16
)