_base_ = [
    './_base_/occ_mamba.py',
]

runner = dict(type='EpochBasedRunner', max_epochs=200)
workflow = [('train', 1), ('val', 1)]

vqvae_path = '/home/zhuyiming/data2/world_model/workdir/20240708_215804/best_miou.pth'
workdir = '/home/zhuyiming/data2/world_model/workdir/occmamba/'
predict_frames_len = 1
seed = 3407
log_interval = 10
save_interval = 10
unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"

freeze_dict = dict(
    vqvae = True,
)


frames = 8
data_train = dict(
    type = 'Nuscenes',
    data_root = 'dataset/nuscenes/',
    frames = frames,
    test_mode = False,
    pkl_path = 'dataset/nuscenes_infos_train_temporal_v3_scene.pkl',
)
data_test = dict(
    type = 'Nuscenes',
    data_root = 'dataset/nuscenes/',
    frames = frames,
    test_mode = True,
    pkl_path = 'dataset/nuscenes_infos_val_temporal_v3_scene.pkl',
)


model = dict(
    predict_frame_len = predict_frames_len
)

optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
)

lr_config = dict(
    min_lr = 1e-5,
    warmup = 'linear',
    warmup_iters = 100,
    warmup_ratio = 0.01,
)