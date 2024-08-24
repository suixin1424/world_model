import argparse
import os
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from utils import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector, build_head
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

from models import *
from dataloader import *
from loss import *

def parse_args():
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--samples_per_gpu', type=int, default=1)
    parser.add_argument('--workers_per_gpu', type=int, default=1)
    parser.add_argument('--start_times', type=int, default=3)
    parser.add_argument('--pred_times', type=int, default=3)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.device = get_device()

    # init distributed
    if args.launcher == 'none':
        distributed = False
        cfg.gpu_ids = range(1)
    else:
        distributed = True
        init_dist(args.launcher)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # build dataset
    test_dataset = build_dataset(cfg.data_test)
    test_dataloader = build_dataloader(test_dataset, args.samples_per_gpu, args.workers_per_gpu, dist=distributed, shuffle=False)
    # build model
    model = build_head(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, test_dataloader, cfg, args.start_times, args.pred_times)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, test_dataloader, cfg, args.start_times, args.pred_times)

    if not distributed or rank == 0:
        l2 = outputs['L2']
        miou = outputs['miou']
        for i in range(args.pred_times):
            print(f'{i+1} s: {l2[i]:.4f}, mIoU: {miou[i]:.4f}')

if __name__ == '__main__':
    main()