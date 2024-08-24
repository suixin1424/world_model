import warnings
import argparse
from mmcv import Config, DictAction
import os
import os.path as osp
import mmcv
import time
from mmdet.utils import get_root_logger, get_device, build_ddp
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_head, build_loss
from mmdet.core import build_optimizer
from mmcv.runner import build_runner, CheckpointHook, get_dist_info, init_dist, EpochBasedRunner, DistSamplerSeedHook, OptimizerHook, CosineAnnealingLrUpdaterHook

from models import *
from dataloader import *
from loss import *
from hooks import *
from utils import *

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume_from', help='resume from checkpoint', default=None, type=str)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--samples_per_gpu', type=int, default=1)
    parser.add_argument('--workers_per_gpu', type=int, default=1)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.device = get_device()
    cfg['resume_from'] = args.resume_from

    #init distributed
    if args.launcher is not None:
        init_dist(args.launcher)
        _, world_size = get_dist_info()
        cfg.dist = True
        cfg.rank = int(os.environ['LOCAL_RANK'])
        cfg.gpu_ids = range(world_size)
    else:
        cfg.dist = False
        cfg.gpu_ids = range(1)

    #build model
    model = build_head(cfg.model)
    if 'vqvae_path' in cfg.keys():
        model.vqvae.load_state_dict(torch.load(cfg.vqvae_path, map_location='cpu')['state_dict'], strict=True)
        freeze_model(model, cfg.freeze_dict)
    if cfg.dist:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=True)

    #build dataset
    train_dataset = build_dataset(cfg.data_train)
    val_dataset = build_dataset(cfg.data_test)
    train_dataloader = build_dataloader(train_dataset, args.samples_per_gpu, args.workers_per_gpu, num_gpus=len(cfg.gpu_ids), dist=cfg.dist, shuffle=False)
    val_dataloader = build_dataloader(val_dataset, args.samples_per_gpu, args.workers_per_gpu, num_gpus=len(cfg.gpu_ids), dist=cfg.dist, shuffle=False)


    #build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)


    #build logger
    logger = None
    if cfg.rank == 0:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        cfg.workdir = osp.join(cfg.workdir, timestamp)
        mmcv.mkdir_or_exist(osp.abspath(cfg.workdir))
        cfg.dump(osp.join(cfg.workdir, "config.py"))
    log_file = osp.join(cfg.workdir, 'output.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    #build runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.workdir,
            logger=logger))
    
    #register hooks
    runner.register_hook(OptimizerHook())
    if cfg.lr_config is not None:
        runner.register_hook(CosineAnnealingLrUpdaterHook(**cfg.lr_config))
    runner.register_hook(train_hook(cfg))
    runner.register_hook(CheckpointHook(cfg.save_interval, by_epoch=True, save_optimizer=True, out_dir=cfg.workdir))
    if cfg.dist:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
    runner.run(
        [train_dataloader, val_dataloader],
        cfg.workflow,
        cfg=cfg
    )


    exit(0)





if __name__ == '__main__':
    main()