from copy import deepcopy
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from .metric_util import multi_step_MeanIou, get_nuScenes_label_name, multi_step_L2
import numpy as np

def single_gpu_test(model, dataloader, cfg, start_times, pred_times):
    model.eval()
    start_frames = start_times * 2
    pred_frames = pred_times * 2

    #miou
    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', cfg.device, pred_frames)
    CalMeanIou_sem.reset()
    #iou
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', cfg.device, pred_frames)
    CalMeanIou_vox.reset()
    #l2
    plan_l2 = multi_step_L2(pred_times)
    #fps
    cost_time = 0

    prog_bar = mmcv.ProgressBar(len(dataloader))
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            result = model.module.forward_test(data, input_times=3, predict_times=3, device=cfg.device)
        
        #calculate miou
        occ_label = data['occs'][:, start_frames:pred_frames+start_frames].to(cfg.device)
        CalMeanIou_sem._after_step(result['occ_pred'], occ_label)

        #calculate iou
        occ = deepcopy(result['occ_pred'])
        label = deepcopy(occ_label)
        occ[occ != 17] = 1
        occ[occ == 17] = 0
        label[label != 17] = 1
        label[label == 17] = 0
        CalMeanIou_vox._after_step(occ, label)

        #calculate l2
        traj_label = data['pose'][:, start_frames:pred_frames+start_frames].to(cfg.device) #[1,6,2]
        traj_pred = result['plan_traj'] #[1,6,2]
        plan_l2._after_step(traj_pred, traj_label)

        #calculate fps
        cost_time += result['cost_time']
        prog_bar.update()
    print('\n')
    miou, _ = CalMeanIou_sem._after_epoch()
    iou, _ = CalMeanIou_vox._after_epoch()
    l2 = plan_l2._after_epoch()

    miou_times = [0]*pred_times
    iou_times = [0]*pred_times
    for i in range(pred_times):
        miou_times[i] = (miou[2*i]+miou[2*i+1])/2
        iou_times[i] = (iou[2*i]+iou[2*i+1])/2

    cost_time /= len(dataloader)
    result = dict(L2=l2, miou=miou_times, iou=iou_times, fps=1/cost_time)

    return result

def multi_gpu_test(model, dataloader, cfg, start_times, pred_times):
    model.eval()
    start_frames = start_times * 2
    pred_frames = pred_times * 2

    rank, world_size = get_dist_info()

    #miou
    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', cfg.device, pred_frames)
    CalMeanIou_sem.reset()
    #iou
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', cfg.device, pred_frames)
    CalMeanIou_vox.reset()
    #l2
    plan_l2 = multi_step_L2(pred_times)
    #fps
    cost_time = 0

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataloader))
    for i, data in enumerate(dataloader):
        print(data.keys())
        exit(0)
        with torch.no_grad():
            result = model.module.forward_test(data, input_times=3, predict_times=3, device=cfg.device)

        #calculate miou
        occ_label = data['occs'][:, start_frames:pred_frames+start_frames].to(cfg.device)
        CalMeanIou_sem._after_step(result['occ_pred'], occ_label)

        #calculate iou
        occ = deepcopy(result['occ_pred'])
        label = deepcopy(occ_label)
        occ[occ != 17] = 1
        occ[occ == 17] = 0
        label[label != 17] = 1
        label[label == 17] = 0
        CalMeanIou_vox._after_step(occ, label)

        #calculate l2
        traj_label = data['pose'][:, start_frames:pred_frames+start_frames].to(cfg.device) #[1,6,2]
        traj_pred = result['plan_traj'] #[1,6,2]
        plan_l2._after_step(traj_pred, traj_label)

        #calculate fps
        cost_time += result['cost_time']
        if rank == 0:
            prog_bar.update()
    if rank == 0:
        print('\n')
    miou, _ = CalMeanIou_sem._after_epoch()
    iou, _ = CalMeanIou_vox._after_epoch()
    l2 = plan_l2._after_epoch()


    miou_times = [0]*pred_times
    iou_times = [0]*pred_times
    for i in range(pred_times):
        miou_times[i] = (miou[2*i]+miou[2*i+1])/2
        iou_times[i] = (iou[2*i]+iou[2*i+1])/2

    cost_time /= len(dataloader)

    #collect
    miou_times = torch.tensor(miou_times).to(cfg.device) #[3]
    dist.all_reduce(miou_times, op=dist.ReduceOp.SUM)
    miou_times /= world_size
    iou_times = torch.tensor(iou_times).to(cfg.device) #[3]
    dist.all_reduce(iou_times, op=dist.ReduceOp.SUM)
    iou_times /= world_size
    l2 = torch.tensor(l2).to(cfg.device) #[3]
    dist.all_reduce(l2, op=dist.ReduceOp.SUM)
    l2 /= world_size
    cost_time = torch.tensor(cost_time).to(cfg.device)
    dist.all_reduce(cost_time, op=dist.ReduceOp.SUM)
    cost_time /= world_size

    result = dict(L2=l2, miou=miou_times, iou=iou_times, fps=1/cost_time)



    return result

def visualization(ego_traj):
    pass