import time
import os.path as osp

from mmcv.runner import Hook, save_checkpoint
import mmcv
from mmdet.utils import get_root_logger, build_ddp
import numpy as np
import torch
from mmcv import ProgressBar

from utils import freeze_model, multi_step_MeanIou, get_nuScenes_label_name
from tensorboardX import SummaryWriter
import subprocess

def find_unused_parameters(model):
    unused_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.grad_fn is None:
            unused_parameters.append(name)
    return unused_parameters

class train_hook(Hook):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        
    def before_run(self, runner):
        if not self.cfg.dist or self.cfg.rank == 0:
            #tensorboard
            self.tensorboard = SummaryWriter(log_dir=self.cfg.workdir)
            #subprocess.Popen(["tensorboard", "--logdir", self.cfg.workdir])

        #print logs
        seed = self.cfg.seed
        torch.manual_seed(seed)
        if not self.cfg.dist or self.cfg.rank == 0:
            runner.logger.info(f'Set random seed to {seed}')
            runner.logger.info(f'Tensorboard log dir is {self.tensorboard.logdir}')

        
        if self.cfg.get('resume_from', None) is not None:
            runner.resume(self.cfg.resume_from)
        runner.model.device = self.cfg.device

        #best val miou
        runner.best_val_miou = [0]*(self.cfg.get('frames', 10) - self.cfg.get('predict_frames_len', 1))


    def before_train_epoch(self, runner):
        if not self.cfg.dist or self.cfg.rank == 0:
            runner.logger.info(f'Epoch {runner.epoch}/{runner.max_epochs}')
        
    def before_train_iter(self, runner):
        pass
    
    def after_train_iter(self, runner):
        if runner.iter % self.cfg.log_interval == 0:
            if not self.cfg.dist or self.cfg.rank == 0:
                #print logs
                runner.logger.info(f'Epoch {runner.epoch}, Iter {runner._inner_iter}/{runner.max_iters//runner.max_epochs}, loss: {runner.outputs["loss"]}, lr: {runner.optimizer.param_groups[0]["lr"]}')
                #tensorboard
                for key in runner.outputs.keys():
                    if 'loss' in key:
                        self.tensorboard.add_scalar(f'train/{key}', runner.outputs[key], runner.iter)
    
    def after_train_epoch(self, runner):
        pass

    def before_val_epoch(self, runner):
        if not self.cfg.dist or self.cfg.rank == 0:
            #init miou
            label_name = get_nuScenes_label_name(self.cfg.label_mapping)
            unique_label = np.asarray(self.cfg.unique_label)
            unique_label_str = [label_name[l] for l in unique_label]
            runner.CalMeanIou_sem = multi_step_MeanIou(unique_label, self.cfg.get('ignore_label', -100), unique_label_str, 'sem', self.cfg.device, self.cfg.frames-self.cfg.predict_frames_len)
            runner.CalMeanIou_sem.reset()

            runner.CalMeanIou_vox = multi_step_MeanIou([1], self.cfg.get('ignore_label', -100), ['occupied'], 'vox', self.cfg.device, self.cfg.frames-self.cfg.predict_frames_len)
            runner.CalMeanIou_vox.reset()
            runner.logger.info('start val')
            self.progress_bar = ProgressBar(len(runner.data_loader))

    def before_val_iter(self, runner):
        pass

    def after_val_iter(self, runner):
        if not self.cfg.dist or self.cfg.rank == 0:
            #update miou
            runner.CalMeanIou_sem._after_step(runner.outputs['occ_rec'], runner.outputs['occ_gt'])
            runner.CalMeanIou_vox._after_step(runner.outputs['occ_m'], runner.outputs['occ_m_gt'])
            self.progress_bar.update()

    
    def after_val_epoch(self, runner):
        if not self.cfg.dist or self.cfg.rank == 0:
            val_miou, _ = runner.CalMeanIou_sem._after_epoch()
            val_iou, _ = runner.CalMeanIou_vox._after_epoch()
            time.sleep(2)
            #print logs
            runner.logger.info(f'Epoch {runner.epoch}/{runner.max_epochs}, val miou: {val_miou}')
            runner.logger.info(f'Epoch {runner.epoch}/{runner.max_epochs}, val iou: {val_iou}')
            #tensorboard
            for i in range(len(val_miou)):
                self.tensorboard.add_scalar(f'val/frame_{i}', val_miou[i], runner.epoch)
            #save best
            if max(val_miou) > max(runner.best_val_miou):
                runner.save_checkpoint(self.cfg.workdir, f'best_miou.pth')
            #best miou
            runner.best_val_miou = [max(runner.best_val_miou[i], val_miou[i]) for i in range(len(runner.best_val_miou))]
        

            
        
    

