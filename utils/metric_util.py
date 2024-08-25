import numpy as np
import torch
import yaml
def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]
    return nuScenes_label_name

class multi_step_L2:
    def __init__(self, times=3):
        self.times = times
        self.frames = times*2
        self.l2_frame = [0]*self.frames
        self.l2 = [0]*self.times
        self.count = 0
    def reset(self):
        self.l2_frame = [0]*self.frames
        self.l2 = [0]*self.times
        self.count = 0
    def _after_step(self, traj_pred, traj_label):
        #traj_pred: [b,9,2]
        #traj_label: [b,9,2]
        self.count += 1
        for i in range(self.frames):
            current_traj_pred = traj_pred[:,:i+1,:] # [b,i+1,2]
            current_traj_label = traj_label[:,:i+1,:] # [b,i+1,2]
            self.l2_frame[i] += torch.sqrt(((current_traj_pred - current_traj_label) ** 2).sum(-1)).sum().mean().item()
    def _after_epoch(self):
        for i in range(self.frames):
            self.l2_frame[i] /= self.count
        for i in range(self.times):
            self.l2[i] = (self.l2_frame[i*2] + self.l2_frame[i*2+1])/2
        return self.l2



class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

    def _after_epoch(self):

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        
        return miou * 100


class multi_step_MeanIou:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 device,
                 times=1):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        self.device = device
        
    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).to(self.device)
        self.total_correct = torch.zeros(self.times, self.num_classes).to(self.device)
        self.total_positive = torch.zeros(self.times, self.num_classes).to(self.device)
    
    def _after_step(self, outputses, targetses):
        
        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:,t, ...][targetses[:,t, ...] != self.ignore_label].to(self.device)
            targets = targetses[:,t, ...][targetses[:,t, ...] != self.ignore_label].to(self.device)
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c)
                                                      & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()
    
    def _after_epoch(self):

        mious = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            miou = np.mean(ious)
            mious.append(miou * 100)
        return mious, np.mean(mious)