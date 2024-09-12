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
        self.l2 = [0]*self.times
        self.count = 0
    def reset(self):
        self.l2 = [0]*self.times
        self.count = 0
    def compute_L2(self, trajs, gt_trajs):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        '''
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        # import pdb; pdb.set_trace()
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )
        
        return ade
    def _after_step(self, traj_pred, traj_label):
        #traj_pred: [b,9,2]
        #traj_label: [b,9,2]
        self.count += 1
        traj_pred = torch.cumsum(traj_pred, dim=1)
        traj_label = torch.cumsum(traj_label, dim=1)
        for i in range(self.times):
            frame = (i+1)*2
            traj = traj_pred[:,:frame,:]
            gt_traj = traj_label[:,:frame,:]
            self.l2[i] += sum(
                [self.compute_L2(traj[j], gt_traj[j]) for j in range(traj.shape[0])]
            )/traj.shape[0]
        
    def _after_epoch(self):
        l2 = [l/self.count for l in self.l2]
        return l2



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