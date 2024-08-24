import os
import pickle
from mmdet.datasets import DATASETS
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@DATASETS.register_module()
class Nuscenes(Dataset):
    def __init__(
            self,
            data_root: str = '/home/share/datasets/nuscenes/',
            frames: int = 10,
            test_mode: bool = False,
            pkl_path: str = 'data/nuscenes_infos_train_temporal_v3_scene.pkl',
            ) -> None:
        ''' Initialize a NuScenes dataset.'''
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.scenes = data['infos']
        self.scene_names = list(self.scenes.keys())
        self.data_root = data_root
        self.frames = frames
        self.test_mode = test_mode
     
    def __len__(self):
        ''' Return the number of scenes.'''
        return len(self.scenes)
    
    def get_lidar2img(self,info):
        lidar2img = []
        for cam_type, cam_info in info['cams'].items():
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T) # 4x4
            lidar2img.append(lidar2img_rt)
        lidar2img = np.stack(lidar2img)
        return lidar2img
    
    def __getitem__(self, idx):
        ''' Return the idx-th scene.'''
        scene_name = self.scene_names[idx]
        scene = self.scenes[scene_name]
        scene_len = len(scene)
        idx = np.random.randint(0, scene_len - self.frames + 1)
        occs = []
        pose = []
        gt_mode = []
        cams = []
        lidar2img = []
        for i in range(self.frames):
            token = scene[idx + i]['token']
            lidar2img.append(self.get_lidar2img(scene[idx + i]))
            cam = scene[idx + i]['cams']
            imgs = []
            for c in cam.values():
                img_path = c['data_path'].split('nuscenes/')[-1]
                img_path = os.path.join(self.data_root, img_path)
                img = Image.open(img_path)
                img = np.array(img)
                imgs.append(img)
            imgs = np.stack(imgs)
            cams.append(imgs)
            pose.append(scene[idx + i]['gt_ego_fut_trajs'][0])
            gt_mode.append(scene[idx + i]['pose_mode'])
            label_file = os.path.join(self.data_root, f'gts/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        occs = np.stack(occs, dtype=np.int64)
        pose = np.asarray(pose)  # [frame,2]
        gt_mode = np.asarray(gt_mode) #[frame,3]
        cams = np.stack(cams)
        lidar2img = np.stack(lidar2img)
        occs = torch.from_numpy(occs)
        pose = torch.from_numpy(pose)
        gt_mode = torch.from_numpy(gt_mode)
        cams = torch.from_numpy(cams)
        lidar2img = torch.from_numpy(lidar2img) # [frame, view, 4, 4]
        return dict(
            occs=occs,
            pose=pose,
            gt_mode=gt_mode,
            cams=cams,
            lidar2img=lidar2img,
        )

