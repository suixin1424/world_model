import os
import cv2
from mmdet.datasets import DATASETS
from nuscenes.nuscenes import NuScenes
import torch

@DATASETS.register_module()
class nuscenes:
    def __init__(self, version='v1.0-trainval', data_root='/home/share/datasets/nuscenes/') -> None:
        ''' Initialize a NuScenes dataset.

        Parameters
        ----------
        version : str
            The version of the NuScenes dataset to load.
        data_root : str
            The root directory of the dataset.
        '''
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        self.data_root = data_root
        self.len = len(self.nusc.sample)
    
    def __getitem__(self, index):

        sample = self.nusc.sample[index]
        cam_front_token = sample['data']['CAM_FRONT']
        cam_front_data = self.nusc.get('sample_data', cam_front_token)
        img_path = cam_front_data['filename']
        img = cv2.imread(os.path.join(self.data_root, img_path))
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __len__(self):
        return self.len


