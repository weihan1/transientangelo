import math
import collections
import json
import os
from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import logging
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions, find_mean_focus_point_regnerf, LearnRays, compute_normals_from_transient_captured
from utils.misc import get_rank
from systems.criterions import get_depth_from_transient, get_depth_from_transient_captured
import h5py
import collections
import scipy.io as sio
from models.utils import torch_laser_kernel




def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

class BaselineDatasetCapturedBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        print(f"Preparing the {self.split} split! 🏃🏃🏃")
        self.rank = get_rank()
        self.rfilter_sigma = config.rfilter_sigma
        self.sample_as_per_distribution = config.sample_as_per_distribution
        self.OPENGL_CAMERA = True
        self.apply_mask = False
        self.has_mask = False
        self.exposure_time = 299792458*4e-12*2
        self.n_bins = 1500
        self.n_views = config.num_views
        self.params = np.load(self.config.intrinsics, allow_pickle=True)[()]
        self.shift = self.params['shift'].numpy()
        self.rays = self.params['rays']
        self.scene = self.config.scene
        transient_scale_dict = {'cinema_raxel': {"two": 483.755615234375, "three": 483.755615234375, "five": 491.021728515625}, 'carving_raxel': {"two": 299.24365234375, "three": 299.24365234375, "five": 323.334228515625}, 'boots_raxel': {"two": 273.02276611328125, "three": 273.02276611328125, "five": 277.6478271484375}, 'food_raxel': {"two": 553.1838989257812, "three": 553.1838989257812, "five": 561.6094970703125}, 'chef_raxel': {"two": 493.50701904296875, "three": 493.50701904296875, "five": 548.0447998046875}, 'baskets_raxel': {"two": 308.7045593261719, "three": 319.92572021484375, "five": 326.42620849609375}}
        self.view_scale = transient_scale_dict[self.scene][self.n_views]
        self.div_vals = {'boots_raxel': 4470.0,
                        'baskets_raxel': 6624.0,
                        'carving_raxel': 4975.0,
                        'chef_raxel': 9993.0,
                        'cinema_raxel': 8478.0,
                        'food_raxel': 16857.0}
        self.dataset_scale = self.div_vals[self.scene]
        laser_pulse_dic = sio.loadmat('./load/captured_data/pulse_low_flux.mat')['out'].squeeze()
        laser_pulse = laser_pulse_dic
        laser_pulse = (laser_pulse[::2] + laser_pulse[1::2])/2
        lidx = np.argmax(laser_pulse)
        loffset = 50
        laser = laser_pulse[lidx-loffset:lidx+loffset+1]
        laser = laser / laser.sum()
        laser = laser[::-1]
        self.laser = torch.tensor(laser.copy()).float().to(self.rank)
        self.laser_kernel = torch_laser_kernel(self.laser, device=self.rank)
        
        if self.split in ["train", "val"]:
            with open(os.path.join(self.config.root_dir, "final_cams", self.config.num_views + "_views", f"transforms_{self.split}.json"), 'r') as f:
                meta = json.load(f)
                
        else: #NOTE: the captured dataset test poses are in a separate folder 
            with open(os.path.join(self.config.root_dir, "final_cams", "test_jsons", f"transforms_{self.split}.json"), "r") as f:
                meta = json.load(f)

        W, H = self.config.img_wh
        self.w, self.h = W, H
        self.img_wh = (self.w, self.h)
        self.meta = meta
        self.near, self.far = self.config.near_plane, self.config.far_plane
        self.num_views = self.config.num_views
        self.all_images = np.zeros((len(meta['frames']), 512, 512, 1500, 3))
        self.all_c2w = np.zeros((len(meta['frames']), 4, 4))
        self.all_fg_masks = np.zeros((len(meta['frames']), 512, 512))
        
        
        exposure_time = 299792458*4e-12
        x = (torch.arange(self.w, device="cpu")-self.w//2+0.5)/(self.w//2-0.5)
        y = (torch.arange(self.w, device="cpu")-self.w//2+0.5)/(self.w//2-0.5)
        z = torch.arange(self.n_bins*2, device="cpu").float()
        X, Y, Z = torch.meshgrid(x, y, z, indexing="xy")
        Z = Z*exposure_time/2
        Z = Z - self.shift[0]
        Z = Z*2/exposure_time
        Z = (Z-self.n_bins*2//2+0.5)/(self.n_bins*2//2-0.5)
        grid = torch.stack((Z, X, Y), dim=-1)[None, ...]
        del X
        del Y
        del Z
        
        self.K = LearnRays(self.rays, device=self.rank, img_shape=self.img_wh).to(self.rank)
        if self.split == "train" or self.split == "test":
            for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                number = int(frame["file_path"].split("_")[-1])
                transient_path = os.path.join(self.config.root_dir,f"transient{number:03d}" + ".pt")
                rgba = torch.load(transient_path).to_dense() #r_i Loads the corresponding transient00i
                rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
                rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
                rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2 #(512,512, 1500)
                c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                self.all_c2w[i] = c2w
                rgba = torch.clip(rgba, 0, None)      
                rgba = rgba[..., None].repeat(1, 1, 1, 3) #(512,512, 1500, 3)      
                self.all_images[i] = rgba[...,:3]
                if self.config.use_mask:
                    mask_dir = os.path.join(self.config.root_dir, f"train_{number:03d}_mask.png")
                    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
                    self.all_fg_masks[i] = mask
                
            
            self.all_images = torch.from_numpy(self.all_images)
            self.all_c2w = torch.from_numpy(self.all_c2w)
            
            
            if self.config.scale_down_photon_factor < 1:
                self.all_images *= self.config.scale_down_photon_factor
                self.all_images = torch.poisson(self.all_images)
                
            
            self.n_bins = rgba.shape[-2]
            self.max = torch.max(self.all_images)
            
            
        else: #For validation
            for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                self.all_c2w[i] = c2w
            self.all_c2w = torch.from_numpy(self.all_c2w).float()
            camera_angle_x = float(meta["camera_angle_x"])
            self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)
            
        if self.config.use_gt_depth_normal and self.split == 'train':
            try:
                print("Loading saved depth")
                self.all_depths = np.load(f"{self.scene}-{self.num_views}-depths.npy")
            except:
                print("No depth found...Calculating the ground truth depth and normals...🤖")
                self.all_depths = get_depth_from_transient_captured(self.all_images, self.laser.cpu().numpy(), self.all_fg_masks if self.config.use_mask else None)
                np.save(f"{self.scene}-{self.num_views}-depths.npy", self.all_depths)
            self.all_normals = compute_normals_from_transient_captured(self.all_depths.reshape(-1, self.h, self.w), self.K, self.all_c2w)
            
        
        #NOTE: Finding the mean focus point
        print(f"Finding mean focus point for {self.split} 🤔")
        known_rotation_matrices = self.all_c2w[:,:3,:3]
        optical_axes = known_rotation_matrices[...,2]
        known_camera_locations = self.all_c2w[:, :3, -1]
        self.mean_focus_point = find_mean_focus_point_regnerf(known_camera_locations, optical_axes)
        print(f"The mean focus point is {self.mean_focus_point.tolist()} 🤓")
        print(f"its average distance to existing camera locations is {np.mean(np.linalg.norm(known_camera_locations - self.mean_focus_point, axis=-1))} 📏")
        
        
class BaselineDataset(Dataset, BaselineDatasetCapturedBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BaselineIterableDataset(IterableDataset, BaselineDatasetCapturedBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('baseline-captured')
class BaselineDataCapturedModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    #sets up the train dataset and the val dataset when you call trainer.fit
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BaselineIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BaselineDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BaselineDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BaselineDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        #Check this to see how the process won't get killed
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=1, 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       

