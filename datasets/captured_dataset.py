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
import scipy.io as sio
import logging
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import datasets
from models.ray_utils import get_ray_directions, LearnRays, find_mean_focus_point_regnerf
from utils.misc import get_rank
from models.utils import torch_laser_kernel


import h5py



import collections



def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

class CapturedDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.rfilter_sigma = config.rfilter_sigma #0.1
        self.sample_as_per_distribution = config.sample_as_per_distribution
        self.OPENGL_CAMERA = True
        self.apply_mask = False
        self.has_mask = True
        self.rep = 1
        
        self.exposure_time = 299792458*4e-12*2
        self.n_bins = 1500
        self.scene = config.scene
        self.n_views = config.num_views
        self.params = np.load(self.config.intrinsics, allow_pickle=True)[()]
        self.shift = self.params['shift'].numpy()
        self.rays = self.params['rays']
        transient_scale_dict = {'cinema': {"two": 483.755615234375, "three": 483.755615234375, "five": 491.021728515625}, 'carving': {"two": 299.24365234375, "three": 299.24365234375, "five": 323.334228515625}, 'boots': {"two": 273.02276611328125, "three": 273.02276611328125, "five": 277.6478271484375}, 'food': {"two": 553.1838989257812, "three": 553.1838989257812, "five": 561.6094970703125}, 'chef': {"two": 493.50701904296875, "three": 493.50701904296875, "five": 548.0447998046875}, 'baskets': {"two": 308.7045593261719, "three": 319.92572021484375, "five": 326.42620849609375}}
        self.transient_scale = transient_scale_dict[self.scene][self.n_views]
        self.rep_number = 30
        self.div_vals = {'boots': 4470.0,
                        'baskets': 6624.0,
                        'carving': 4975.0,
                        'chef': 9993.0,
                        'cinema': 8478.0,
                        'food': 16857.0}
        
        var_increase = self.config.var_increase
        
        
        laser_pulse_dic = sio.loadmat('./load/captured/pulse_low_flux.mat')['out'].squeeze()
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
            #Rendering the training poses during validation as well
            with open(os.path.join(self.config.root_dir, "final_cams", self.config.num_views + "_views", f"transforms_train.json"), 'r') as f: 
                meta = json.load(f)
                
        else: #NOTE: the captured dataset test poses are in a separate folder 
            with open(os.path.join(self.config.root_dir, "final_cams", "test_jsons", f"transforms_{self.split}.json"), "r") as f:
                meta = json.load(f)


        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 512, 512

        #Used for downscaling
        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)
        self.meta = meta
        self.near, self.far = self.config.near_plane, self.config.far_plane
        self.depth_path = os.path.join(self.config.root_dir, self.config.scene + "_jsons", self.config.num_views + "_views", "depth")
        

        # self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.all_images = np.zeros((len(meta['frames']), h, w, self.n_bins, 3))
        self.all_c2w = np.zeros((len(meta['frames']), 4, 4))
        self.all_fg_masks = np.zeros((len(meta['frames']), h, w))
        
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
        
        if self.split == "train" or self.split == "test":
            if self.split == "train":
                if self.config.photon_level == 0:
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
                        if self.config.use_bkgd_masking:
                            mask_path = os.path.join(self.config.root_dir, f"train_{number:03d}_mask.png")  
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            assert mask.max() == 255
                            rgba *= (mask[...,None,None]/255)
                            print("using masked transients")
                        self.all_images[i] = rgba #(h,w,1500,3)
                else: #Low photon experiments
                    print("Starting the low photon experiments with scale equal to ", self.config.photon_level, "🔦")
                    photon_level = self.config.photon_level
                    photon_dir = "./low_photon_transients/captured"
                    for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                        number = int(frame["file_path"].split("_")[-1])
                        actual_scene = self.scene.split("_")[0]
                        transient_path = os.path.join(photon_dir, actual_scene, f"{photon_level}", f"transient_{number:02d}" + ".pt")
                        rgba = torch.load(transient_path).to_dense() #r_i Loads the corresponding transient00i
                        rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
                        rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
                        rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2 #(512,512, 1500)
                        c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                        self.all_c2w[i] = c2w
                        rgba = torch.clip(rgba, 0, None)
                        rgba = rgba[..., None].repeat(1, 1, 1, 3) #(512,512, 1500, 3)
                        if self.config.use_bkgd_masking:
                            mask_path = os.path.join(self.config.root_dir, f"train_{number:03d}_mask.png")  
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            assert mask.max() == 255
                            rgba *= (mask[...,None,None]/255)
                            print("using masked transients")
                        self.all_images[i] = rgba #(h,w,1500,3)
                        
            elif self.split == "test":       
                for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                    number = int(frame["file_path"].split("_")[-1])
                    print(f"loading number {number}")
                    transient_path = os.path.join(self.config.root_dir,f"transient{number:03d}" + ".pt")
                    rgba = torch.load(transient_path).to_dense() #r_i Loads the corresponding transient00i
                    rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
                    rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
                    rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2 #(512,512, 1500)
                    c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                    self.all_c2w[i] = c2w
                    rgba = torch.clip(rgba, 0, None)
                    rgba = rgba[..., None].repeat(1, 1, 1, 3) #(512,512, 1500, 3)
                    self.all_images[i] = rgba #(h,w,1500,3)
                    
                    
            self.all_images = torch.from_numpy(self.all_images)
            self.all_c2w = torch.from_numpy(self.all_c2w)
            np.save(os.path.join(self.config.root_dir, "max.npy"), torch.max(self.all_images).cpu().numpy())
            if self.split == "train" and self.config.photon_level!=0:
                preloaded_max_dir = os.path.join(photon_dir, actual_scene, f"{photon_level}", f"{self.n_views}_views_max.npy")
                preloaded_max = np.load(preloaded_max_dir)
                assert (torch.max(self.all_images)).numpy() - preloaded_max < 1e-3
            #Normalizing the images
            max = torch.max(self.all_images)
            self.all_images /= max
            self.focal = max
        

        else:
            for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                self.all_c2w[i] = c2w
            self.all_c2w = torch.from_numpy(self.all_c2w)
            camera_angle_x = float(meta["camera_angle_x"])
            self.focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
            
        #rays have some rays that are empty
        self.K = LearnRays(self.rays, device=self.rank, img_shape=self.img_wh).to(self.rank)
        known_rotation_matrices = self.all_c2w[:,:3,:3]
        optical_axes = known_rotation_matrices[...,2]
        known_camera_locations = self.all_c2w[:, :3, -1]
        self.mean_focus_point = find_mean_focus_point_regnerf(known_camera_locations, optical_axes)
        
class CapturedDataset(Dataset, CapturedDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class CapturedIterableDataset(IterableDataset, CapturedDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('captured_dataset')
class CapturedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    #sets up the train dataset and the val dataset when you call trainer.fit
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = CapturedIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = CapturedDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = CapturedDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = CapturedDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
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

