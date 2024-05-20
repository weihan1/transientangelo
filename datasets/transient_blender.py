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

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions, find_mean_focus_point_regnerf
from utils.misc import get_rank
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py



import collections



def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

class TransientDatasetBase():
    def setup(self, config, split):
        #NOTE: DO NOT USE LOGGING!
        self.config = config
        self.split = split
        print(f"Preparing the {self.split} split! 🏃🏃🏃")
        self.rank = get_rank()
        self.rfilter_sigma = config.rfilter_sigma
        self.sample_as_per_distribution = config.sample_as_per_distribution
        self.OPENGL_CAMERA = True
        self.apply_mask = False
        
        
        #Maybe the best way to do testing is to still split it into two 
        if self.split in ["train", "val"]:
            with open(os.path.join(self.config.root_dir, self.config.scene + "_jsons", self.config.num_views + "_views", f"transforms_{self.split}.json"), 'r') as f:
                meta = json.load(f)
        else:
            with open(os.path.join(self.config.root_dir, self.config.scene + "_jsons", self.config.num_views + "_views", f"transforms_{self.split}.json"), 'r') as f:
                meta = json.load(f)

        W, H = self.config.img_wh
        if self.config.downsample: #NOTE: we can only downsample by half the length
            W, H = W//2, H//2
            
        self.w, self.h = W, H
        self.img_wh = (self.w, self.h)
        self.meta = meta
        self.near, self.far = self.config.near_plane, self.config.far_plane
        self.depth_path = os.path.join(self.config.root_dir, self.config.scene + "_jsons", self.config.num_views + "_views", "depth")
        self.focal = 0.5 * self.w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length
        self.num_views = self.config.num_views
        self.scene = self.config.scene
        
        #view scale is dependent on scene and number views
        view_scale_dict = {'lego': {'two': 137568.875, "three": 137568.875, "five": 141280.171875}, 'chair': {"two": 147216.21875, "three": 147216.21875, "five": 147197.421875}, 'ficus': {"two": 145397.578125, "three": 146952.296875, "five": 149278.296875}, 'hotdog': {"two": 138315.34375, "three": 138315.34375, "five": 150297.515625}, 'benches': {"two": 26818.662109375, "three": 26818.662109375, "five": 27648.828125}}
        self.view_scale = view_scale_dict[self.scene][self.num_views]
        
        dataset_scale_dict = {'lego': 1170299.068884274, 'chair': 1553458.125, 'ficus': 1135563.421617064, 'hotdog': 1078811.3736717035, 'benches': 202672.34663868704}
        #dataset scale is only dependent on dataset 
        self.dataset_scale = dataset_scale_dict[self.scene]
        
        self.all_images = np.zeros((len(meta['frames']), 512, 512, 1200, 3))
        self.all_c2w = np.zeros((len(meta['frames']), 4, 4))
        self.all_fg_masks = np.zeros((len(meta['frames']), 512, 512))
        self.scene = self.config.root_dir.split("/")[-1]
        
        if self.scene == "lego":
            emoji = "🚂"
        elif self.scene == "hotdog":
            emoji = "🌭"
        elif self.scene == "chair":
            emoji = "🪑"
        elif self.scene == "benches":
            emoji = "🛋️"
        elif self.scene == "ficus":
            emoji = "🌿"
            
        print(f"Loading the transients for the {self.scene} scene {emoji}!")  
        if self.split == "train" or self.split == "test":
            if self.config.scale_down_photon_factor == 1: #Normal experiments
                for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                    c2w = torch.from_numpy(np.array(frame['transform_matrix']))

                    #Verify if the transients are noisy by checking if the background is all zero
                    self.all_c2w[i] = c2w
                    number = int(frame["file_path"].split("_")[-1])
                    transient_path = os.path.join(self.config.root_dir, "out",f"{split}_{number:03d}" + ".h5")
                    rgba = read_h5(transient_path) #(h,w,1200,4)                    
                    rgba = torch.from_numpy(rgba)
                    rgba = torch.clip(rgba, 0, None)            
                    self.all_images[i] = rgba[...,:3]
                    
            else: #Low photon experiments
                print("Starting the low photon experiments with scale equal to ", self.config.scale_down_photon_factor, "🔦")
                photon_level = self.config.scale_down_photon_factor
                for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                    c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                    self.all_c2w[i] = c2w
                    number = int(frame["file_path"].split("_")[-1])
                    transient_path = os.path.join(self.config.root_dir, "clean_transients", f"{split}_{number:03d}_sampled_{photon_level}.h5")
                    rgba = read_h5(transient_path) #(h,w,1200,4)
                    rgba = torch.from_numpy(rgba)
                    rgba = torch.clip(rgba, 0, None)            
                    self.all_images[i] = rgba[...,:3]
                
                
            self.all_images = torch.from_numpy(self.all_images)  
            self.all_c2w = torch.from_numpy(self.all_c2w)
            self.n_bins = rgba.shape[-2]
            self.max = torch.max(self.all_images)
            
            #Normalizing the images 
            if self.config.gamma:
                print("using gamma")
                rgba_sum = rgba.sum(-2)
                rgba_sum_normalized = rgba_sum/rgba_sum.max()
                rgba_sum_norm_gamma = rgba_sum_normalized**(1/2.2)
                rgba = (rgba*rgba_sum_norm_gamma[..., None, :])/(rgba_sum[..., None, :]+1e-10)
                
            else: #the images are always normalized, whether in training or testing
                self.all_images /= torch.max(self.all_images)
            
            if self.config.downsample:
                print(f"Using downsampled version of shape {self.w, self.h}")
                self.all_images = (self.all_images[:,1::2, ::2] + self.all_images[:,::2, ::2] + self.all_images[:,::2, 1::2] + self.all_images[:,1::2, 1::2])/4
                                                                                                                
            
        else: #For validation
            for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing {self.split} frames")):
                c2w = torch.from_numpy(np.array(frame['transform_matrix']))
                self.all_c2w[i] = c2w
            self.all_c2w = torch.from_numpy(self.all_c2w).float()
            
            
        self.K = torch.tensor(
            [
                [self.focal, 0, self.w / 2.0],
                [0, self.focal, self.h / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        #NOTE: Finding the mean focus point
        print(f"Finding mean focus point for {self.split} 🤔")
        known_rotation_matrices = self.all_c2w[:,:3,:3]
        optical_axes = known_rotation_matrices[...,2]
        known_camera_locations = self.all_c2w[:, :3, -1]
        self.mean_focus_point = find_mean_focus_point_regnerf(known_camera_locations, optical_axes)
        print(f"The mean focus point is {self.mean_focus_point.tolist()} 🤓")
        
class TransientDataset(Dataset, TransientDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class TransientIterableDataset(IterableDataset, TransientDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('transient_blender')
class TransientDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    #sets up the train dataset and the val dataset when you call trainer.fit
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = TransientIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = TransientDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = TransientDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = TransientDataset(self.config, self.config.train_split)

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

