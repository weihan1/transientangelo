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


class CapturedDatasetMovieBase():
    def setup(self, config, split):
        '''
        Use this dataset to watch movies
        '''
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
        


        W, H = self.config.img_wh
        self.w, self.h = W, H
        self.img_wh = (self.w, self.h)
        self.near, self.far = self.config.near_plane, self.config.far_plane
        self.num_views = self.config.num_views
        self.K = LearnRays(self.rays, device=self.rank, img_shape=self.img_wh).to(self.rank)
        
        with open(os.path.join(self.config.root_dir, "final_cams", "movie_jsons", f"transforms_movie_combined.json"), "r") as f:
            meta = json.load(f)
        
        self.all_c2w = np.zeros((len(meta), 4, 4))
        
        for i,key in enumerate(tqdm(meta, desc=f"Processing {self.split} frames")):
            frame = meta[key]["frames"][0]
            number = frame["file_path"].split("/")[-1].split("_")[-1]
            print(f"Processing number {number}", end='\r')
            c2w = torch.from_numpy(np.array(frame['transform_matrix']))
            self.all_c2w[i] = c2w

        self.all_c2w = torch.from_numpy(self.all_c2w)
        print("FINSIHED LOADING THESE CONFIGS")
        
class CapturedMovieDataset(Dataset, CapturedDatasetMovieBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class CapturedMovieIterableDataset(IterableDataset, CapturedDatasetMovieBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('captured-movie')
class CapturedMovieModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    #sets up the train dataset and the val dataset when you call trainer.fit
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = CapturedMovieIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = CapturedMovieDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = CapturedMovieDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = CapturedMovieDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        #Check this to see how the process won't get killed
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

