from align_data import depth2dist, get_rays
from align_data import read_json_poses
import torch 
import numpy as np 
import os 
import json 
from read_depth import read_array
from align_data import plot_pcs
import tqdm
import matplotlib.pyplot as plt 
import re
import math
import h5py
import cv2
import scipy

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def get_depth_from_transient(transient, exposure_time, tfilter_sigma, mask=None):
    '''
    Get ground truth depth from transient by computing the correlation between the filter and the first channel of the transient  
    '''
    transient = transient[..., 0] #(n_rays,n_bins)
    f = lambda x: np.exp(-x**2/(2*tfilter_sigma**2)) - np.exp(-8)
    filter = np.arange(-4*tfilter_sigma+1, 4*tfilter_sigma, 1)
    filter = f(filter) #(8*tfilter_sigma,)
    bin_numbers, vals = log_matched_filter(transient, filter)
    depths = bins_to_depth_mitsuba(bin_numbers=bin_numbers, exposure_time=exposure_time)
    if mask is not None:
        depths[mask==0] = 0
    return depths



def bins_to_depth_mitsuba(bin_numbers, exposure_time):
    times = bin_numbers*exposure_time
    times= times/2
    return times 


def log_matched_filter(transient, filter):
    transient_shape = transient.shape
    transient = transient.reshape(-1, transient_shape[-1])
    corr = scipy.signal.correlate(transient, filter[:, None], mode="same")
    corr = torch.from_numpy(corr) 
    vals, argmax_bin = torch.max(corr, -1, keepdim=True)
    # argmax_bin = np.argmax(corr, axis=-1, keepdims=True)
    # values = corr[:, argmax_bin.squeeze()]
    if transient_shape[0] == transient_shape[1]:
        argmax_bin = argmax_bin.reshape(transient_shape[0], transient_shape[1])
        vals = vals.reshape(transient_shape[0], transient_shape[1])
    return argmax_bin.numpy(), vals.numpy()



def plot_from_depth():
    
    json_file = '/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/lego/lego_jsons/ten_views/transforms_train.json'
    with open(json_file) as f:
        meta = json.load(f)
        
    w,h = 512,512
    focal =  0.5 * w / math.tan(0.5 * meta['camera_angle_x'])
    
    K = torch.tensor(
            [
                [focal, 0, w / 2.0],
                [0, focal, h / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
    )
    
    mask_dir = "/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/lego/masks"
    transient_dir = "/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/lego/out"
    # camera id
    colmap_pcs = []
    exposure_time = 0.01
    tfilter_sigma = 3
    for i, frame in enumerate((meta['frames'])):
        c2w = torch.from_numpy(np.array(frame['transform_matrix']))
        origins, viewdirs = get_rays(K, c2w)
        
        number = int(frame["file_path"].split("_")[-1])
        mask_path = os.path.join(mask_dir, f"train_{number:03d}_mask" + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # plt.imshow(mask)
        # plt.savefig(f"mask_{i}.png")
        # plt.close()        
        try:
            print("loading depths")
            depth = np.load(f"depth_{i}.npy")
            plt.imshow(depth)
            # plt.savefig(f"depth_{i}.png")
            plt.clf()
            plt.close()  
            
        except:
            print("depth not found, loading transietns")
            transient_path = os.path.join(transient_dir, f"train_{number:03d}.h5")
            rgba = read_h5(transient_path) #(h,w,1200,4)                    
            rgba = torch.from_numpy(rgba)
            rgba = torch.clip(rgba, 0, None)
            transient_image = rgba.sum(-2)**(1/2.2)
            transient_image_normalized = transient_image / transient_image.max()
            plt.imshow(transient_image_normalized)
            # plt.savefig(f"transient_{i}.png")
            plt.clf()
            plt.close()
            depth = get_depth_from_transient(rgba, exposure_time, tfilter_sigma, mask).reshape(h, w)
            # np.save(f"depth_{i}.npy", depth)
            
        
        colmap_pc = (torch.from_numpy(depth[...,None]) * viewdirs + origins).reshape(-1, 3)
        colmap_mask = (depth < 60).flatten()*(depth >0).flatten()
        colmap_pc = colmap_pc[colmap_mask, :]
        colmap_pcs.append(colmap_pc)
        
    colmap_pc = torch.concat(colmap_pcs, dim=0)
    plot_pcs([colmap_pc[::90], ])


if __name__=="__main__":
    plot_from_depth()