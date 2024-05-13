from align_data import depth2dist, get_rays
from align_data import read_json_poses
import torch 
import numpy as np 
import os 
import json 
from read_depth import read_array
from align_data import plot_pcs, plot_pcs_depth
import tqdm
import matplotlib.pyplot as plt 
import re
import math
import h5py
import cv2
import scipy

def compute_normals(depth_map, K, c2w):
    '''Compute the normals from the depth map using camera intrinsics.'''
    n, h, w = depth_map.shape
    # Compute ray origins and directions
    origins, viewdirs = get_rays(K, c2w)
    # Reshape depth_map for broadcasting
    depth_map = torch.from_numpy(depth_map).reshape(n, h, w, 1)

    # Compute 3D points
    points = (depth_map * viewdirs + origins).reshape(n, h, w, 3)
    padded = torch.nn.functional.pad(points, (0, 0, 1, 1, 1, 1), mode='replicate')
    g_x = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2 #each of these terms is of shape (n, 512,512,3)
    g_y = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2
    N = torch.cross(g_x, g_y, dim=-1)
    N_norm = torch.linalg.norm(N, dim=-1, keepdim=True)
    N_normalized = N / torch.where(N_norm == 0, torch.ones_like(N_norm), N_norm)
    
    return N_normalized



def plot_from_depth():
    
    json_file = '/scratch/ondemand28/weihanluo/transientangelo/load/transient_nerf_synthetic/lego/lego_jsons/ten_views/transforms_train.json'
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
    
    # camera id
    colmap_pcs = []
    colmap_normals = []
    for i, frame in enumerate((meta['frames'])):
        if i == 1:
            print("here")
        c2w = torch.from_numpy(np.array(frame['transform_matrix']))
        # c2w = torch.eye(4)
        origins, viewdirs = get_rays(K, c2w)
        # origins = torch.zeros(512,512,3)
        print("loading depths")
        depth = np.load(f"depth_{i}.npy")
        # depth = np.ones((512,512))
        plt.imshow(depth)
        plt.clf()
        plt.close()  
        
        print("calculating normals")
        normal = compute_normals(depth[None,...], K, c2w)
        normal = torch.squeeze(normal)
        plt.imshow(normal)
        plt.savefig(f"normal_{i}.png")
        plt.clf()
        plt.close()        
        
        normal_pc = normal.reshape(-1, 3)
        colmap_pc = (torch.from_numpy(depth[...,None]) * viewdirs + origins).reshape(-1, 3)
        colmap_mask = (depth < 60).flatten()*(depth >0).flatten()
        colmap_pc = colmap_pc[colmap_mask, :]
        normal_pc = normal_pc[colmap_mask, :]
        
        colmap_normals.append(normal_pc)
        colmap_pcs.append(colmap_pc)
        
    colmap_pc = torch.concat(colmap_pcs, dim=0)
    colmap_normal = torch.concat(colmap_normals, dim=0)
    
    plot_pcs([colmap_pc[::90], ], [colmap_normal[::90], ])


if __name__=="__main__":
    plot_from_depth()