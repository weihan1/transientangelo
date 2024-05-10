import torch
import tqdm
import matplotlib.pyplot as plt 
import re
import math
import h5py
import cv2
import scipy
import os
import numpy as np
from argparse import ArgumentParser



def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def main():
    '''
    Simple script to read a poisson sampled transient image and visualize it
    '''
    for number in range(10):
        print(f"Visualizing transient {number} for {args.scene} at scale {args.scale}...")
        transient_dir = f"/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/{args.scene}/clean_transients"
        transient_path = os.path.join(transient_dir, f"train_{number:03d}_sampled_{args.scale}.h5")
        if args.scale == 1.0:
            transient_path = os.path.join(transient_dir, f"train_{number:03d}.h5")
        rgba = read_h5(transient_path) #(h,w,1200,4)                    
        rgba = torch.from_numpy(rgba)
        rgba = torch.clip(rgba, 0, None)
        rgba = rgba[...,:3]
        transient_image = rgba.sum(-2)**(1/2.2)
        transient_image_normalized = transient_image / transient_image.max()
        plt.imshow(transient_image_normalized)
        plt.savefig(os.path.join(transient_dir, f"train_{number:03d}_sampled_transient_{args.scale}.png"))
        plt.clf()
        plt.close()
        
    print("Finished visualizing all transients!")
    
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, default="lego")
    parser.add_argument("--scale", type=float, default=0.1, choices=[0.5,0.1, 0.25, 1.0])
    args = parser.parse_args()
    main()