import numpy as np
import os
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import torch


def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def main():
    root_dir = f"./load/transient_nerf_synthetic/{args.scene}"
    with open(os.path.join(root_dir, args.scene + "_jsons", args.num_views + "_views", f"transforms_train.json"), 'r') as f:
        meta = json.load(f)
    print("Config loaded")
    all_images = torch.zeros((len(meta['frames']), 512, 512, 1200, 3))
    photon_level = args.photon_level
    photon_dir = "/scratch/ondemand28/weihanluo/transientangelo/clean_transients/simulated"
    for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing train frames")):
        number = int(frame["file_path"].split("_")[-1])
        transient_path = os.path.join(photon_dir, args.scene, "clean_transients", f"train_{number:03d}_sampled_{photon_level}.h5")
        rgba = read_h5(transient_path) #(h,w,1200,4)                    
        rgba = torch.from_numpy(rgba)
        rgba = torch.clip(rgba, 0, None)            
        all_images[i] = rgba[...,:3]
        
    max_val_train = torch.max(all_images)
    print(f"max_val_train: {max_val_train}")
    outpath = os.path.join(photon_dir, args.scene, "clean_transients")
    np.savetxt(f"{outpath}/max_{args.num_views}_{args.scene}_{args.photon_level}.txt", np.array([max_val_train.item()]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--photon_level", type=int, choices=[600, 300, 150, 50, 10], default=10)
    parser.add_argument("--num_views", type=str, default="two")
    parser.add_argument("--scene", type=str, default="lego")
    args = parser.parse_args()
    main()