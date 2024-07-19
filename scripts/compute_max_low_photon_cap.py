import numpy as np
import os
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import h5py
import torch
import sio

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames





def main():
    exposure_time = 299792458*4e-12
    w,h = 512,512
    n_bins = 1500
    intrinsics = f"./load/captured_data/{args.scene}/david_calibration_1005/intrinsics_david.npy"
    params = np.load(intrinsics, allow_pickle=True)[()]
    shift = params['shift'].numpy()
    rays = params['rays']
    laser_pulse_dic = sio.loadmat('./load/captured_data/pulse_low_flux.mat')['out'].squeeze()
    laser_pulse = laser_pulse_dic
    laser_pulse = (laser_pulse[::2] + laser_pulse[1::2])/2
    lidx = np.argmax(laser_pulse)
    loffset = 50
    laser = laser_pulse[lidx-loffset:lidx+loffset+1]
    laser = laser / laser.sum()
    laser = laser[::-1]
    laser = torch.tensor(laser.copy()).float().to("cpu")
    laser_kernel = torch_laser_kernel(laser, device="cpu")
                

    x = (torch.arange(w, device="cpu")-w//2+0.5)/(w//2-0.5)
    y = (torch.arange(w, device="cpu")-w//2+0.5)/(w//2-0.5)
    z = torch.arange(n_bins*2, device="cpu").float()
    X, Y, Z = torch.meshgrid(x, y, z, indexing="xy")
    Z = Z*exposure_time/2
    Z = Z - shift[0]
    Z = Z*2/exposure_time
    Z = (Z-n_bins*2//2+0.5)/(n_bins*2//2-0.5)
    grid = torch.stack((Z, X, Y), dim=-1)[None, ...]
    del X
    del Y
    del Z
    
    root_dir = f"./load/captured_data/{args.scene}"
    with open(os.path.join(root_dir, "final_cams", args.num_views + "_views", f"transforms_train.json"), 'r') as f: 
        meta = json.load(f)
    print("Config loaded")
    all_images = torch.zeros((len(meta['frames']), 512, 512, 1500, 3))
    photon_level = args.photon_level
    photon_dir = "/scratch/ondemand28/weihanluo/transientangelo/clean_transients/captured"
    for i, frame in enumerate(tqdm(meta['frames'], desc=f"Processing train frames")):
        number = int(frame["file_path"].split("_")[-1])
        actual_scene = args.scene.split("_")[0]
        transient_path = os.path.join(photon_dir, actual_scene, f"{photon_level}", f"transient_{number:02d}" + ".pt")
        rgba = torch.load(transient_path).to_dense() #r_i Loads the corresponding transient00i
        rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
        rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
        rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2 #(512,512, 1500)
        rgba = torch.clip(rgba, 0, None)
        rgba = rgba[..., None].repeat(1, 1, 1, 3) #(512,512, 1500, 3)
        all_images[i] = rgba #(h,w,1500,3)
        
    max_val_train = torch.max(all_images)
    print(f"max_val_train: {max_val_train}")
    outpath = os.path.join(photon_dir, args.scene, "clean_transients")
    np.savetxt(f"{outpath}/max_{args.num_views}_{args.scene}_{args.photon_level}.txt", np.array([max_val_train.item()]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--photon_level", type=int, choices=[600, 300, 150, 50, 10], default=10)
    parser.add_argument("--num_views", type=str, default="two")
    parser.add_argument("--scene", type=str, default="cinema_raxel")
    args = parser.parse_args()
    main()