import torch
import numpy as np
from argparse import ArgumentParser
import os
import h5py
from PIL import Image

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def save_h5(path, data):
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)

def main():
    #1. Load all training transients for one scene
    #Over all 10 training views
    root_dir = f"/scratch/ondemand28/weihanluo/transientangelo/load/transient_nerf_synthetic/{args.scene}/clean_transients"
    
    gt_transient = np.zeros((10,512,512,1200,3))
    for i in range(10):
        print(f"loading clean transients {i} for {args.scene}")
        transient_path = os.path.join(root_dir, f"train_00{i}" + ".h5")
        rgba = read_h5(transient_path) #(h,w,1200,4)
        rgba = np.clip(rgba, 0, None)
        gt_transient[i] = rgba[...,:3]    
    
    # Load the noisy captured transients and then find the optimal scale factor such that the average intensity per pixel matches that of the captured transients
    # Loop over each scene in captured and load scene/low_photon_transients/scene_{low_photon_level}_transient00{i}.mat
    captured_root_dir = "/scratch/ondemand28/weihanluo/transientangelo/load/captured_data"
    captured_scene = ["baskets_raxel", "boots_raxel", "carving_raxel", "chef_raxel", "cinema_raxel", "food_raxel"]
    sum_of_masked_pixels_captured = 0
    total_intensity_captured = 0
    
    for scene in captured_scene:
        gt_captured_transient = np.zeros((20,512,512,1200,3))
        captured_mask = np.zeros(512,512)
        low_photon_dir = os.path.join(captured_root_dir, scene, "low_photon_transients", args.photon_level)
        mask_dir = os.path.join(captured_root_dir, scene)
        for i in range(20):
            print(f"loading captured transients {i} for {scene}")
            transient_path = os.path.join(low_photon_dir, f"transient00{i}.mat")
            mask_path = os.path.join(mask_dir, f"mask{i}.png")
            gt_captured_transient[i] = read_h5(transient_path)
            captured_mask[i] = Image.open(mask_path)
        captured_transient_img = gt_captured_transient.sum(-2) #(20,512,512,3)
        sum_of_masked_pixels_captured += np.sum(captured_mask)
        total_intensity_captured += captured_transient_img[mask].sum() 
        
    average_intensity_per_pixel_captured = total_intensity_captured/sum_of_masked_pixels_captured  
    print(f"The average intensity per pixel for the captured transients for level {args.photon_level} is {average_intensity_per_pixel_captured}")
        

    #TODO: Need to solve for the scale_factor such that the average intensity per pixel matches that of the captured transients
    
        
    #1. Integrate all transients over all views and divide by the sum of masked pixels. Adjust the scale factor such that we get any average intensity per pixel we want.
    scaled_gt_transient = gt_transient*args.scale_factor
    transient_img = scaled_gt_transient.sum(-2) #(10,512,512,3)
    sum_of_masked_pixels = np.sum(transient_img>0)
    average_intensity_per_pixel = transient_img.sum()/sum_of_masked_pixels
    print(f"The average intensity per pixel for the {args.scene} scene is {average_intensity_per_pixel}")
    
    #3. Poisson sample the transients
    average_global_transient = torch.tensor(average_intensity_per_pixel).float()
    background = (average_global_transient/2850)*0.001
    background_img = torch.full_like(torch.tensor(transient_img), background)
    
    #Only add the background to the transients that are black
    torch_transient_img = torch.from_numpy(transient_img)
    background_img[torch_transient_img<=0] = 0
    transient_plus_background = torch.from_numpy(scaled_gt_transient) + background_img[...,None,:]
    noisy_transient = torch.poisson(transient_plus_background)
    print(f"the SBR ratio is {average_intensity_per_pixel/background}")

    #4. Save the noisy transients
    #Save them individually  
    for i in range(10):
        output_dir = os.path.join(root_dir, f"train_00{i}_sampled_{args.scale_factor}.h5")
        save_h5(output_dir, noisy_transient[i].cpu().numpy())
        print(f"Saved {args.scene}{i} to {output_dir}")
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--photon_level", type=int, default=2)
    parser.add_argument("--scene", type=str, default="lego")
    parser.add_argument("--scale_factor", type=float, default= 0.5, help="The scale factor to multiply the transients by before poisson sampling")
    args = parser.parse_args()
    main()