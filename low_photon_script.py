import torch
import numpy as np
from argparse import ArgumentParser
import os
import h5py
from PIL import Image
import mat73
import matplotlib.pyplot as plt
def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def save_h5(path, data):
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)

def main():
    #1. Load all simulated transients for one scene
    root_dir = f"/scratch/ondemand28/weihanluo/transientangelo/load/transient_nerf_synthetic/{args.scene}/clean_transients"
    gt_transient = np.zeros((10,512,512,1200,3))
    print("Generating noisy transients at photon_level", args.photon_level)
    for i in range(10):
        print(f"loading clean transients {i} for {args.scene}")
        transient_path = os.path.join(root_dir, f"train_00{i}" + ".h5")
        rgba = read_h5(transient_path) #(h,w,1200,4)
        rgba = np.clip(rgba, 0, None)
        gt_transient[i] = rgba[...,:3]    
        image = gt_transient[i].sum(-2)
        plt.imshow(image/image.max())
        plt.savefig(os.path.join(root_dir, f"train_00{i}_clean_image.png"))
    
    # # Load the noisy captured transients and then find the optimal scale factor such that the average intensity per pixel matches that of the captured transients
    # # Loop over each scene in captured and load scene/low_photon_transients/scene_{low_photon_level}_transient00{i}.mat
    # captured_root_dir = "/scratch/ondemand28/weihanluo/transientangelo/load/captured_data"
    # captured_scene = ["baskets_raxel", "boots_raxel", "carving_raxel", "chef_raxel", "cinema_raxel", "food_raxel"]
    # sum_of_masked_pixels_captured = 0
    # total_intensity_captured = 0
    
    # for scene in captured_scene:
    #     w,h = 512,512
    #     n_bins=1500
    #     intrinsics = f"./load/captured_data/{scene}/david_calibration_1005/intrinsics_david.npy"
    #     params = np.load(intrinsics, allow_pickle=True)[()]
    #     shift = params['shift'].numpy()
    #     exposure_time = 299792458*4e-12
    #     x = (torch.arange(w, device="cpu")-w//2+0.5)/(w//2-0.5)
    #     y = (torch.arange(w, device="cpu")-w//2+0.5)/(w//2-0.5)
    #     z = torch.arange(n_bins*2, device="cpu").float()
    #     X, Y, Z = torch.meshgrid(x, y, z, indexing="xy")
    #     Z = Z*exposure_time/2
    #     Z = Z - shift[0]
    #     Z = Z*2/exposure_time
    #     Z = (Z-n_bins*2//2+0.5)/(n_bins*2//2-0.5)
    #     grid = torch.stack((Z, X, Y), dim=-1)[None, ...]
    #     del X
    #     del Y
    #     del Z
    #     gt_captured_transient = np.zeros((20,512,512,1500))
    #     captured_mask = np.zeros((20,512,512))
    #     low_photon_dir = os.path.join(captured_root_dir, scene, "low_photon_transients", f"{scene}_{args.photon_level}")
    #     mask_dir = os.path.join(captured_root_dir, scene)
    #     for i in range(20):
    #         print(f"loading captured transients {i} for {scene}")
    #         number = f"00{i+1}"
    #         if i > 8:
    #             number = f"0{i+1}"
    #         transient_path = os.path.join(low_photon_dir, f"transient{number}.mat")
    #         mask_path = os.path.join(mask_dir, f"train_{number}_mask.png")
    #         matlab_transient = mat73.loadmat(transient_path)["transients"].reshape(h,w,-1)
    #         matlab_transient = torch.from_numpy(matlab_transient[..., :3000]).float()
    #         matlab_transient = torch.nn.functional.grid_sample(matlab_transient[None, None, ...], grid, align_corners=True).squeeze()
    #         matlab_transient = (matlab_transient[..., 1::2]+matlab_transient[..., ::2])/2
    #         gt_captured_transient[i] = matlab_transient.numpy()
    #         mask = Image.open(mask_path)
    #         mask = np.array(mask)
    #         captured_mask = mask/255
    #     captured_transient_img = gt_captured_transient.sum(-1) #(20,512,512)
    #     sum_of_masked_pixels_captured += np.sum(captured_mask)
    #     total_intensity_captured += (captured_transient_img *captured_mask).sum() 
    #     np.save(f"sum_of_masked_pixels_captured_after_{scene}.npy", np.array(sum_of_masked_pixels_captured))
    #     np.save(f"total_intensity_captured_after_{scene}.npy", np.array(total_intensity_captured))
        
        
        
    # average_intensity_per_pixel_captured = total_intensity_captured/sum_of_masked_pixels_captured  
    # print(f"The average intensity per pixel for the captured transients for level {args.photon_level} is {average_intensity_per_pixel_captured}")
    # np.save(f"Average_intensity_captured_{args.photon_level}.npy", np.array(average_intensity_per_pixel_captured))
    
    #Solve for scale factor
    decrease_factor = args.photon_level/2.0
    average_intensity_per_pixel_captured = np.load(f"Average_intensity_captured_2.npy")/ decrease_factor  #exposure time 
    transient_img_unscaled = gt_transient.sum(-2) #(10,512,512,3)
    sum_of_masked_pixels = np.sum(transient_img_unscaled>0)
    scale_factor = average_intensity_per_pixel_captured*sum_of_masked_pixels/(transient_img_unscaled.sum())
    
    
    
    scaled_gt_transient = gt_transient*scale_factor
    scaled_transient_img = scaled_gt_transient.sum(-2) #(10,512,512,3)
    scaled_sum_of_masked_pixels = np.sum(scaled_transient_img>0)
    average_intensity_per_pixel = scaled_transient_img.sum()/scaled_sum_of_masked_pixels
    assert (average_intensity_per_pixel - average_intensity_per_pixel_captured <  1e-1)
    print(f"The average intensity per pixel for the {args.scene} scene is {average_intensity_per_pixel}")
    
    #3. Poisson sample the transients
    average_global_transient = torch.tensor(average_intensity_per_pixel).float()
    background = (average_global_transient/2850)*0.001
    background_img = torch.full_like(torch.tensor(scaled_transient_img), background) #repeat background to match the shape of the transient image
    
    #Only add the background to the transients that are not black
    torch_transient_img = torch.from_numpy(scaled_transient_img)
    background_img[torch_transient_img<=0] = 0
    transient_plus_background = torch.from_numpy(scaled_gt_transient) + background_img[...,None,:]
    noisy_transient = torch.poisson(transient_plus_background)
    print(f"the SBR ratio is {average_intensity_per_pixel/background}")

    #4. Save the noisy transients
    #Save them individually  
    for i in range(10):
        output_dir = os.path.join(root_dir, f"train_00{i}_sampled_{args.photon_level}.h5")
        noisy_image = noisy_transient[i].cpu().numpy().sum(-2)
        plt.imshow(noisy_image/noisy_image.max())
        plt.savefig(os.path.join(root_dir, f"train_00{i}_sampled_{args.photon_level}_image.png"))
        plt.close()
        save_h5(output_dir, noisy_transient[i].cpu().numpy())
        print(f"Saved {args.scene}{i} to {output_dir}")
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--photon_level", type=int, default=2)
    parser.add_argument("--scene", type=str, default="lego")
    args = parser.parse_args()
    main()