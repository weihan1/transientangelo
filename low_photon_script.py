import torch
import numpy as np
from argparse import ArgumentParser
import os
import h5py

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def save_h5(path, data):
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)

def main():
    if args.version == "simulated":
        #1. Load all training transients for one scene
        #Over all 10 training views
        root_dir = f"/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/{args.scene}/clean_transients"
        
        gt_transient = np.zeros((10,512,512,1200,3))
        for i in range(10):
            print(f"loading clean transients {i} for {args.scene}")
            transient_path = os.path.join(root_dir, f"train_00{i}" + ".h5")
            rgba = read_h5(transient_path) #(h,w,1200,4)
            rgba = np.clip(rgba, 0, None)
            gt_transient[i] = rgba[...,:3]    
        
        
        #2. Mask and calculate transient average for each scene
        #Average across all visible entries in the gt_transient
        print(f"performing transient average calculation with scale factor {args.scale_factor}")
        gt_transient = gt_transient*args.scale_factor
        transient_img = gt_transient.sum(-2)
        visible_transient_mask = transient_img>0 #Mask out the black pixels
        average_global_transient = np.mean(transient_img[visible_transient_mask])
        print(f"The average transient flux for the {args.scene} scene is {average_global_transient}")
        
        #3. Poisson sample the transients
        # flux_2m = mean over i {img_i[img_i>0].mean()} 
        # flux_2 = flux_2 + flux_2m/2850*0.001
        # constant = constant for each scene
        average_global_transient = torch.tensor(average_global_transient).float()
        background = (average_global_transient/2850)*0.001
        
        transient_plus_background = torch.from_numpy(gt_transient)+background
        noisy_transient = torch.poisson(transient_plus_background)

        #4. Save the noisy transients
        #Save them individually  
        for i in range(10):
            output_dir = os.path.join(root_dir, f"train_00{i}_sampled_{args.scale_factor}.h5")
            save_h5(output_dir, noisy_transient[i].cpu().numpy())
            print(f"Saved {args.scene}{i} to {output_dir}")
        
    # if args.version =="captured":
    #     intrinsics = f"./load/captured_data/{args.scene}/david_calibration_1005/intrinsics_david.npy" #NOTE: The intrinsics are the same for each scene
    #     params = np.load(intrinsics, allow_pickle=True)[()]
    #     shift = params['shift'].numpy()
    #     rays = params['rays']
    #     W,H = 512,512
    #     n_bins = 1500
    #     exposure_time = 299792458*4e-12
    #     x = (torch.arange(W, device="cpu")-W//2+0.5)/(W//2-0.5)
    #     y = (torch.arange(W, device="cpu")-W//2+0.5)/(W//2-0.5)
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
        
    #     #NOTE: Captured only use these scenes for training, they are:  r_1, r_3, r_5, r_9, r_13, r_17, r_19
    #     gt_transient = np.zeros(7,512,512,1500,3)
        
    #     for i in [1,3,5,9,13,17,19]:
    #         number = i
    #         transient_path = os.path.join(args.root_dir,f"transient{number:03d}" + ".pt")
    #         rgba = torch.load(transient_path).to_dense() #r_i Loads the corresponding transient00i
    #         rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
    #         rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
    #         rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2 #(512,512, 1500)
    #         rgba = np.clip(rgba, 0, None)
    #         rgba = rgba[..., None].repeat(1, 1, 1, 3) #(512,512, 1500, 3)
    #         gt_transient[i] = rgba[...,:3] 
        
    #     visible_transient_mask = gt_transient>0
    #     average_global_transient = np.mean(gt_transient[visible_transient_mask])
    #     print(f"The average transient flux for the {args.scene} scene is {average_global_transient}")
        
    #     gt_transient = torch.from_numpy(gt_transient)
    #     gt_transient *= args.scale_factor 
    #     # constant = constant for each scene
    #     background = (average_global_transient/2850)*0.001
    #     noisy_transient = torch.poisson(gt_transient+background)

    #     #4. Save the noisy transients
    #     for i in [1,3,5,9,13,17,19]:
    #         output_dir = os.path.join(args.output_dir, f"train_00{i}_sampled_{args.scale_factor}.h5")
    #         save_h5(output_dir, noisy_transient.cpu().numpy())
        
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, default="lego")
    parser.add_argument("--scale_factor", type=float, default= 0.5, help="The scale factor to multiply the transients by before poisson sampling")
    parser.add_argument("--version", type=str, default="simulated", choices=["captured", "simulated"], help="Dataset being trained, captured or simulated.")
    args = parser.parse_args()
    main()