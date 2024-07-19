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
    '''Sample noisy transients at one photon level args.photon_level for args.scene aross all views
    args.photon_level corresponds to the index into [600, 300, 150, 50, 10], i.e. args.photon_level=0 corresponds to 600 photons per pixel
    1. Start by loading all clean transients for a specific scene args.scene.
    2. Then, solve for the scale factor that will match each of the average photon intensity per pixel
    3. Create a background and add it to the noisy transient and poisson sample it. 
    '''
    
    root_dir = f"/scratch/ondemand28/weihanluo/transientangelo/clean_transients/simulated/{args.scene}/clean_transients"
    #1. Open the clean transient for one scene across all views
    gt_transient = np.zeros((10,512,512,1200,3))
    noisy_transient = np.zeros((10,512,512,1200,3))
    lst_of_avg_photon_intensity_per_pixel = [600, 300, 150, 50, 10]
    
    for i in range(10):
        print(f"loading clean transients {i} for {args.scene}")
        transient_path = os.path.join(root_dir, f"train_00{i}" + ".h5")
        rgba = read_h5(transient_path) #(h,w,1200,4)
        rgba = np.clip(rgba, 0, None)
        gt_transient[i] = rgba[...,:3]    
        image = gt_transient[i].sum(-2)
        plt.imshow(image/image.max()) #saving the images just for cross-referencing
        plt.savefig(os.path.join(root_dir, f"train_00{i}_clean_image.png"))
        
    #2. Average photon intensity per pixel
    average_intensity_per_pixel = lst_of_avg_photon_intensity_per_pixel[args.photon_level]
    print(f"calculating the scale factor with the average photon intensity per pixel {average_intensity_per_pixel}")
    transient_img_unscaled = gt_transient.sum(-2) #(10,512,512,3)
    sum_of_masked_pixels = np.sum(transient_img_unscaled>0)
    scale_factor = average_intensity_per_pixel*sum_of_masked_pixels/(transient_img_unscaled.sum()) #Solve for the scale factor
    photon_dir = "/scratch/ondemand28/weihanluo/transientangelo/clean_transients/simulated"
    outpath = os.path.join(photon_dir, args.scene, "clean_transients")
    np.savetxt(f"{outpath}/scale_factor_{args.scene}_{average_intensity_per_pixel}.txt", np.array([scale_factor]))
    
    scaled_gt_transient = gt_transient*scale_factor
    scaled_transient_img = scaled_gt_transient.sum(-2) #(10,512,512,3)
    scaled_sum_of_masked_pixels = np.sum(scaled_transient_img>0)
    average_intensity_per_pixel_prime = scaled_transient_img.sum()/scaled_sum_of_masked_pixels 
    assert average_intensity_per_pixel_prime - average_intensity_per_pixel < 1e-1, f"the average intensity per pixel is {average_intensity_per_pixel_prime} and should be {average_intensity_per_pixel}"
    
    
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
        output_dir = os.path.join(root_dir, f"train_00{i}_sampled_{average_intensity_per_pixel}.h5")
        noisy_image = noisy_transient[i].cpu().numpy().sum(-2)
        plt.imshow(noisy_image/noisy_image.max())
        plt.savefig(os.path.join(root_dir, f"train_00{i}_sampled_{average_intensity_per_pixel}_image.png"))
        plt.close()
        save_h5(output_dir, noisy_transient[i].cpu().numpy())
        print(f"Saved {args.scene}{i} to {output_dir}")
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--photon_level", type=int, default=0, choices=[0,1,2,3,4])
    parser.add_argument("--scene", type=str, default="lego")
    args = parser.parse_args()
    main()