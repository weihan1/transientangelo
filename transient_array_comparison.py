import numpy as np
import matplotlib.pyplot as plt
import os


# base_directory = "/Users/weihanluo/transientneuralangel Dropbox/Wei Han Luo/multiview_transient_surface/figures/simulation"

# pred_directories_ours = ["two_views/ours/lego", "three_views/ours/lego", "five_views/ours/lego",
#                          "two_views/ours/chair", "three_views/ours/chair", "five_views/ours/chair",
#                          "two_views/ours/ficus", "three_views/ours/ficus", "five_views/ours/ficus",
#                          "two_views/ours/hotdog", "three_views/ours/hotdog", "five_views/ours/hotdog","two_views/ours/benches", "three_views/ours/benches", "five_views/ours/benches"]




# pred_directories_transient_nerf = ["two_views/transientnerf/lego", "three_views/transientnerf/lego", "five_views/transientnerf/lego",
#                          "two_views/transientnerf/chair", "three_views/transientnerf/chair", "five_views/transientnerf/chair",
#                          "two_views/transientnerf/ficus", "three_views/transientnerf/ficus", "five_views/transientnerf/ficus",
#                          "two_views/transientnerf/hotdog", "three_views/transientnerf/hotdog", "five_views/transientnerf/hotdog","two_views/transientnerf/benches", "three_views/transientnerf/benches", "five_views/transientnerf/benches"]



# gt_directories = ["two_views/gt/lego", "three_views/gt/lego", "five_views/gt/lego",
#                          "two_views/gt/chair", "three_views/gt/chair", "five_views/gt/chair",
#                          "two_views/gt/ficus", "three_views/gt/ficus", "five_views/gt/ficus",
#                          "two_views/gt/hotdog", "three_views/gt/hotdog", "five_views/gt/hotdog","two_views/gt/benches", "three_views/gt/benches", "five_views/gt/benches"]




normal_gt_lego_transient_path = "/scratch/ondemand28/weihanluo/TransientNeRF/gt_array/simulated/lego_two_views/gt_array/0_transient_array.npy"
normal_pred_lego_transient_path ="/scratch/ondemand28/weihanluo/transientangelo/exp/transient-neuralangelo-blender-lego/lego-three-views-1@20240614-185629/save/it0-test/0_transient_array.npy"
low_photon_50_pred_lego_transient_path = "/scratch/ondemand28/weihanluo/transientangelo/exp/transient-neuralangelo-blender-lego/lego-two-views-50@baseline/save/it0-test/0_transient_array.npy"
low_photon_50_gt_lego_transient_path = "/scratch/ondemand28/weihanluo/transientangelo/exp/transient-neuralangelo-blender-lego/lego-two-views-50@baseline/save/it0-test/0_transient_gt_array.npy"




def main():
    print("loading transient arrays")
    normal_gt_transient = np.load(normal_gt_lego_transient_path)
    normal_pred_lego_transient = np.load(normal_pred_lego_transient_path)
    low_photon_50_pred_lego_transient = np.load(low_photon_50_pred_lego_transient_path)
    low_photon_50_gt_lego_transient = np.load(low_photon_50_gt_lego_transient_path)
    
    normal_gt_transient_mid = normal_gt_transient[256,256,:,0] #(1200,1)
    normal_pred_transient_mid = normal_pred_lego_transient[256,256,:,0] #(1200,1)
    low_photon_50_pred_transient_mid = low_photon_50_pred_lego_transient[256,256,:,0]#(1200,1)
    low_photon_50_gt_transient_mid = low_photon_50_gt_lego_transient[256,256,:,0]#(1200,1)
    
    plt.figure(figsize=(12, 8))
    print("starting to print some cool transients")
    plt.plot(normal_gt_transient_mid, label='Ground Truth (Full Photon)', color='b', linestyle='-')
    plt.plot(normal_pred_transient_mid, label='Prediction (Full Photon)', color='r', linestyle='--')
    plt.plot(low_photon_50_gt_transient_mid, label='Ground Truth (Low Photon)', color='g', linestyle='-.')
    plt.plot(low_photon_50_pred_transient_mid, label='Prediction (Low Photon)', color='m', linestyle=':')
    
    plt.title('Comparison of Ground Truth and Predictions for Full and Low Photon Transient Data')
    plt.xlabel('Time')
    plt.ylabel('Normal Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("all transients plotted together")
    # iou = calc_transient_iou(pred_transient_array, gt_transient_array)
    # print(f"IOU for {scene_name}: ", iou)
        
def calc_transient_iou(gt, pred):
    intersection = np.minimum(pred, gt)
    union = np.maximum(pred, gt)
    iou = np.sum(intersection) / np.sum(union)
    return iou

if __name__ == "__main__":
    main()  
