from align_data import depth2dist, get_rays
from align_data import read_json_poses
import torch 
import scipy.io as sio
import numpy as np 
import os 
import json 
from read_depth import read_array
from align_data import plot_pcs
from scipy.ndimage import correlate1d
import tqdm
import matplotlib.pyplot as plt 
import re
import scipy

def read_json(fname):
    with open(
            os.path.join(fname), "r"
    ) as fp:
        meta = json.load(fp)
    # camtoworlds = []

    # for i in range(len(meta["frames"])):
    #     frame = meta["frames"][i]
    #     camtoworlds.append(frame["transform_matrix"])

    # camtoworlds = np.stack(camtoworlds, axis=0)

    return meta["frames"]

class LearnRays(torch.nn.Module):
    '''
    This is used to generate rays direction for the captured dataset
    '''
    def __init__(self, rays, device ="cuda:0", img_shape = (256, 256)):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnRays, self).__init__()
        self.device = device
        self.init_c2w = None
        self.img_shape = img_shape

        x = np.arange(32, 480)
        X, Y = np.meshgrid(x, x)

        tar_x = np.arange(0, 512)
        tar_X, tar_Y = np.meshgrid(tar_x, tar_x)
        # rays = rays.detach().cpu().numpy()

        ray_x = scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, 0].transpose(1, 0), np.stack([tar_X, tar_Y], axis=-1).squeeze().flatten(), bounds_error = False, fill_value=None).reshape(512, 512)
        ray_y = scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, 1].transpose(1, 0), np.stack([tar_X, tar_Y], axis=-1).squeeze().flatten(), bounds_error = False, fill_value=None).reshape(512, 512)
        ray_z = scipy.interpolate.interpn((x, x), rays[32:-32, 32:-32, 2].transpose(1, 0), np.stack([tar_X, tar_Y], axis=-1).squeeze().flatten(), bounds_error = False, fill_value=None).reshape(512, 512)

        rays = torch.from_numpy(np.stack([ray_x, ray_y, ray_z], axis=-1)).to(self.device)
        if img_shape[0] == 256:
            rays = (rays[::2, ::2] + rays[::2, 1::2] +  rays[1::2, ::2]+ rays[1::2, +1::2] )/4

        rays = rays/torch.linalg.norm(rays, dim=-1, keepdims=True)
        self.rays = rays

    def forward(self, x0, y0):
        """input coord = (n, 2)
        rays = (512, 512, 3)
        """
        rays = self.rays
        x1, y1 = torch.floor(x0.float()), torch.floor(y0.float())
        x2, y2 = x1+1, y1+1
        """
        Perform bilinear interpolation to estimate the value of the function f(x, y)
        at the continuous point (x0, y0), given that f is known at integer values of x, y.
        """
        if (y1>self.img_shape[0]-1).any() or (x1>self.img_shape[0]-1).any():
            print("hello")
        x1, y1 = torch.clip(x1, 0, self.img_shape[0]-1), torch.clip(y1, 0, self.img_shape[0]-1)

        # x2, y2 = torch.clip(x2, 0, self.img_shape[0]-1), torch.clip(y2, 0, self.img_shape[0]-1)

        # Compute the weights for the interpolation
        wx1 = ((x2 - x0) / (x2 - x1 + 1e-8))[:, None]
        wx2 = ((x0 - x1) / (x2 - x1 + 1e-8))[:, None]
        wy1 = ((y2 - y0) / (y2 - y1 + 1e-8))[:, None]
        wy2 = ((y0 - y1) / (y2 - y1 + 1e-8))[:, None]

        x1, y1, x2, y2 = x1.long(), y1.long(), x2.long(), y2.long()
        x2, y2 = torch.clip(x2, 0, self.img_shape[0] - 1), torch.clip(y2, 0, self.img_shape[0] - 1)

        # Compute the interpolated value of f(x, y) at (x0, y0)
        f_interp = wx1 * wy1 * rays[y1, x1] + \
                wx1 * wy2 * rays[y2, x1] + \
                wx2 * wy1 * rays[y1, x2] + \
                wx2 * wy2 * rays[y2, x2]

        f_interp = f_interp/torch.linalg.norm(f_interp, dim=-1, keepdims=True) 
        return f_interp.float()

def torch_laser_kernel(laser, device='cuda'):
    m = torch.nn.Conv1d(1, 1, laser.shape[0], padding=(laser.shape[0] - 1) // 2, padding_mode="zeros", device=device)
    m.weight.requires_grad = False
    m.bias.requires_grad = False
    m.bias *= 0
    m.weight = torch.nn.Parameter(laser[None, None, ...])
    return m


def convolve_colour(color, kernel, n_bins):
    color = color.transpose(1, 2).reshape(-1, n_bins)
    color = kernel(color[:, None, :]).squeeze()
    color = color.reshape(-1, 3, n_bins).transpose(1, 2)
    return color



def plot_from_depth():
    device = "cuda"
    dataset_scene = "chef_raxel"
    intrinsics = f"/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/captured_data/{dataset_scene}/david_calibration_1005/intrinsics_david.npy"
    params = np.load(intrinsics, allow_pickle=True)[()]
    shift = params['shift'].numpy()
    rays = params['rays']
    K = LearnRays(rays, device=device, img_shape=(512,512))
    json_file = f'/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/captured_data/{dataset_scene}/final_cams/two_views/unseen_train.json'
    # depths_path = "/home/anagh/PycharmProjects/multiview_lif/data/cinema_checkerboard_05_02_masked/dense/stereo/depth_maps"
    # camtoworlds = torch.from_numpy(read_json_poses(json_file))
    frames= read_json(json_file)
    
    
    n_bins = 1500
    colmap_pcs = []
    exposure_time = 299792458*4e-12
    x = (torch.arange(512, device="cpu")-512//2+0.5)/(512//2-0.5)
    y = (torch.arange(512, device="cpu")-512//2+0.5)/(512//2-0.5)
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
        
        
        
    for camid in range(len(frames)):
        frame = frames[camid]
        camtoworld = frame["transform_matrix"]
        camtoworld = torch.from_numpy(np.array(camtoworld))
        root_dir= "/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/captured_data/cinema_raxel"
        
        number = int(frame["file_path"].split("_")[-1])
        transient_path = os.path.join(root_dir,f"transient{number:03d}" + ".pt")
        rgba = torch.load(transient_path).to_dense() #r_i Loads the corresponding transient00i
        rgba = torch.Tensor(rgba)[..., :3000].float().cpu()
        if rgba.shape[0]==256:
            rgba = (rgba[::2, ::2] + rgba[::2, 1::2] +  rgba[1::2, ::2]+ rgba[1::2, 1::2] )/4
        
        rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
        rgba = (rgba[..., 1::2]+ rgba[..., ::2] )/2 #(512,512, 1500)
        rgba = torch.clip(rgba, 0, None)
        rgba[rgba<=1] = 0

        rgba = rgba[..., None].repeat(1, 1, 1, 3) #(512,512, 1500, 3)
        # camtoworld[:2, -1] *= -1
        img_wh = (512, 512)
        x, y = torch.meshgrid(
                torch.arange(img_wh[0],device="cuda"),
                torch.arange(img_wh[1],device="cuda"),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
        x = x.flatten()
        y = y.flatten()
        camera_dirs = K(x, y) 
        rays_d = (camera_dirs[:, None, :] * camtoworld[:3, :3].to(device)).sum(dim=-1).float()
        rays_o = torch.broadcast_to(camtoworld[:3, -1], rays_d.shape).to(device)

        filename = frame["file_path"].split("/")[-1][:-2]+"png"
        exposure_time = 0.002398339664
        gt_pixs = rgba
        
        laser_pulse_dic = sio.loadmat('/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/captured_data/pulse_low_flux.mat')['out'].squeeze()
        laser_pulse = laser_pulse_dic
        lidx = np.argmax(laser_pulse)
        loffset = 50
        laser = laser_pulse[lidx-loffset:lidx+loffset+1]
        laser = laser / laser.sum()
        laser = torch.tensor(laser[::-1].copy(), device="cuda").float()
        laser_kernel = torch_laser_kernel(laser, device="cuda")
        
        lm = correlate1d(gt_pixs[..., 0], laser.cpu().numpy(), axis=-1)
        exr_depth = np.argmax(lm, axis=-1)
        exr_depth = (exr_depth*exposure_time)/2


        # depth = torch.from_numpy(read_array(f"{depths_path}/{filename}.geometric.bin"))[..., None]
        
        # depth = depth2dist(depth, K)


        #scale up extrinics

        # get rays

        # convert colmap depth to dist
        # Note: (this is done when loading now, comment out)
        # colmap_dists[camid] = depth2dist(colmap_dists[camid], K)

        # get point cloud for colmap and transient
        # colmap_pc = origins.reshape(-1, 3)
        # viewdirs = viewdirs/np.abs(viewdirs[..., 2][..., None])
        # depth = depth/np.abs(viewdirs[..., 2][..., None])
        colmap_pc = (np.reshape(exr_depth, (262144,1)) * rays_d.cpu().detach().numpy() + rays_o.cpu().detach().numpy()).reshape(-1, 3)
        colmap_mask = (exr_depth < 60).flatten()*(exr_depth >0).flatten()
        colmap_pc = colmap_pc[colmap_mask, :]
        # colmap_pc = torch.concat([colmap_pc, origins.reshape(-1, 3)])
        
        # viewdirs[..., 1]*= -1
        # viewdirs[..., 2]*= -1

        
        # colmap_pc = (viewdirs[256, 256]*np.linspace(0, 3)[:, None]) + origins[0, 0]
        # colmap_pc = colmap_pc.reshape(-1, 3)
        colmap_pcs.append(colmap_pc)
        
    colmap_pc = np.concatenate(colmap_pcs, axis=0)
    plot_pcs([colmap_pc[::90], ])


if __name__=="__main__":
    plot_from_depth()
