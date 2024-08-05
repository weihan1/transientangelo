import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
from scipy.ndimage import correlate1d
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
import imageio
import models
from models.utils import cleanup
from models.ray_utils import get_rays, spatial_filter, generate_unseen_poses, find_surface_points
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy, MSELoss, L1_depth, MSE_depth, get_gt_depth, get_depth_from_transient, SSIM, LPIPS, KL_DIV, TransientIOU


@systems.register('movie-system')
class CapturedMovie(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'l1_depth': L1_depth(),
            'MSE_depth': MSE_depth(),
            'psnr': PSNR(),
            'MSE': MSELoss(),
            'SSIM': SSIM(), 
            'LPIPS': LPIPS(),
            'Transient_IOU': TransientIOU()
            # 'kl_div': KL_DIV()
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays
        self.unseen_rays = None
        self.annealing_factor = 1
        self.register_buffer("train_rep", torch.tensor(1, dtype=torch.long))

    def forward(self, batch):
        #NOTE: if using transient-based method on the captured dataset, then perform forward pass using laser kernel
        if hasattr(self.dataset, "laser_kernel") and "baseline" not in self.config.name:
            return self.model(batch['rays_dict'], batch["laser_kernel"])
        #NOTE: else, normal forward pass 
        else:
            return self.model(batch['rays_dict'])
    
    def preprocess_data(self, batch, stage):
        #This populates the batch to feed into forward
        if stage in ["train", "validation"]:
            print("THIS IS TO WATCH MOVIE NOT FOR TRAINING")
            exit(1)
        rays_dict = {}
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
                
        index = index.cpu()
            
        x, y = torch.meshgrid(
            torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
            torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
            indexing="xy",
        ) #(self.dataset.w, self.dataset.h)
        x = x.flatten()
        y = y.flatten()
        
        self.train_rep = torch.tensor(1, dtype=torch.long)
        x = x.to(self.rank)
        y = y.to(self.rank)
        s_x, s_y = x,y
            
        #Converts pixel coordinates to normalized camera coordinates    
        c2w = self.dataset.all_c2w[index]
        
        if hasattr(self.dataset, "laser_kernel"): #Different intinsic param for captured transient
            camera_dirs = self.dataset.K(s_x, s_y)
            
        else:
            camera_dirs = F.pad(
            torch.stack(
                [
                    (s_x - self.dataset.K[0, 2] + 0.5) / self.dataset.K[0, 0],
                    (s_y - self.dataset.K[1, 2] + 0.5)
                    / self.dataset.K[1, 1]
                    * (-1.0 if self.dataset.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.dataset.OPENGL_CAMERA else 1.0),
            )
                
        rays_d = (camera_dirs[:x.shape[0], None, :] * c2w[:, :3, :3].to(self.rank)).sum(dim=-1) #upper left corner of the c2w
        rays_o = torch.broadcast_to(c2w[:,:3, -1], rays_d.shape).to(self.rank) #Camera origins from the same view are the same 
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1).float() #(n_rays*rep, 6)
        rays_dict.update({"training_rays": {"rays": rays, 'rotation_matrix': c2w[:, :3, :3], 'index': index}})
        if self.trainer.global_step == 1:
            self.print("Training rays has been added!")
        
        rays_dict.update({"global_step": self.global_step})
        rays_dict.update({"stage": stage})
        
        if self.config.model.background_color == 'white':
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        elif self.config.model.background_color == 'random':
            self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
        elif self.config.model.background_color == "black":
            self.model.background_color = torch.zeros(3, dtype = torch.float32, device=self.rank)
        else:
            raise NotImplementedError
        
        
        batch.update({
            'rays_dict': rays_dict,
            'laser_kernel': self.dataset.laser_kernel if hasattr(self.dataset, "laser_kernel") else None
        })
    
    
    def training_step(self, batch, batch_idx):
        pass

    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        pass
        

    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    #NOTE: No metric to aggregate during validation
    # def validation_epoch_end(self, out):
    #     out = self.all_gather(out)
    #     # if self.trainer.is_global_zerosel:
    #     #     out_set = {}
    #     #     for step_out in out:
    #     #         # DP
    #     #         if step_out['index'].ndim == 1:
    #     #             out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
    #     #         # DDP
    #     #         else:
    #     #             for oi, index in enumerate(step_out['index']):
    #     #                 out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
    #     #     psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
    #     #     self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)   

    def test_step(self, batch, batch_idx):  
        
        W, H = self.dataset.img_wh
        
        out = self(batch)
            
        try:
            rgb = out["comp_rgb_full"].reshape(512,512,3).numpy() # (512,512,3) when loading baselines, we get images
            depth = out["depth"].reshape(512,512).numpy()
        except:
            rgb = out["rgb"].reshape(512,512,self.dataset.n_bins,3).numpy()
            rgb = rgb.sum(axis=-2)*self.dataset.view_scale/self.dataset.dataset_scale
            rgb = np.clip(rgb, 0, 1) ** (1 / 2.2)
            depth = out["depth"].reshape(512,512).numpy()
        
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_predicted_RGB.png"), (rgb*255.0).astype(np.uint8))
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_viz.png"), depth, cmap='inferno', vmin=0.8, vmax=1.5)

        return {
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                r'(\d+)_predicted_RGB\.png', #assuming the file is called *_predicted_RGB.png, where * is any natural number
                save_format='mp4',
                fps=30,
                reversed = True if "captured" in self.config.dataset else False #reverse the video if using the captured movie pose
            )
        print("ENJOY THE MOVIE")
            
    
    def export(self):
        mesh = self.model.export(self.config.export)
        mesh_path = self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )
        return mesh_path        
