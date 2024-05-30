import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
from scipy.ndimage import correlate1d
import os
import matplotlib.pyplot as plt
import imageio
import imgviz

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
from pytorch_lightning.loggers import TensorBoardLogger


import models
from models.ray_utils import get_rays, spatial_filter
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, MSELoss, L1_depth, MSE_depth, get_gt_depth, get_depth_from_transient, SSIM, LPIPS, KL_DIV, TransientIOU
from skimage.metrics import structural_similarity 


@systems.register('captured-nerf-system')
class CapturedNeRFSystem(BaseSystem):
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
        # self.register_buffer("train_rep", torch.tensor(1, dtype=torch.long))

    
    def forward(self, batch):
        return self.model(batch["rays"], batch["laser_kernel"])
    
    def preprocess_data(self, batch, stage):
        #This populates the batch to feed into forward
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        index = index.cpu()
        if stage in ['train']:

            #Randomly select some pixels during training        
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).float().to(self.rank)
            x = x.to(self.rank)
            y = y.to(self.rank)  
            s_x, s_y, weights = spatial_filter(x, y, sigma=self.dataset.rfilter_sigma, rep = self.dataset.rep, prob_dithering=self.dataset.sample_as_per_distribution)
            s_x = torch.clip(x + torch.from_numpy(s_x).to(self.rank), 0, self.dataset.w)
            s_y = torch.clip(y + torch.from_numpy(s_y).to(self.rank), 0, self.dataset.h)
            weights = torch.Tensor(weights).to(self.rank)
            
        elif stage in ["validation"]:
            #Selecting all tuples (x,y) of the image
            x, y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            x = x.flatten()
            y = y.flatten()
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x = x
            s_y = y 
            
            
        elif stage in ["test"]:
            x, y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            x = x.flatten()
            y = y.flatten()
            rgb = self.dataset.all_images[index, y, x]
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x, s_y, weights = spatial_filter(x, y, sigma=self.dataset.rfilter_sigma, rep = self.dataset.rep, prob_dithering=self.dataset.sample_as_per_distribution)
            s_x = (torch.clip(x + torch.from_numpy(s_x).to(self.rank), 0, self.dataset.w).to(torch.float32))
            s_y = (torch.clip(y + torch.from_numpy(s_y).to(self.rank), 0, self.dataset.h).to(torch.float32))
            weights = torch.Tensor(weights).to(self.rank)
            
            
        c2w = self.dataset.all_c2w[index] # (n_rays, 4, 4), repeats the same c2w matrix for all rays belonging to the same image
        camera_dirs = self.dataset.K(s_x, s_y) #NOTE: one big difference between synthetic dataset and captured dataset is this camera directions transformation
        rays_d = (camera_dirs[:, None, :] * c2w[:, :3, :3].to(self.rank)).sum(dim=-1).float()
        rays_o = torch.broadcast_to(c2w[:,:3, -1], rays_d.shape).to(self.rank)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1).float()

        
        if self.config.model.background_color == 'white':
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        elif self.config.model.background_color == 'random':
            self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
        elif self.config.model.background_color == "black":
            self.model.background_color = torch.zeros(3, dtype = torch.float32, device=self.rank)
        else:
            raise NotImplementedError
        
        # if self.dataset.apply_mask:
        #     rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])        
        
        if stage in ["train"]:
            batch.update({
                'rays': rays,
                'rgb': rgb.view(-1, self.dataset.n_bins*3),
                'weights': weights,
                'laser_kernel': self.dataset.laser_kernel
            })
            
        elif stage in ["validation"]:
            batch.update({
                'rays': rays,
                'laser_kernel': self.dataset.laser_kernel
            })
        
        elif stage in ["test"]:
            batch.update({
                'rays': rays,
                'rgb': rgb.view(-1, self.dataset.n_bins*3),
                'weights': weights,
                'laser_kernel': self.dataset.laser_kernel
            })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.
        #doen't do anything if self.dataset.rep=1
        gt_pixs = torch.reshape(batch["rgb"][:int(batch["rays"].shape[0]/self.dataset.rep)], (-1, self.dataset.n_bins, 3)) #Shape (n_rays,1200, 3)
        
        alive_ray_mask = out["rays_valid"].squeeze(-1) #(n_rays, 1)
        alive_ray_mask = alive_ray_mask.reshape(self.dataset.rep, -1)
        alive_ray_mask = alive_ray_mask.sum(0).bool() #shape of (num_rays,), #binary mask
        
        predicted_rgb = torch.reshape(out["rgb"], (-1, self.dataset.n_bins, 3))*batch["weights"][:, None, None] #(16, 1200, 3)
        comp_weights = out["comp_weights"][gt_pixs.sum(-1).repeat(self.dataset.rep, 1)<1e-7].mean() #space carving loss (1,)
        
        
        #the code below only makes a difference if self.rep > 1
        rgb = torch.zeros((int(predicted_rgb.shape[0]/self.dataset.rep), self.dataset.n_bins, 3)).to(self.rank)
        index = torch.arange(int(predicted_rgb.shape[0]/self.dataset.rep)).repeat(self.dataset.rep)[:, None, None].expand(-1, self.dataset.n_bins, 3).to(self.rank) #(?,1200,3)
        rgb.scatter_add_(0, index.type(torch.int64), predicted_rgb)
        
        gt_pixs = torch.log(gt_pixs + 1) #(num_ray,n_bins,RGB)
        predicted_rgb = torch.log(rgb + 1) # (num_rays, n_bins, RGB)

        #photometrics loss
        loss_rgb = F.l1_loss(predicted_rgb[alive_ray_mask], gt_pixs[alive_ray_mask])
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)
        
        #space carving loss proposed in https://arxiv.org/pdf/2307.09555.pdf
        space_carving_loss = comp_weights
        self.log('train/space_carving_loss', space_carving_loss)
        loss += space_carving_loss * self.C(self.config.system.loss.lambda_space_carving) 
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        return {
            'loss': loss
        }
    
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
        out = self(batch)
        predicted_image = out["rgb"].reshape(self.dataset.w, self.dataset.h, self.model.n_bins, 3)  #(512, 512, 1500, 3)
        predicted_image = predicted_image.sum(-2)   #(512,512,3)
        predicted_image = predicted_image.mean(-1) #(512,512)
        predicted_image = torch.stack([predicted_image, predicted_image, predicted_image], -1) #(512,512,3)
        predicted_image = (predicted_image/predicted_image.max())**(1/2.2) #(512,512,3)
        W, H = self.dataset.img_wh
        
        depth_image = (out["depth"]*torch.squeeze(out["opacity"])).view(H,W)
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': predicted_image.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': depth_image, 'kwargs': {}}
        ])
        # psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        return True
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        # if self.trainer.is_global_zerosel:
        #     out_set = {}
        #     for step_out in out:
        #         # DP
        #         if step_out['index'].ndim == 1:
        #             out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
        #         # DDP
        #         else:
        #             for oi, index in enumerate(step_out['index']):
        #                 out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
        #     psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
        #     self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):  
        #This computes metrics for each image individually, the next method aggregates all
        W, H = self.dataset.img_wh
        
        rgb = np.zeros((H, W, self.dataset.n_bins,3))
        depth = np.zeros((H, W))
        opacity = np.zeros((H, W))
        depth_viz = np.zeros((H, W))
        weights_sum = 0
        
        
        rep_number=30
        for j in range(rep_number):
            self.print(f"At rep {j}")
            gt_pixs = batch["rgb"].view(H, W, self.dataset.n_bins,3).detach().cpu().numpy()
            sample_weights = batch["weights"]
            out = self(batch)
            depth += (torch.squeeze(out["depth"]*sample_weights.detach().cpu()).reshape(H, W)).detach().cpu().numpy()
            depth_viz += (out["depth"]*sample_weights.detach().cpu()*torch.squeeze((out["opacity"]>0))).reshape(H, W).detach().cpu().numpy()
            opacity += (torch.squeeze(out["opacity"]) *sample_weights.detach().cpu().numpy()).reshape(H,W).detach().cpu().numpy()
            rgb += (out["rgb"] * sample_weights[:,None][:,None].detach().cpu().numpy()).reshape(H, W, self.dataset.n_bins, 3).detach().cpu().numpy()        
            weights_sum += sample_weights.detach().cpu().numpy()
            del out
        
        rgb = rgb / weights_sum.reshape(H, W, 1,1)
        depth = depth / weights_sum.reshape(H, W)
        opacity = opacity/weights_sum.reshape(H,W)
        depth_viz = depth_viz / weights_sum.reshape(H, W)     
        depth_image = depth * opacity   
        gt_pixs = gt_pixs.reshape(H, W, self.dataset.n_bins, 3)
        lm = correlate1d(gt_pixs[..., 0], self.dataset.laser.cpu().numpy(), axis=-1)
        exr_depth = np.argmax(lm, axis=-1)
        exr_depth = (exr_depth*2*299792458*4e-12)/2
        mask = (gt_pixs.sum((-1, -2)) > 0) # (H,W)
        
        
        #1. Calculate the L1 and MSE error b/w rendered depth vs gt depths (gt depths can be extracted using the get_gt_depth function)
        l1_depth = self.criterions["l1_depth"](exr_depth, depth, mask)
        # MSE_depth = self.criterions["MSE_depth"](exr_depth, depth, mask)
        
        #2. Calculate the L1 and MSE error b/w lg_depth from predicted transient and gt depth
        # lg_depth = correlate1d(rgb[..., 0], self.dataset.laser.cpu().numpy(), axis=-1)
        # lg_depth = np.argmax(lg_depth, axis=-1)
        # lg_depth = (lg_depth*self.model.exposure_time)/2
        # l1_depth_gt_lm_ours = self.criterions["l1_depth"](exr_depth, lg_depth, mask)
        # MSE_depth_gt_lm_ours = self.criterions["MSE_depth"](exr_depth, lg_depth, mask)
        
        
        #3. First get the transient images, sum across all transients (dim=-2) and gamma correct (gt and predicted)
        scaling = self.dataset.focal
        rgb_image = torch.from_numpy(rgb.sum(axis=-2))
        rgb_image = (rgb_image*self.dataset.transient_scale/self.dataset.div_vals[self.config.dataset.scene]) ** (1 / 2.2)
        
        ground_truth_image = torch.from_numpy(gt_pixs.sum(axis=-2))
        ground_truth_image = (scaling*ground_truth_image/self.dataset.div_vals[self.config.dataset.scene]) ** (1 / 2.2)
    
        #4. MSE between transient images vs predicted images
        # mse = self.criterions["MSE"](ground_truth_image.cpu(), rgb_image)
        
        #4. Transient IOU
        iou = self.criterions["Transient_IOU"](rgb, gt_pixs)
        
        #5. PSNR between transient images vs predicted images
        psnr = self.criterions["psnr"](ground_truth_image.cpu(), rgb_image)
        
        #7. SSIM between transient images vs predicted images
        ssim = torch.tensor(self.criterions["SSIM"](ground_truth_image.detach().cpu().numpy(), rgb_image.detach().cpu().numpy()), dtype=torch.float64)
        
        #8. LPIPS between transient images vs predicted images
        lpips = torch.tensor(self.criterions["LPIPS"](ground_truth_image.detach().cpu().numpy(), rgb_image.detach().cpu().numpy()), dtype=torch.float64)
        #9. KL divergence between transient images normalized vs predicted images normalized
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth.png"), exr_depth, cmap='inferno', vmin=0.8, vmax=1.5)
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_viz.png"), depth_viz, cmap='inferno', vmin=0.8, vmax=1.5)
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_image.png"), depth_image, cmap='inferno', vmin=0.8, vmax=1.5)
        
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_array"), depth)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_transient_array"), rgb)
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_predicted_RGB.png"), (rgb_image.numpy()*255.0).astype(np.uint8))
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_gt_RGB.png"), (ground_truth_image.numpy()*255.0).astype(np.uint8))
        
        self.save_image_plot_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_image, 'kwargs': {"title": "Predicted Integrated Transient"}},
            {'type': 'rgb', 'img': ground_truth_image, 'kwargs': {"title": "Ground Truth Integrated Transient"}},
            {'type': 'depth', 'img': depth_viz, 'kwargs': {"title": "Predicted Depth", "cmap": "inferno", "vmin":0.8, "vmax":1.5},},
            {'type': 'depth', 'img': exr_depth, 'kwargs': {"title": "Ground Truth Depth", "cmap": "inferno", "vmin":0.8, "vmax":1.5},},
        ])
        
        
        W, H = self.dataset.img_wh
        
        return {
            'l1_depth': l1_depth,
            # 'MSE_depth': MSE_depth,
            # 'l1_depth_gt_lm_ours':l1_depth_gt_lm_ours,
            # 'MSE_depth_gt_lm_ours': MSE_depth_gt_lm_ours,
            'psnr': psnr,
            # 'mse': mse,
            'ssim': ssim,
            'lpips': lpips,
            'iou': iou,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    index = step_out['index'].item()
                    if index not in out_set:
                        out_set[index] = {}
                    out_set[index]['l1_depth'] = step_out['l1_depth']
                    out_set[index]['MSE_depth'] = step_out['MSE_depth']
                    out_set[index]['l1_depth_gt_lm_ours'] = step_out['l1_depth_gt_lm_ours']
                    out_set[index]['MSE_depth_gt_lm_ours'] = step_out['MSE_depth_gt_lm_ours']
                    out_set[index]['psnr'] = step_out['psnr']
                    out_set[index]['mse'] = step_out['mse']
                    out_set[index]['ssim'] = step_out['ssim']
                    out_set[index]['lpips'] = step_out['lpips']
                # DDP
                else:
                    for oi, index_tensor in enumerate(step_out['index']):
                        index = index_tensor[0].item()
                        if index not in out_set:
                            out_set[index] = {}
                        out_set[index]['l1_depth'] = step_out['l1_depth'][oi]
                        # out_set[index]['MSE_depth'] = step_out['MSE_depth'][oi]
                        # out_set[index]['l1_depth_gt_lm_ours'] = step_out['l1_depth_gt_lm_ours'][oi]
                        # out_set[index]['MSE_depth_gt_lm_ours'] = step_out['MSE_depth_gt_lm_ours'][oi]
                        out_set[index]['psnr'] = step_out['psnr'][oi]
                        # out_set[index]['mse'] = step_out['mse'][oi]
                        out_set[index]['ssim'] = step_out['ssim'][oi]
                        out_set[index]['lpips'] = step_out['lpips'][oi]
                        out_set[index]['iou'] = step_out['iou'][oi]
                        
            rank_zero_info("Aggregating Metrics...")     
            l1_depth = torch.mean(torch.stack([o['l1_depth'] for o in out_set.values()]))
            # MSE_depth = torch.mean(torch.stack([o['MSE_depth'] for o in out_set.values()]))
            # l1_depth_gt_lm_ours = torch.mean(torch.stack([o['l1_depth_gt_lm_ours'] for o in out_set.values()]))
            # MSE_depth_gt_lm_ours = torch.mean(torch.stack([o['MSE_depth_gt_lm_ours'] for o in out_set.values()]))
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            # mse = torch.mean(torch.stack([o['mse'] for o in out_set.values()]))
            ssim = torch.mean(torch.stack([o['ssim'] for o in out_set.values()]))
            lpips = torch.mean(torch.stack([o['lpips'] for o in out_set.values()]))
            iou = torch.mean(torch.stack([o['iou'] for o in out_set.values()]))
            
            # self.log('test/chamfer_distance', chamfer_distance, prog_bar=True, rank_zero_only=True, sync_dist=True)
            self.log('test/l1_depth', l1_depth, prog_bar=True, rank_zero_only=True, sync_dist=True)    
            # self.log('test/MSE_depth', MSE_depth, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/l1_depth_gt_lm_ours', l1_depth_gt_lm_ours, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/MSE_depth_gt_lm_ours', MSE_depth_gt_lm_ours, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/mse', mse, prog_bar=True, rank_zero_only=True,sync_dist=True) 
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            self.log('test/ssim', ssim, prog_bar=True, rank_zero_only=True,sync_dist=True)
            self.log('test/lpips', lpips, prog_bar=True, rank_zero_only=True,sync_dist=True)       
            self.log('test/iou', iou, prog_bar=True, rank_zero_only=True, sync_dist=True)     
            
            metrics_dict = {"L1 Depth": l1_depth.item(), "PSNR": psnr.item(), "SSIM": ssim.item(), "LPIPS": lpips.item(), "IOU": iou.item()}
            self.save_metrics(f"it{self.global_step}-test.txt", metrics_dict)

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=10
            )
            
            rank_zero_info("Exporting mesh...")
            self.export()
            rank_zero_info("Mesh finished exporting")

    def export(self):
        mesh = self.model.export(self.config.export)
        mesh_path = self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}-{self.dataset.n_views}_{self.dataset.scene}.obj",
            **mesh
        )    
        return mesh_path
