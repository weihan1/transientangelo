import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
from pytorch_lightning.loggers import TensorBoardLogger
import imageio

import models
from models.ray_utils import get_rays, spatial_filter
from models.mesh_utils import calculate_chamfer_distance
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, MSELoss, L1_depth, MSE_depth, get_gt_depth, get_depth_from_transient, SSIM, LPIPS, TransientIOU
from skimage.metrics import structural_similarity 


@systems.register('transient-nerf-system')
class TransientNeRFSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        #TODO: Implement metrics without the percentage error first 
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
        self.register_buffer("train_rep", torch.tensor(1, dtype=torch.long))

    def forward(self, batch):
        #TODO: Possible extension here where i pass in not only the rays but also the gt transients 
        return self.model(batch['rays'])
    
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
            
            #Multi-ray shooting - Important for convergence
            if self.config.model.multi_ray_shooting:
                if self.trainer.global_step%1000==0:
                    if self.train_rep <30:
                        self.train_rep += 2
                    
            #Randomly select some pixels during training        
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            x = x.repeat(self.train_rep)
            y = y.repeat(self.train_rep)
            index = index.repeat(self.train_rep)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).float().to(self.rank)
            
        elif stage in ["validation"]:
            #Selecting all tuples (x,y) of the image
            x, y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            x = x.flatten()
            y = y.flatten()
            x = x.repeat(self.train_rep) 
            y = y.repeat(self.train_rep)  
            
            
        elif stage in ["test"]:
            self.train_rep = torch.tensor(1, dtype=torch.long)
            x, y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            x = x.flatten()
            y = y.flatten()
            x = x.repeat(self.train_rep) 
            y = y.repeat(self.train_rep)  
            rgb = self.dataset.all_images[index, y, x]
        
        c2w = self.dataset.all_c2w[index]
        scale = self.dataset.rfilter_sigma   
        
            
        if stage in ['train']:  
            x = x.to(self.rank)
            y = y.to(self.rank)       
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.train_rep, prob_dithering=self.dataset.sample_as_per_distribution)
            s_x = torch.clip(x + torch.from_numpy(s_x).to(self.rank), 0, self.dataset.w)
            s_y = torch.clip(y + torch.from_numpy(s_y).to(self.rank), 0, self.dataset.h)
            weights = torch.Tensor(weights).to(self.rank)
            
        
        elif stage in ["validation"]:
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x = x
            s_y = y
        
        elif stage in ["test"]:
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.train_rep, prob_dithering=self.dataset.sample_as_per_distribution, normalize=False)
            s_x = (torch.clip(x + torch.from_numpy(s_x).to(self.rank), 0, self.dataset.w).to(torch.float32))
            s_y = (torch.clip(y + torch.from_numpy(s_y).to(self.rank), 0, self.dataset.h).to(torch.float32))
            weights = torch.Tensor(weights).to(self.rank)
            
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
        rays_d = (camera_dirs[:, None, :] * c2w[:, :3, :3].to(self.rank)).sum(dim=-1)
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
                'weights': weights
            })
            
        elif stage in ["validation"]:
            batch.update({
                'rays': rays
            })
        
        elif stage in ["test"]:
            batch.update({
                'rays': rays,
                'rgb': rgb.view(-1, self.dataset.n_bins*3),
                'weights': weights
            })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        
        #doen't do anything if self.train_rep=1
        gt_pixs = torch.reshape(batch["rgb"][:int(batch["rays"].shape[0]/self.train_rep)], (-1, self.dataset.n_bins, 3)) #Shape (n_rays,1200, 3)
        
        alive_ray_mask = out["rays_valid"].squeeze(-1) #(n_rays, 1)
        alive_ray_mask = alive_ray_mask.reshape(self.train_rep, -1)
        alive_ray_mask = alive_ray_mask.sum(0).bool() #shape of (num_rays,), #binary mask
        
        predicted_rgb = torch.reshape(out["rgb"], (-1, self.dataset.n_bins, 3))*batch["weights"][:, None, None] #(16, 1200, 3)
        comp_weights = out["comp_weights"][gt_pixs.sum(-1).repeat(self.train_rep, 1)<1e-7].mean() #space carving loss (1,)
        
        
        #the code below only makes a difference if self.rep > 1
        rgb = torch.zeros((int(predicted_rgb.shape[0]/self.train_rep), self.dataset.n_bins, 3)).to(self.rank)
        index = torch.arange(int(predicted_rgb.shape[0]/self.train_rep)).repeat(self.train_rep)[:, None, None].expand(-1, self.dataset.n_bins, 3).to(self.rank) #(?,1200,3)
        rgb.scatter_add_(0, index.type(torch.int64), predicted_rgb)
        
        gt_pixs = torch.log(gt_pixs + 1) #(num_ray,n_bins,RGB)
        predicted_rgb = torch.log(rgb + 1) # (num_rays, n_bins, RGB)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / ((out['num_samples']).sum().item() + 1e-5)))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        #normal loss, l1 loss works pretty good
        loss_rgb = F.l1_loss(predicted_rgb[alive_ray_mask], gt_pixs[alive_ray_mask])
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)

        
        #space carving loss proposed in https://arxiv.org/pdf/2307.09555.pdf
        space_carving_loss = comp_weights
        self.log('train/space_carving_loss', space_carving_loss)
        loss += space_carving_loss * self.C(self.config.system.loss.lambda_space_carving) 
        
        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        self.log("train/rep", float(self.train_rep), prog_bar=True)

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
        #Doesn't do anything if self.train_rep = 1
        predicted_transient = torch.reshape(out["rgb"][:int(batch["rays"].shape[0]/self.train_rep)], (-1, 1200, 3))
        integrated_transient = predicted_transient.sum(-2)
        integrated_transient = (integrated_transient / integrated_transient.max()) ** (1 / 2.2)
        
        W, H = math.sqrt(integrated_transient.shape[0]), math.sqrt(integrated_transient.shape[0])
        
        if W.is_integer() and H.is_integer(): #Checks if W is an integer even if it has decimal part
            W,H = int(W), int(H)
        else:
            return True
        depth = torch.reshape(out["depth"][:int(batch["rays"].shape[0]/self.train_rep)], (H, W))
        opacity = torch.reshape(out["opacity"][:int(batch["rays"].shape[0]/self.train_rep)], (H, W))
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': integrated_transient.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': (depth*opacity).view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': opacity.view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
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
        meta_data = self.dataset.meta
        
        # #Initialize all gt tensors
        gt_pixs = batch["rgb"].view(H, W, self.dataset.n_bins,3).detach().cpu().numpy()
        # #Get predicted and ground_truth depth 
        gt_depth = get_gt_depth(meta_data["frames"][batch_idx], self.dataset.all_c2w[batch_idx].cpu().numpy(), self.dataset.depth_path) #(512,512)
        if self.config.dataset.downsample: #downsampling the gt depth to match that of the predicted depth
            gt_depth = (gt_depth[1::2, ::2] + gt_depth[::2, ::2] + gt_depth[::2, 1::2] + gt_depth[1::2, 1::2])/4
        weights_sum = 0
        rgb = np.zeros((H, W, self.dataset.n_bins, 3))
        depth = np.zeros((H, W))
        opacity = np.zeros((H, W))
        
        rep_number = 30
        for j in range(rep_number):
            self.print(f"At rep {j}")
            sample_weights = batch["weights"]
            out = self(batch)
            depth += (torch.squeeze(out["depth"]*sample_weights.detach().cpu()).reshape(W, H)).detach().cpu().numpy()
            rgb += (out["rgb"] * sample_weights[:, None][:, None].detach().cpu()).reshape(H, W, self.dataset.n_bins, 3).detach().cpu().numpy()        
            weights_sum += sample_weights.detach().cpu().numpy()
            opacity += (torch.squeeze(out["opacity"]) *sample_weights.detach().cpu().numpy()).reshape(H,W).detach().cpu().numpy()
            del out
        
        rgb = rgb/weights_sum.reshape(H, W, 1,1)
        depth = depth/weights_sum.reshape(H,W)
        opacity = opacity/weights_sum.reshape(H,W)
        mask = (gt_pixs.sum((-1, -2)) > 0) # (H,W)
        # #1. Calculate the L1 and MSE error b/w rendered depth vs gt depths (gt depths can be extracted using the get_gt_depth function)
        l1_depth = self.criterions["l1_depth"](gt_depth, depth, mask)
        depth_image = depth*opacity    
        # MSE_depth = self.criterions["MSE_depth"](exr_depth, depth, mask)
        
        # #2. Calculate the L1 and MSE error b/w lg_depth from predicted transient and gt depth
        # lg_depth = get_depth_from_transient(rgb, exposure_time=0.01, tfilter_sigma=3)
        # l1_depth_gt_lm_ours = self.criterions["l1_depth"](exr_depth, lg_depth, mask)
        # MSE_depth_gt_lm_ours = self.criterions["MSE_depth"](exr_depth, lg_depth, mask)
        
        # #3. Calculate the L1 and MSE error b/w lg_depth from gt transient and predicted depth
        # gt_depth = get_depth_from_transient(gt_pixs, exposure_time=0.01, tfilter_sigma=3)
        # l1_depth_gt_lm_gt = self.criterions["l1_depth"](exr_depth, gt_depth, mask)
        # MSE_depth_gt_lm_gt = self.criterions["MSE_depth"](exr_depth, gt_depth, mask)

        #4. Transient IOU
        iou = self.criterions["Transient_IOU"](rgb, gt_pixs)

        
        # #4. First get the transient images, sum across all transients (dim=-2) and gamma correct (gt and predicted)
        rgb_image = rgb.sum(axis=-2)*self.dataset.view_scale/self.dataset.dataset_scale
        ground_truth_image = gt_pixs.sum(axis=-2)*np.array(self.dataset.max)/self.dataset.dataset_scale
        rgb_image = np.clip(rgb_image,0,1) ** (1 / 2.2)
        data_image = (ground_truth_image) ** (1 / 2.2)
        
        # #5. MSE between transient images vs predicted images
        # mse = self.criterions["MSE"](torch.from_numpy(data_image), torch.from_numpy(rgb_image))
        
        # #6. PSNR between transient images vs predicted images
        psnr = self.criterions["psnr"](torch.from_numpy(data_image), torch.from_numpy(rgb_image))
        
        # #7. SSIM between transient images vs predicted images
        ssim = torch.tensor(self.criterions["SSIM"](data_image, rgb_image), dtype=torch.float64)
        
        # #8. LPIPS between transient images vs predicted images
        lpips = torch.tensor(self.criterions["LPIPS"](data_image, rgb_image), dtype=torch.float64)
        
        # #9. KL divergence between transient images normalized vs predicted images normalized
        
        
        #Preprocess the exr_depth to have a black background
        gt_depth[gt_depth == gt_depth[0][0]] = 0
        
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth.png"), gt_depth, cmap='inferno', vmin=2.5, vmax=5.5)
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_viz.png"), depth_image, cmap='inferno', vmin=2.2, vmax=5.5)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_array"), depth)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_transient_array"), rgb)
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_predicted_RGB.png"), (rgb_image*255.0).astype(np.uint8))
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_gt_RGB.png"), (data_image*255.0).astype(np.uint8))
        self.save_image_plot_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_image, 'kwargs': {"title": "Predicted Integrated Transient"}},
            {'type': 'rgb', 'img': data_image, 'kwargs': {"title": "Ground Truth Integrated Transient"}},
            {'type': 'depth', 'img': depth_image, 'kwargs': {"title": "Predicted Depth", "cmap": "inferno", "vmin":2.5, "vmax":5.5},},
            {'type': 'depth', 'img': gt_depth, 'kwargs': {"title": "Ground Truth Depth", "cmap": "inferno", "vmin":2.5, "vmax":5.5},},
        ])
        
        
        W, H = self.dataset.img_wh
        
        return {
            'l1_depth': l1_depth,
            # 'MSE_depth': MSE_depth,
            # 'l1_depth_gt_lm_ours':l1_depth_gt_lm_ours,
            # 'MSE_depth_gt_lm_ours': MSE_depth_gt_lm_ours,
            # 'l1_depth_gt_lm_gt': l1_depth_gt_lm_gt,
            # 'MSE_depth_gt_lm_gt': MSE_depth_gt_lm_gt,
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
            self.print("Exporting Mesh...")
            mesh_path = self.export()
            self.print("Calculating Chamfer Distance...")
            chamfer_distance = calculate_chamfer_distance(mesh_path, self.config.export.ref_mesh_path, self.config.export.num_samples)
            
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
                    out_set[index]['l1_depth_gt_lm_gt'] = step_out['l1_depth_gt_lm_gt']
                    out_set[index]['MSE_depth_gt_lm_gt'] = step_out['MSE_depth_gt_lm_gt']
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
                        # out_set[index]['l1_depth_gt_lm_gt'] = step_out['l1_depth_gt_lm_gt'][oi]
                        # out_set[index]['MSE_depth_gt_lm_gt'] = step_out['MSE_depth_gt_lm_gt'][oi]
                        out_set[index]['psnr'] = step_out['psnr'][oi]
                        # out_set[index]['mse'] = step_out['mse'][oi]
                        out_set[index]['ssim'] = step_out['ssim'][oi]
                        out_set[index]['lpips'] = step_out['lpips'][oi]
                        out_set[index]['iou'] = step_out['iou'][oi]
                        
            self.print("Aggregating the results...")    
            l1_depth = torch.mean(torch.stack([o['l1_depth'] for o in out_set.values()]))
            # MSE_depth = torch.mean(torch.stack([o['MSE_depth'] for o in out_set.values()]))
            # l1_depth_gt_lm_ours = torch.mean(torch.stack([o['l1_depth_gt_lm_ours'] for o in out_set.values()]))
            # MSE_depth_gt_lm_ours = torch.mean(torch.stack([o['MSE_depth_gt_lm_ours'] for o in out_set.values()]))
            # l1_depth_gt_lm_gt = torch.mean(torch.stack([o['l1_depth_gt_lm_gt'] for o in out_set.values()]))
            # MSE_depth_gt_lm_gt = torch.mean(torch.stack([o['MSE_depth_gt_lm_gt'] for o in out_set.values()]))
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            # mse = torch.mean(torch.stack([o['mse'] for o in out_set.values()]))
            ssim = torch.mean(torch.stack([o['ssim'] for o in out_set.values()]))
            lpips = torch.mean(torch.stack([o['lpips'] for o in out_set.values()]))
            iou = torch.mean(torch.stack([o['iou'] for o in out_set.values()]))
            
            self.log('test/chamfer_distance', chamfer_distance, prog_bar=True, rank_zero_only=True, sync_dist=True)
            self.log('test/l1_depth', l1_depth, prog_bar=True, rank_zero_only=True, sync_dist=True)    
            # self.log('test/MSE_depth', MSE_depth, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/l1_depth_gt_lm_ours', l1_depth_gt_lm_ours, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/MSE_depth_gt_lm_ours', MSE_depth_gt_lm_ours, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/l1_depth_gt_lm_gt', l1_depth_gt_lm_gt, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/MSE_depth_gt_lm_gt', MSE_depth_gt_lm_gt, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            # self.log('test/mse', mse, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True,sync_dist=True)        
            self.log('test/ssim', ssim, prog_bar=True, rank_zero_only=True)
            self.log('test/lpips', lpips, prog_bar=True, rank_zero_only=True)
            self.log('test/iou', iou, prog_bar=True, rank_zero_only=True, sync_dist=True)
            
            metrics_dict = {"Chamfer Distance": chamfer_distance, "L1 Depth": l1_depth.item(), "PSNR": psnr.item(), "SSIM": ssim.item(), "LPIPS": lpips.item(), "Transient IOU": iou.item()}         
            self.save_metrics(f"it{self.global_step}-test.txt", metrics_dict)
            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=10
            )
            

    def export(self):
        mesh = self.model.export(self.config.export)
        mesh_path = self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )    
        return mesh_path
