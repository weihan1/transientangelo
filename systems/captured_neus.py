import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
from scipy.ndimage import correlate1d
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
from pytorch_lightning.loggers import TensorBoardLogger


import models
from models.ray_utils import get_rays, spatial_filter, generate_unseen_poses
from models.mesh_utils import calculate_chamfer_distance
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, MSELoss, L1_depth, MSE_depth, get_gt_depth, get_depth_from_transient, binary_cross_entropy, SSIM, LPIPS, KL_DIV, TransientIOU
from skimage.metrics import structural_similarity 


@systems.register('captured-neus-system')
class CapturedNeuSSystem(BaseSystem):
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
        self.register_buffer("train_rep", torch.tensor(1, dtype=torch.long))

    
    def forward(self, batch):
        return self.model(batch["rays_dict"], batch["laser_kernel"])
    
    def preprocess_data(self, batch, stage):
        #This populates the batch to feed into forward
        rays_dict = {}
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        index = index.cpu()
        if stage in ['train']:
            
            train_x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            train_y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            train_x = train_x.repeat(self.train_rep) #(n_rays*rep)
            train_y = train_y.repeat(self.train_rep) #(n_rays*rep)
            index = index.repeat(self.train_rep)
            rgb = self.dataset.all_images[index, train_y, train_x].view(-1, self.dataset.all_images.shape[-1]).float().to(self.rank) #(n_rays*rep, 1200, 3)
            
                
            train_x = train_x.to(self.rank)
            train_y = train_y.to(self.rank)
            train_sx, train_sy, weights = spatial_filter(train_x, train_y, sigma=self.dataset.rfilter_sigma, rep = self.train_rep, prob_dithering=self.dataset.sample_as_per_distribution)
            s_x = torch.clip(train_x + torch.from_numpy(train_sx).to(self.rank), 0, self.dataset.w) #Small perturbations to the pixel locations
            s_y = torch.clip(train_y + torch.from_numpy(train_sy).to(self.rank), 0, self.dataset.h)
            weights = torch.Tensor(weights).to(self.rank)
            
            if self.config.model.use_reg_nerf:
                #NOTE: the regnerf patch rays are always concatenated at the end
                patch_size = self.config.model.train_patch_size 
                n_sparse_rays = self.config.model.sparse_rays_size
                img_wh = self.dataset.w, self.dataset.h
                regnerf_x, regnerf_y, unseen_rotation_matrix, unseen_translation_vector = generate_unseen_poses(patch_size, img_wh, self.dataset.all_c2w, n_sparse_rays=n_sparse_rays, 
                                                                                                bounding_form =self.config.model.bounding_form, mean_focus_point = self.dataset.mean_focus_point,
                                                                                                sample_boundary = self.config.model.sample_boundary, write_poses = self.config.model.write_poses)
                regnerf_x, regnerf_y = regnerf_x.to(self.rank), regnerf_y.to(self.rank)
                unseen_rotation_matrix = torch.from_numpy(unseen_rotation_matrix).to(self.rank)
                #Calculate the rays for the unseen patch using the new rotation matrix and translation vector
                regnerf_sx, regnerf_sy, _ = spatial_filter(regnerf_x, regnerf_y, sigma=self.dataset.rfilter_sigma , rep = 1, prob_dithering=self.dataset.sample_as_per_distribution)
                regnerf_sx = torch.clip(regnerf_x + torch.from_numpy(regnerf_sx).to(self.rank), 0, self.dataset.w) #Small perturbations to the random pixel locations
                regnerf_sy = torch.clip(regnerf_y + torch.from_numpy(regnerf_sy).to(self.rank), 0, self.dataset.h)
                s_x, s_y = torch.cat([s_x, regnerf_sx], dim=0), torch.cat([s_y, regnerf_sy], dim=0)
            
        elif stage in ["validation"]:
            #Selecting all tuples (x,y) of the image
            train_x, train_y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            train_x = train_x.flatten()
            train_y = train_y.flatten()
            train_x = train_x.to(self.rank)
            train_y = train_y.to(self.rank)
            s_x = train_x
            s_y = train_y
            
            if self.config.model.use_reg_nerf:
                #NOTE: use this for visualization
                patch_size = 32
                n_sparse_rays = self.config.model.sparse_rays_size
                img_wh = self.dataset.w, self.dataset.h
                regnerf_x, regnerf_y, unseen_rotation_matrix, unseen_translation_vector = generate_unseen_poses(patch_size, img_wh, self.dataset.all_c2w, n_sparse_rays=n_sparse_rays, 
                                                                                                bounding_form =self.config.model.bounding_form, mean_focus_point = self.dataset.mean_focus_point,
                                                                                                sample_boundary = self.config.model.sample_boundary)
                regnerf_x, regnerf_y = regnerf_x.to(self.rank), regnerf_y.to(self.rank)
                unseen_rotation_matrix = torch.from_numpy(unseen_rotation_matrix).to(self.rank)
                #Calculate the rays for the unseen patch using the new rotation matrix and translation vector
                regnerf_sx, regnerf_sy, _ = spatial_filter(regnerf_x, regnerf_y, sigma=self.dataset.rfilter_sigma , rep = 1, prob_dithering=self.dataset.sample_as_per_distribution)
                regnerf_sx = torch.clip(regnerf_x + torch.from_numpy(regnerf_sx).to(self.rank), 0, self.dataset.w) #Small perturbations to the random pixel locations
                regnerf_sy = torch.clip(regnerf_y + torch.from_numpy(regnerf_sy).to(self.rank), 0, self.dataset.h)
                s_x, s_y = torch.cat([s_x, regnerf_sx], dim=0), torch.cat([s_y, regnerf_sy], dim=0)
            
            
        elif stage in ["test"]:
            train_x, train_y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            train_x = train_x.flatten()
            train_y = train_y.flatten()
            rgb = self.dataset.all_images[index, train_y, train_x]
            
            self.train_rep = torch.tensor(1, dtype=torch.long)
            train_x = train_x.to(self.rank)
            train_y = train_y.to(self.rank)
            s_x, s_y, weights = spatial_filter(train_x, train_y, sigma=self.dataset.rfilter_sigma, rep = self.dataset.rep, prob_dithering=self.dataset.sample_as_per_distribution, normalize=False)
            s_x = (torch.clip(train_x + torch.from_numpy(s_x).to(self.rank), 0, self.dataset.w).to(torch.float32))
            s_y = (torch.clip(train_y + torch.from_numpy(s_y).to(self.rank), 0, self.dataset.h).to(torch.float32))
            weights = torch.Tensor(weights).to(self.rank)
    
    
        
        c2w = self.dataset.all_c2w[index] # (n_rays, 4, 4), repeats the same c2w matrix for all rays belonging to the same image
        camera_dirs = self.dataset.K(s_x, s_y) 
        if stage in ["train", "validation"]:
            if self.config.model.use_reg_nerf:
            #NOTE: the regnerf patch rays are always concatenated at the end, used for rawnerf
                regnerf_rays_d = (camera_dirs[-self.config.model.train_patch_size**2:, None, :] * unseen_rotation_matrix.to(self.rank)).sum(dim=-1) #upper left corner of the c2w
                unseen_rays_o = torch.broadcast_to(unseen_translation_vector, regnerf_rays_d.shape).to(self.rank).float()
                unseen_rays = torch.cat([unseen_rays_o, F.normalize(regnerf_rays_d, p=2, dim=-1)], dim=-1).float() #(self.patch_size**2 + sparse_rays, 6)
                rays_dict.update({"regnerf_patch": {'rays': unseen_rays, 'rotation_matrix': unseen_rotation_matrix}})
                if self.trainer.global_step == 1:
                    self.print("Regnerf patch rays has been added!")
        
        rays_d = (camera_dirs[:train_x.shape[0], None, :] * c2w[:, :3, :3].to(self.rank)).sum(dim=-1) 
        #NOTE: in captured, we use pixel coordinates and interpolate the camera directions based on the known camera directions stored in rays
        rays_o = torch.broadcast_to(c2w[:,:3, -1], rays_d.shape).to(self.rank)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1).float()
        rays_dict.update({"training_rays": {"rays": rays, 'rotation_matrix': c2w[:, :3, :3], 'index': index}})
        if self.trainer.global_step == 1:
            self.print("Training rays has been added!")
        
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
                'rays_dict': rays_dict,
                'rgb': rgb.view(-1, self.dataset.n_bins*3),
                'weights': weights,
                'laser_kernel': self.dataset.laser_kernel
            })
            
        elif stage in ["validation"]:
            batch.update({
                'rays_dict': rays_dict,
                'laser_kernel': self.dataset.laser_kernel
            })
        
        elif stage in ["test"]:
            batch.update({
                'rays_dict': rays_dict,
                'rgb': rgb.view(-1, self.dataset.n_bins*3),
                'weights': weights,
                'laser_kernel': self.dataset.laser_kernel
            })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        
        training_rays = batch["rays_dict"]["training_rays"]["rays"]
        loss = 0.
        if self.config.model.use_reg_nerf:     
            if out["num_samples_regnerf"] > 0:       
                color_patch = out["color_patch"]
                depth_patch = out["depth_patch"]
                opacity_patch = out["opacity_patch"]
                transient_patch = out["transient_patch"]
                depth_variance_patch = out["depth_variance_patch"]
                sdf_grad_samples = out["sdf_grad_samples"]
                
                assert (color_patch == out["color_patch"][:64]).all()
                assert (depth_patch == out["depth_patch"][:64]).all()
                assert (transient_patch == out["transient_patch"][:64]).all()
                assert (depth_variance_patch == out["depth_variance_patch"][:64]).all()
                
                ##NOTE: Depth smoothness regularizer
                differences_row = depth_patch[:, 1:] - depth_patch[:, :-1] 
                differences_col = depth_patch[1:, :] - depth_patch[:-1, :]
                smoothness_loss_row = torch.sum((differences_row)**2)
                smoothness_loss_col = torch.sum((differences_col)**2)
                smoothness_loss = smoothness_loss_row + smoothness_loss_col
                loss += smoothness_loss * self.C(self.config.system.loss.lambda_depth_smoothness)

        
        #doen't do anything if self.dataset.rep=1
        gt_pixs = torch.reshape(batch["rgb"][:int(training_rays.shape[0]/self.dataset.rep)], (-1, self.dataset.n_bins, 3)) #Shape (n_rays,1200, 3)
        
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
        #Plot the predicted and gt images on top of each other
        
        
        if not self.global_step % 1000:
            self.save_plot(f"it{self.global_step}-transient_plot", torch.mean(predicted_rgb[alive_ray_mask], dim=(0,2)), torch.mean(gt_pixs[alive_ray_mask], dim=(0,2)))
            self.save_depth_plot(f"it{self.global_step}-transient_depth_plot", predicted_rgb[alive_ray_mask], gt_pixs[alive_ray_mask], out["depth"][alive_ray_mask], self.model.exposure_time)
            self.save_sdf_plot(f"it{self.global_step}-sdf_depth_plot", out["sdf_samples"], self.model.exposure_time, out["distances_from_origin"],out["depth"][alive_ray_mask], out["ray_indices"], alive_ray_mask)

            
            
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / ((out['num_samples']).sum().item() + 1e-5)))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        
        #integrated transient loss
        #Normalize + Gamma correct
        integrated_gt = gt_pixs[alive_ray_mask].sum(-2)
        # integrated_gt /= integrated_gt.max()
        # integrated_gt = integrated_gt**(1/2.2)
        
        integrated_pred = predicted_rgb[alive_ray_mask].sum(-2)
        # integrated_pred /= integrated_pred.max()
        # integrated_pred = integrated_pred**(1/2.2)
        
        loss_integrated = F.l1_loss(integrated_pred.mean(-1), integrated_gt[...,0], reduction="mean")
        loss += loss_integrated * self.C(self.config.system.loss.lambda_integrated_l1)
        
        
        
        #photometric loss
        loss_rgb_l1 = F.l1_loss(predicted_rgb.mean(-1)[alive_ray_mask], gt_pixs[...,0][alive_ray_mask])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        
        #space carving loss proposed in https://arxiv.org/pdf/2307.09555.pdf
        space_carving_loss = comp_weights
        self.log('train/space_carving_loss', space_carving_loss)
        loss += space_carving_loss * self.C(self.config.system.loss.lambda_space_carving) 
        
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal, prog_bar=True)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        # loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        # self.log('train/loss_mask', loss_mask)
        # loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)
        
        
        loss_variance = out["weight_variance"].mean()
        loss += loss_variance * self.C(self.config.system.loss.lambda_depth_variance)
        
        
        
        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        
        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)
            
        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)      
        
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        self.log('train/inv_s', out['inv_s'], prog_bar=True) 
        
        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
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
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': predicted_image.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': (out['depth']*torch.squeeze(out["opacity"])).view(H, W), 'kwargs': {}}
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
        
        rgb = np.zeros((H, W, self.dataset.n_bins, 3))
        depth = np.zeros((H, W))
        opacity = np.zeros((H,W))
        weights_sum = 0
        
        rep_number=30
        for j in range(1):
            self.print(f"At rep {j}")
            gt_pixs = batch["rgb"].view(H, W, self.dataset.n_bins,3).detach().cpu().numpy()
            sample_weights = batch["weights"]
            out = self(batch)
            depth += (torch.squeeze(out["depth"]*sample_weights.detach().cpu()).reshape(W, H)).detach().cpu().numpy()
            rgb += (out["rgb"] * sample_weights[:,None][:,None].detach().cpu().numpy()).reshape(H, W, self.dataset.n_bins, 3).detach().cpu().numpy()        
            opacity += (torch.squeeze(out["opacity"]) *sample_weights.detach().cpu().numpy()).reshape(H,W).detach().cpu().numpy()
            weights_sum += sample_weights.detach().cpu().numpy()
        
        
        rgb = rgb / weights_sum.reshape(H, W, 1,1) #(H,W,1500,3)
        depth = depth / weights_sum.reshape(H, W)
        opacity = opacity/weights_sum.reshape(H,W)
        depth_image = depth * opacity
        gt_pixs = gt_pixs.reshape(H, W, self.dataset.n_bins, 3)
        
        
        lm = correlate1d(gt_pixs[..., 0], self.dataset.laser.cpu().numpy(), axis=-1)
        exr_depth = np.argmax(lm, axis=-1)
        exr_depth = (exr_depth*2*self.model.exposure_time)/2
        mask = (gt_pixs.sum((-1, -2)) > 0) # (H,W)
        
        
        #1. Calculate the L1 and MSE error b/w rendered depth vs gt depths (gt depths can be extracted using the get_gt_depth function)
        l1_depth = self.criterions["l1_depth"](exr_depth, depth, mask)
        # MSE_depth = self.criterions["MSE_depth"](exr_depth, depth, mask)
        
        # #2. Calculate the L1 and MSE error b/w lg_depth from predicted transient and gt depth
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
        ground_truth_image = ((scaling * ground_truth_image/self.dataset.div_vals[self.config.dataset.scene]) ** (1 / 2.2)).to(torch.float64)
    
        # #4. MSE between transient images vs predicted images
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
        
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_image, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': ground_truth_image, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'depth', 'img': depth_image, 'kwargs': {"title": "Predicted Depth", "cmap": "inferno", "vmin":0.8, "vmax":1.5},},
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
            rank_zero_info("Exporting Mesh...")
            mesh_path = self.export()
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
            

    def export(self):
        mesh = self.model.export(self.config.export)
        mesh_path = self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}-{self.dataset.n_views}_{self.dataset.scene}.obj",
            **mesh
        )
        return mesh_path    
