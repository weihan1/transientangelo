import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import imageio
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays, spatial_filter, generate_unseen_poses, find_surface_points
from models.mesh_utils import calculate_chamfer_distance
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy, MSELoss, L1_depth, MSE_depth, get_gt_depth, get_depth_from_transient, SSIM, LPIPS, KL_DIV, TransientIOU


@systems.register('transient-neus-system')
class TransientNeuSSystem(BaseSystem):
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
        '''
        Forward pass the rays dictionary
        '''
        return self.model(batch['rays_dict'])
    
    def preprocess_data(self, batch, stage):
        #This populates the batch to feed into forward
        
        #a rays dict with their own extrinsic params (location + rotation matrix). Only need to store the extrinsic params for all rays
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
            #Multi-ray shooting 
            if self.config.model.multi_ray_shooting:
                if self.trainer.global_step%1000==0:
                    if self.train_rep < 30:
                        self.train_rep += 2
                        
            #Randomly select some pixels during training, these pixels are only used to index the gt transients        
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
            
            
            #NOTE: Use this when we need to visualize a patch of rays of the image
            if self.config.model.use_patch_training:
                training_patch_size = 64
                start_y = torch.randint(0, self.dataset.h - training_patch_size, size=(1,)).item()
                start_x = torch.randint(0, self.dataset.w - training_patch_size, size=(1,)).item()
                rows = torch.arange(start_y, start_y + training_patch_size, device=self.rank)
                cols = torch.arange(start_x, start_x+ training_patch_size, device=self.rank)
                train_patch_x, train_patch_y = torch.meshgrid(rows, cols, indexing="xy")
                train_patch_x, train_patch_y = train_patch_x.flatten(), train_patch_y.flatten()
                train_patch_sx, train_patch_sy, _ = spatial_filter(train_patch_x, train_patch_y, sigma=self.dataset.rfilter_sigma , rep = 1, prob_dithering=self.dataset.sample_as_per_distribution)
                train_patch_sx = torch.clip(train_patch_x + torch.from_numpy(train_patch_sx).to(self.rank), 0, self.dataset.w) 
                train_patch_sy = torch.clip(train_patch_y + torch.from_numpy(train_patch_sy).to(self.rank), 0, self.dataset.h)
                s_x, s_y = torch.cat([s_x, train_patch_sx], dim=0), torch.cat([s_y, train_patch_sy], dim=0)
                
                
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
            x, y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            x = x.flatten()
            y = y.flatten()
            #NOTE: because there's no spatial filter, no need to set the self.train_rep = 1
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x = x
            s_y = y
            if self.config.model.use_reg_nerf:
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
            #NOTE: self.train_rep=1 during testing
            x, y = torch.meshgrid(
                torch.arange(self.dataset.w, device=self.dataset.all_c2w.device),
                torch.arange(self.dataset.h, device=self.dataset.all_c2w.device),
                indexing="xy",
            ) #(self.dataset.w, self.dataset.h)
            x = x.flatten()
            y = y.flatten()
            rgb = self.dataset.all_images[index, y, x]
            
            self.train_rep = torch.tensor(1, dtype=torch.long)
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x, s_y, weights = spatial_filter(x, y, sigma=self.dataset.rfilter_sigma, rep = self.train_rep, prob_dithering=self.dataset.sample_as_per_distribution, normalize=False)
            s_x = (torch.clip(x + torch.from_numpy(s_x).to(self.rank), 0, self.dataset.w).to(torch.float32))
            s_y = (torch.clip(y + torch.from_numpy(s_y).to(self.rank), 0, self.dataset.h).to(torch.float32))
            weights = torch.Tensor(weights).to(self.rank)
            
        c2w = self.dataset.all_c2w[index]
        
        #Applying K^{-1}, mapping image coordinates to camera coordinates
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
        
        
        if stage in ["train"]:  
            
            rays_d = (camera_dirs[:self.train_num_rays*self.train_rep, None, :] * c2w[:, :3, :3].to(self.rank)).sum(dim=-1) #upper left corner of the c2w
            if self.config.model.use_patch_training:
                #NOTE: The patch rays sampled during training used to perform warping
                random_number = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
                random_camera_index = index[random_number]
                sampled_c2w = self.dataset.all_c2w[random_camera_index]
                patch_rays_d = (camera_dirs[self.train_num_rays*self.train_rep:self.train_num_rays*self.train_rep+self.config.model.train_patch_size**2, None, :] * sampled_c2w[:,:3, :3].to(self.rank)).sum(dim=-1) #upper left corner of the c2w
                patch_rays_o = torch.broadcast_to(sampled_c2w[:,:3, -1], patch_rays_d.shape).to(self.rank) #Camera origins from the same view are the same
                #NOTE: Need to resample a c2w matrix for the new patch trainig rays 
                training_patch_rays = torch.cat([patch_rays_o, F.normalize(patch_rays_d, p=2, dim=-1)], dim=-1).float() #(self.patch_size**2, 6)
                rays_dict.update({"training_patch_rays": {"rays": training_patch_rays, 'rotation_matrix': sampled_c2w[:, :3, :3], 'index': index}}) 
                if self.trainer.global_step == 1:
                    self.print("Training patch rays has been added!")
                    
            if self.config.model.use_reg_nerf:
                #NOTE: the regnerf patch rays are always concatenated at the end, used for rawnerf
                regnerf_rays_d = (camera_dirs[-self.config.model.train_patch_size**2:, None, :] * unseen_rotation_matrix.to(self.rank)).sum(dim=-1) #upper left corner of the c2w
                unseen_rays_o = torch.broadcast_to(unseen_translation_vector, regnerf_rays_d.shape).to(self.rank).float()
                unseen_rays = torch.cat([unseen_rays_o, F.normalize(regnerf_rays_d, p=2, dim=-1)], dim=-1).float() #(self.patch_size**2 + sparse_rays, 6)
                rays_dict.update({"regnerf_patch": {'rays': unseen_rays, 'rotation_matrix': unseen_rotation_matrix}})
                if self.trainer.global_step == 1:
                    self.print("Regnerf patch rays has been added!")
            
            #Add more rays if needed
            
            
        if stage in ["validation", "test"]:
            rays_d = (camera_dirs[:self.dataset.w*self.dataset.h, None, :] * c2w[:, :3, :3].to(self.rank)).sum(dim=-1) 
            
        rays_o = torch.broadcast_to(c2w[:,:3, -1], rays_d.shape).to(self.rank) #Camera origins from the same view are the same 
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1).float() #Contains all the rays
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
        
        if stage in ["train"]:
            batch.update({
                'rays_dict': rays_dict,
                'rgb': rgb.view(-1, self.dataset.n_bins*3), # (n_rays*rep, 1200*3)
                'weights': weights
            })
            
        elif stage in ["validation"]:
            batch.update({
                'rays_dict': rays_dict
            })
        
        elif stage in ["test"]:
            batch.update({
                'rays_dict': rays_dict,
                'rgb': rgb.view(-1, self.dataset.n_bins*3),
                'weights': weights
            })
    
    def training_step(self, batch, batch_idx):
        
        out = self(batch)
        
        training_rays = batch["rays_dict"]["training_rays"]["rays"]
        loss = 0.
        
        if self.config.model.multi_ray_shooting:
            self.log('train/rep', self.train_rep)
            
        self.log('train/number_of_samples', out["num_samples"].float())
        
        #TODO: Do a quick visualization of the warping here
        if self.config.model.use_patch_training: #assumes regnerf rays exist
            #1. get the pixel locations from the regnerf rays and denote them as p_j = batch["rays_dict"]["regnerf_patch"]["rays"]
            #2. multiply it by K^-1 (which maps pixel space to camera space of unseen viewpoints)
            #3. multiply the result by either 1) out["depth"] 
            #4. multiply it by the rotation matrix of the unseen viewpoint (it's now in world space)
            #5. multiply it by the inverse of the rotation matrix of the seen viewpoint
            #6. multiply it by K (which maps camera space to pixel space of seen viewpoint)
            regnerf_rays = batch["rays_dict"]["regnerf_patch"]["rays"]
                    
        if self.config.model.use_reg_nerf:     
            if out["num_samples_regnerf"] > 0:
                if self.config.model.train_patch_size>0:
                    #NOTE: the following tensors are always used if patch regnerf is used 
                    color_patch = out["color_patch"]
                    depth_patch = out["depth_patch"].reshape(self.config.model.trian_patch_size, self.config.model.train_patch_size)
                    transient_patch = out["transient_patch"]
                    depth_variance_patch = out["depth_variance_patch"]
                    sdf_grad_samples = out["sdf_grad_samples"]
                    
                    
                    # #NOTE: Assumption that we only use 8x8 patch
                    # assert (color_patch == out["color_patch"][:64]).all()
                    # assert (depth_patch == out["depth_patch"][:64]).all()
                    # assert (transient_patch == out["transient_patch"][:64]).all()
                    # assert (depth_variance_patch == out["depth_variance_patch"][:64]).all()
                    
                    ##NOTE: Depth smoothness regularizer
                    v00 = depth_patch[:-1, :-1]
                    v01 = depth_patch[:-1, 1:]
                    v10 = depth_patch[1:, :-1]
                    smoothness_loss = (torch.sum(((v00 - v01) ** 2) + ((v00 - v10) ** 2))).item()
                    loss += smoothness_loss * self.C(self.config.system.loss.lambda_depth_smoothness)
                    
                    
                    #NOTE: Transient noise regularizer
                    transient_noise_loss = transient_patch[transient_patch<1e-7].sum()
                    loss += transient_noise_loss * self.C(self.config.system.loss.lambda_transient_noise) 
                    
                    #NOTE: Argmax consistency regularizer
                    peaks = torch.argmax(transient_patch, dim=1) #(patch_size**2 , 3) 
                    peaks_size = int(math.sqrt(peaks.shape[0]))
                    averaged_peaks = torch.mean(peaks.float(), dim=-1).view(peaks_size, peaks_size) #averaged across the channel dimension
                    peak_differences_row = averaged_peaks[:, 1:] - averaged_peaks[:, :-1] #(patch_size, patch_size-1)
                    peak_differences_col = averaged_peaks[1:, :] - averaged_peaks[:-1, :]
                    peak_smoothness_loss_row = torch.sum((peak_differences_row)**2)
                    peak_smoothness_loss_col = torch.sum((peak_differences_col)**2)
                    peak_smoothness_loss = peak_smoothness_loss_row + peak_smoothness_loss_col
                    loss += peak_smoothness_loss * self.C(self.config.system.loss.lambda_argmax_consistency)
                
                    #NOTE: Color regularization loss
                    integrated_transient = color_patch.sum(-2)
                    integrated_transient = (integrated_transient / integrated_transient.max()) ** (1 / 2.2)
                    #TODO: Add the pretrained normalizing flow model here:
                    #loss += NLL(Flow(color_patch)) * self.C(self.config...)
                    
                    
                    #NOTE: Depth variance regularizer
                    # if self.global_step % 5000 == 0 and self.global_step and self.config.model.use_depth_variance_weight:
                    #     self.regnerf_patch_depth_variance_weight =  min(1, self.regnerf_patch_depth_variance_weight+0.1)
                    #     self.print(f"regnerf patch depth variance weight has been updated to {self.regnerf_patch_depth_variance_weight}")
                    loss += depth_variance_patch.mean() * self.C(self.config.system.loss.lambda_regnerf_depth_variance)
                    
                    #NOTE: Eikonal loss for regnerf
                    eikonal_loss_patch = ((torch.linalg.norm(sdf_grad_samples, ord=2, dim=-1) - 1.)**2).mean()
                    loss += eikonal_loss_patch * self.C(self.config.system.loss.lambda_eikonal_patch)
                    
                    #NOTE: Distortion Loss
                    if self.C(self.config.system.loss.lambda_distortion_patch) > 0:
                        loss_distortion = flatten_eff_distloss(out['weights_patch'], out['points_patch'], out['intervals_patch'], out['ray_indices_patch'])
                        loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion_patch)
                    
                if self.config.model.sparse_rays_size > 0:
                    depth_variance_sparse = out["depth_variance_patch"][-self.config.model.sparse_rays_size:] #always select bottom 64 elements
                    loss += depth_variance_sparse.mean() * self.C(self.config.system.loss.lambda_regnerf_sparse_depth_variance)       
        
        gt_pixs = torch.reshape(batch["rgb"][:int(batch["rgb"].shape[0]/self.train_rep)], (-1, self.dataset.n_bins, 3)) #Shape (n_rays,1200, 3)
    
        #Sums across the train_rep dimension 
        alive_ray_mask = out["rays_valid"].squeeze(-1) #(n_rays, 1)
        alive_ray_mask = alive_ray_mask.reshape(self.train_rep, -1)
        alive_ray_mask = alive_ray_mask.sum(0).bool() #shape of (num_rays,), #binary mask
        self.log('train/alive_ray_mask', float(alive_ray_mask.sum().item()))
        
        
        predicted_rgb = torch.reshape(out["rgb"], (-1, self.dataset.n_bins, 3))*batch["weights"][:, None, None] #(512, 1200, 3)
        
        #Basically penalizes the location in comp_weights where the corresponding locations in gt_pixs have lower than background intensity
        comp_weights = out["comp_weights"][gt_pixs.sum(-1).repeat(self.train_rep, 1)<1e-7].mean() #space carving loss (1,)
        
        rgb = torch.zeros((int(predicted_rgb.shape[0]/self.train_rep), self.dataset.n_bins, 3)).to(self.rank)
        
        #index[i] is of shape (1200,3) denoting the index of the ith ray
        index = torch.arange(int(predicted_rgb.shape[0]/self.train_rep)).repeat(self.train_rep)[:, None, None].expand(-1, self.dataset.n_bins, 3).to(self.rank) #(512,1200,3)
        rgb.scatter_add_(0, index.type(torch.int64), predicted_rgb)
        
        #NOTE: This is a sanity check when self.train_rep = 1
        # if (rgb != predicted_rgb).any() and self.train_rep==1:
        #     self.print("gt and predicted_rgb are not the same")
        #     self.print(f"gt is {rgb}")
        #     self.print(f"predicted is {predicted_rgb}")
        #     exit(1)
        
        if self.config.model.learned_background:
            # https://arxiv.org/pdf/2303.12280.pdf
            scaling_factor = (torch.mean(gt_pixs.sum(-2), dim = (0,-1)) - torch.mean(rgb.sum(-2), dim= (0,-1)))/torch.mean(out["rgb_bg"].sum(-2), dim= (0,-1))
            scaling_factor = 0.1
            rgb = rgb + scaling_factor * out["rgb_bg"]
            
        gt_pixs = torch.log(gt_pixs + 1) #(num_ray,n_bins,RGB)
        predicted_rgb = torch.log(rgb + 1) # (num_rays, n_bins, RGB)
        
        
        #Plot the predicted and gt images on top of each other
        if not self.global_step % 1000:
            self.save_plot(f"it{self.global_step}-transient_plot", torch.mean(predicted_rgb[alive_ray_mask], dim=(0,2)), torch.mean((gt_pixs[alive_ray_mask]), dim=(0,2)))
            self.save_depth_plot(f"it{self.global_step}-transient_depth_plot", predicted_rgb[alive_ray_mask], gt_pixs[alive_ray_mask], out["depth"][:out["depth"].shape[0]//self.train_rep][alive_ray_mask], self.model.exposure_time)
            #only take the ray indices of the first samples
            self.save_sdf_plot(f"it{self.global_step}-sdf_depth_plot", out["sdf_samples"], self.model.exposure_time, out["distances_from_origin"],out["depth"][:out["depth"].shape[0]//self.train_rep][alive_ray_mask], out["ray_indices"], alive_ray_mask)
            self.save_weight_plot(f"it{self.global_step}-weight_plot", out["weights"], self.model.exposure_time, out["distances_from_origin"],out["depth"][:out["depth"].shape[0]//self.train_rep], out["ray_indices"], alive_ray_mask)
            self.save_transient_figures(f"it{self.global_step}-transient_figures", predicted_rgb, gt_pixs)
            self.save_color_figures(f"it{self.global_step}-color_figures", gt_pixs.cpu().detach(),predicted_rgb.cpu().detach())
            
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / (out['num_samples_full'].sum().item() + 1e-10)))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        #Add a loss that penalizes only the intensity of the argmax for each ray and averaged across all rays and color channels
        max_gt = torch.max(gt_pixs[alive_ray_mask], dim=1)[0]
        max_predicted = torch.max(predicted_rgb[alive_ray_mask], dim=1)[0]
        loss_max = F.l1_loss(max_predicted, max_gt, reduction="mean")
        loss += loss_max * self.C(self.config.system.loss.lambda_max_l1)
        self.log('train/loss_max', loss_max)
        
        
        #integrated transient loss
        #Normalize + Gamma correct
        integrated_gt = gt_pixs[alive_ray_mask].sum(-2)
        # integrated_gt /= integrated_gt.max()
        # integrated_gt = integrated_gt**(1/2.2)
        
        integrated_pred = predicted_rgb[alive_ray_mask].sum(-2)
        # integrated_pred /= integrated_pred.max()
        # integrated_pred = integrated_pred**(1/2.2)
        
        loss_integrated = F.l1_loss(integrated_pred, integrated_gt, reduction="mean")
        loss += loss_integrated * self.C(self.config.system.loss.lambda_integrated_l1)
        
        loss_rgb = F.l1_loss(predicted_rgb[alive_ray_mask], gt_pixs[alive_ray_mask], reduction="mean")
        self.log('train/loss_rgb', loss_rgb, prog_bar=True)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb_l1)
        
        #space carving loss proposed in https://arxiv.org/pdf/2307.09555.pdf
        space_carving_loss = comp_weights
        self.log('train/space_carving_loss', space_carving_loss)
        loss += space_carving_loss * self.C(self.config.system.loss.lambda_space_carving) 
        
        # background_transient = min(background_level - predicted_rgb.mean(), 0)
        # average_predicted_rgb = predicted_rgb.mean()
        # background_level = 1e-6
        # background_regularization = torch.relu(background_level - average_predicted_rgb)
        # loss += background_regularization * self.C(self.config.system.loss.lambda_background_regularization)
        #NOTE: Interesting idea but keep commented for now
        # if self.global_step % 3000 == 0 and self.global_step:
        #     self.annealing_factor *= 5
        #     self.annealing_factor = min(self.annealing_factor, 1e2)
        #     self.print(f"Annealing factor has been updated to {self.annealing_factor}")
        #     self.print(f"Eikonal beta has been updated to {self.annealing_factor * self.C(self.config.system.loss.lambda_eikonal)}")
            
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('loss_eikonal', loss_eikonal, prog_bar=True)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        #InfoNeRF (https://arxiv.org/pdf/2112.15399.pdf)
        # num_rays = training_rays.shape[0]
        # alphas = out["alpha_samples"]
        # sums = torch.zeros(num_rays, dtype=torch.float32, device=alphas.device).scatter_add_(0, out["ray_indices"], alphas.squeeze())
        # ray_density =  alphas.squeeze() / sums[out["ray_indices"]]
        # log_ray_density = torch.log(ray_density + 1e-7)
        # summed_quantity = ray_density * log_ray_density
        # shannon_entropy = torch.zeros(num_rays, dtype=summed_quantity.dtype, device=summed_quantity.device).scatter_add_(0, out["ray_indices"], -summed_quantity)
        # # #implement a mask that checks if the cumulative opacities is above a threshold \epsilon

        # loss_ray_entropy = shannon_entropy.mean()
        # self.log('loss_ray_entropy', loss_ray_entropy)
        # loss += loss_ray_entropy * self.C(self.config.system.loss.lambda_ray_entropy)
        
        
        #ASDF Loss (https://www.krafton.ai/en/wp-content/uploads/sites/4/2024/03/3DV_2024_S2F2NeRF_camera_ready-1.pdf)
        
        # geometric smoothing loss - my own version
        
        # a) d(t) = depth - distances from origin
        # noisy_gt_depth = torch.mean(gt_pixs, dim=-1) #average over color channels
        # noisy_depth = torch.argmax(noisy_gt_depth, dim=-1)
        # noisy_depth = noisy_depth*self.model.exposure_time/2 #convert back to spatial domain
        # dist_to_surf = out["depth"][out["ray_indices"]].squeeze() - out["distances_from_origin"]
        # # c) b annealing bound
        # i_opt = 5000
        # b_max = 0
        # b_opt = 3
        # if self.global_step <= i_opt:
        #     b = ((i_opt - self.global_step)/i_opt)*(b_max-b_opt) + b_opt
        # else:
        #     b = b_opt
        
        # # b) |sdf - d(t)| for b>d(t)
        # mask = b > dist_to_surf
        # loss_geometric_smoothing = (out["sdf_samples"]-dist_to_surf).abs()[mask].mean()
        # loss += loss_geometric_smoothing * self.C(self.config.system.loss.lambda_geometric_smoothing)
        
        
        #2. 
        #TODO: Implement MonoSDF loss (https://arxiv.org/pdf/2206.00665.pdf)
        
        
        #NOTE: Incoporate a sdf distance loss, where, along each visible ray, we penalize sdf points whoses distances from the origin are < predicted depth but have negative sdf
        # and whose distances from the origin are > predicted depth but have positive sdf
        
        #1. First the ground truth transients of the alive rays and compute its argmax
        # noisy_depth = torch.argmax(torch.mean(gt_pixs, dim=-1), dim=-1)
        
        # #2. Find indices of the rays that are alive by transforming the alive_ray_mask to a list of indices      
        # ray_indices = out["ray_indices"]
        # alive_ray_indices = alive_ray_mask.nonzero().squeeze(-1)
        # alive_ray_indices = ray_indices[torch.isin(ray_indices, alive_ray_indices)]
        
        # #3. Get the corresponding sdf and distances from the origin
        # sdf_samples = out["sdf_samples"]
        # distances_from_origin = out["distances_from_origin"]
        # distances_from_origin = distances_from_origin*2/self.model.exposure_time
        # #4 For each visible_ray, find the number of samples that violate the sdf rule 
        
        # #Map the gt depth to have the same shape as the shape of the samples
        # noisy_depth = noisy_depth[alive_ray_indices]
        
        # #Find all samples whose distances from the origin are less than the predicted depth but have negative sdf
        # negative_sdf_mask = sdf_samples < 0
        # closer_than_gt_mask = distances_from_origin < noisy_depth
        
        # #Find all samples whose distances from the origin are greater than the predicted depth but have positive sdf
        # positive_sdf_mask = sdf_samples > 0
        # further_than_gt_mask = distances_from_origin > noisy_depth
        
        # penalty_mask = (negative_sdf_mask & closer_than_gt_mask) | (positive_sdf_mask & further_than_gt_mask)
                
        # # sdf_distance_loss: penalize the absolute
        # sdf_distance_loss = torch.mean(torch.abs(sdf_samples[penalty_mask]))
        # loss += sdf_distance_loss * self.C(self.config.system.loss.lambda_sdf_loss)
        
        # #Penalizing non-zero sdf loss on the surface
        # non_zero_sdf_mask = sdf_samples.abs() > 1e-7
        # at_the_surface = torch.abs(distances_from_origin - noisy_depth) < 1e-7
        # surface_sdf_loss = torch.mean(torch.abs(sdf_samples[non_zero_sdf_mask & at_the_surface]))
        # loss += surface_sdf_loss * self.C(self.config.system.loss.lambda_surface_sdf_loss)

        
        #normal loss proposed in https://nv-tlabs.github.io/adaptive-shells-website/assets/adaptiveShells_paper.pdf
        #NOTE: only use this when the network is predicting the normals, otherwise loss is 0
        loss_normal = torch.linalg.norm(out["normal"] - F.normalize(out["sdf_grad_samples"], p=2, dim=-1), ord=2, dim=-1).mean()
        self.log('loss_normal', loss_normal, prog_bar=False)
        loss += loss_normal * self.C(self.config.system.loss.lambda_normal)
        
        
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        
        if "fg_mask" in batch:
            loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
            self.log('train/loss_mask', loss_mask)
            loss += loss_mask * self.C(self.config.system.loss.lambda_mask)

            
        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)


        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        
        loss_variance = out["depth_variance"].mean()
        loss += loss_variance * self.C(self.config.system.loss.lambda_depth_variance)
        
        loss_total_variation = torch.sum(torch.norm(out["sdf_grad_samples"], dim=-1, p=2))
        
        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        # if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
        #     loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
        #     self.log('train/loss_distortion_bg', loss_distortion_bg)
        #     loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        #NOTE: inv_s should be steadily increasing to > 100
        self.log('train/inv_s', out['inv_s'], prog_bar=True) 


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
        

        integrated_transient = out["rgb"].sum(-2)
        W, H = math.sqrt(integrated_transient.shape[0]), math.sqrt(integrated_transient.shape[0])

        if W.is_integer() and H.is_integer():
            W,H = int(W), int(H)
        else:
            self.print("W or H is not an integer")
            return True
        
        integrated_transient = (integrated_transient / integrated_transient.max()) ** (1 / 2.2)
        
        
        #Plotting patch rays and transients
        
        #1. Find roughly the midpoints of the patch and plot a 3x3 grid of points
        point_x_patch, point_y_patch = np.meshgrid(np.arange(W//2, W//2 + 3),np.arange(H//2, H//2 + 3),indexing="xy")
        point_x_patch, point_y_patch = point_x_patch.flatten(), point_y_patch.flatten()
        self.save_plot_grid(f"it{self.global_step}-{batch['index'][0].item()}-regnerf_patch_transient", point_x_patch, point_y_patch, out["rgb"].view(W,H,self.model.n_bins,3),self.global_step)
        
        
        #Sample 9 points, 3 from the top, 3 from the middle and 3 from the bottom
        point_x = np.array([W//4,W//2,3*W//4,W//4,W//2,3*W//4,W//4,W//2,3*W//4])
        point_y = np.array([H//4,H//4,H//4,H//2,H//2,H//2,3*H//4,3*H//4,3*H//4])    
        self.save_plot_grid(f"it{self.global_step}-{batch['index'][0].item()}-regnerf_transient.png", point_x, point_y, out["rgb"].view(W,H,self.model.n_bins,3), self.global_step)

        # PLotting 
        # self.save_weight_grid(f"it{self.global_step}-{batch['index'][0].item()}-regnerf_patch_weight", point_x, point_y, out["rgb"].view(W,H,self.model.n_bins,3), out["weights"], out["ray_indices"], self.global_step)


        # #1. Choose three points on the image near the center of the image and plot the image with the dots
        # quotient = 2
        # rendering_ind_x = np.array([256]*(W // 2 ))
        # rendering_ind_y = np.arange(0, W, quotient) #pixel indices from 0 to 512 in increment of 2
        # plotting_subset_y = np.array([50, 100, 150])
        # img_plotting_subset_y = plotting_subset_y*2 #([100,200, 300])
        # exposure_time = self.model.exposure_time
        
        
        # #NOTE: ray_indices is bounded between (0, self.config.model.ray_chunk), therefore, it's non-trivial to determine the indices of the rays
        # #that correspond to the exact image pixel
        # ray_indices_count_sorted = torch.argsort(torch.bincount(out["ray_indices"]))
        # top_3_indices = ray_indices_count_sorted[-3:]
        
        # self.save_image_n_dots(f"it{self.global_step}-image_dots", predicted_image.view(H,W,3), rendering_ind_x[plotting_subset_y], img_plotting_subset_y)
        
        # self.save_sdf_plots(f"it{self.global_step}-sdf_samples", top_3_indices, out["distances_from_origin"], out["ray_indices"], exposure_time, out["depth"], out["sdf_samples"])
        
        #TODO: Plot the Distance Transform of the integrated transient or some sort of contour plot
        
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': integrated_transient.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': (out['depth']*out['opacity']).view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
            {'type': 'rgb', 'img': out["surface_normal"].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ])
        # psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        return True

    
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
        #This computes metrics for each image individually, the next method aggregates all
        W, H = self.dataset.img_wh
        meta_data = self.dataset.meta
        

        # #Initialize all gt tensors
        gt_pixs = batch["rgb"].view(H, W, self.dataset.n_bins,3).detach().cpu().numpy()
        
        
        # #Get predicted and ground_truth depth 
        exr_depth = get_gt_depth(meta_data["frames"][batch_idx], self.dataset.all_c2w[batch_idx].cpu().numpy(), self.dataset.depth_path) #(512,512)
        if self.config.dataset.downsample: #downsampling the gt depth to match that of the predicted depth
            exr_depth = (exr_depth[1::2, ::2] + exr_depth[::2, ::2] + exr_depth[::2, 1::2] + exr_depth[1::2, 1::2])/4
            
        weights_sum = 0
        rgb = np.zeros((H, W, self.dataset.n_bins, 3))
        depth = np.zeros((H, W))
        opacity = np.zeros((H, W))
        
        rep_number=30
        for j in range(rep_number):
            self.print(f"Rep number {j}")
            sample_weights = batch["weights"]
            out = self(batch)
            depth += ((torch.squeeze(out["depth"])*sample_weights.detach().cpu()).reshape(W, H)).detach().cpu().numpy()
            rgb += (out["rgb"] * sample_weights[:, None][:, None].detach().cpu()).reshape(H, W, self.dataset.n_bins, 3).detach().cpu().numpy()        
            opacity += (torch.squeeze(out["opacity"]) *sample_weights.detach().cpu().numpy()).reshape(H,W).detach().cpu().numpy()
            weights_sum += sample_weights.detach().cpu().numpy()
            del out
            
        rgb = rgb/weights_sum.reshape(H, W, 1,1)
        depth = depth/weights_sum.reshape(H,W)
        opacity = opacity/weights_sum.reshape(H,W)
        depth_image = depth*opacity        
        
        
        mask = (gt_pixs.sum((-1, -2)) > 0) # (H,W)
        # #1. Calculate the L1 and MSE error b/w rendered depth vs gt depths (gt depths can be extracted using the get_gt_depth function)
        l1_depth = self.criterions["l1_depth"](exr_depth, depth, mask)
        # # MSE_depth = self.criterions["MSE_depth"](exr_depth, depth, mask)
        
        # #2. Calculate the L1 and MSE error b/w lg_depth from predicted transient and gt depth
        # # lg_depth = get_depth_from_transient(rgb, exposure_time=0.01, tfilter_sigma=3)
        # # l1_depth_gt_lm_ours = self.criterions["l1_depth"](exr_depth, lg_depth, mask)
        # # MSE_depth_gt_lm_ours = self.criterions["MSE_depth"](exr_depth, lg_depth, mask)
        
        # #3. Calculate the L1 and MSE error b/w lg_depth from gt transient and predicted depth
        # # gt_depth = get_depth_from_transient(gt_pixs, exposure_time=0.01, tfilter_sigma=3)
        # # l1_depth_gt_lm_gt = self.criterions["l1_depth"](exr_depth, gt_depth, mask)
        # # MSE_depth_gt_lm_gt = self.criterions["MSE_depth"](exr_depth, gt_depth, mask)
        
        #4. Transient IOU
        iou = self.criterions["Transient_IOU"](rgb, gt_pixs)
        
        # #4. First get the transient images, sum across all transients (dim=-2) and gamma correct (gt and predicted)
        rgb_image = rgb.sum(axis=-2)*self.dataset.view_scale/self.dataset.dataset_scale
        ground_truth_image = gt_pixs.sum(axis=-2)*np.array(self.dataset.max)/self.dataset.dataset_scale
        rgb_image = np.clip(rgb_image,0,1) ** (1 / 2.2)
        data_image = (ground_truth_image) ** (1 / 2.2)
    
        # #5. MSE between transient images vs predicted images
        # # mse = self.criterions["MSE"](torch.from_numpy(data_image), torch.from_numpy(rgb_image))
        
        # #6. PSNR between transient images vs predicted images
        psnr = self.criterions["psnr"](torch.from_numpy(data_image), torch.from_numpy(rgb_image))
        
        # #7. SSIM between transient images vs predicted images
        ssim = torch.tensor(self.criterions["SSIM"](data_image, rgb_image), dtype=torch.float64)
        
        # #8. LPIPS between transient images vs predicted images
        lpips = torch.tensor(self.criterions["LPIPS"](data_image, rgb_image), dtype=torch.float64)
        
        
        # #9. KL divergence between transient images normalized vs predicted images normalized
        # # kl_div = self.criterions["kl_div"](rgb, gt_pixs, mask)
        
        #Preprocess the exr_depth to have a black background
        exr_depth[exr_depth == exr_depth[0][0]] = 0
        
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth.png"), exr_depth, cmap='inferno', vmin=2.5, vmax=5.5)
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_viz.png"), depth_image, cmap='inferno', vmin=2.2, vmax=5.5)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_array"), depth)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_transient_array"), rgb)
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_predicted_RGB.png"), (rgb_image*255.0).astype(np.uint8))
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_gt_RGB.png"), (data_image*255.0).astype(np.uint8))
        
        self.save_image_plot_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_image, 'kwargs': {"title": "Predicted Integrated Transient"}},
            {'type': 'rgb', 'img': data_image, 'kwargs': {"title": "Ground Truth Integrated Transient"}},
            {'type': 'depth', 'img': depth_image, 'kwargs': {"title": "Predicted Depth", "cmap": "inferno", "vmin":2.5, "vmax":5.5},},
            {'type': 'depth', 'img': exr_depth, 'kwargs': {"title": "Ground Truth Depth", "cmap": "inferno", "vmin":2.5, "vmax":5.5},},
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
            rank_zero_info("Exporting Mesh...")
            mesh_path = self.export()
            rank_zero_info("Calculating Chamfer Distance...")
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
                        
            rank_zero_info("Aggregating Metrics...")           
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
            # self.log('test/MSE_depth', MSE_depth, prog_bar=True, rank_zero_only=True)    
            # self.log('test/l1_depth_gt_lm_ours', l1_depth_gt_lm_ours, prog_bar=True, rank_zero_only=True)    
            # self.log('test/MSE_depth_gt_lm_ours', MSE_depth_gt_lm_ours, prog_bar=True, rank_zero_only=True)    
            # self.log('test/l1_depth_gt_lm_gt', l1_depth_gt_lm_gt, prog_bar=True, rank_zero_only=True)    
            # self.log('test/MSE_depth_gt_lm_gt', MSE_depth_gt_lm_gt, prog_bar=True, rank_zero_only=True)    
            # self.log('test/mse', mse, prog_bar=True, rank_zero_only=True)    
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True, sync_dist=True) 
            self.log('test/ssim', ssim, prog_bar=True, rank_zero_only=True, sync_dist=True)
            self.log('test/lpips', lpips, prog_bar=True, rank_zero_only=True, sync_dist=True)
            self.log('test/iou', iou, prog_bar=True, rank_zero_only=True, sync_dist=True)
            
            metrics_dict = {"Chamfer Distance": chamfer_distance, "L1 Depth": l1_depth.item(), "PSNR": psnr.item(), "SSIM": ssim.item(), "LPIPS": lpips.item(), "IOU": iou.item()}
            
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
