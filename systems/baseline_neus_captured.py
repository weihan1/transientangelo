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


@systems.register('baseline-neus-captured-system')
class BaselineNeusCapturedSystem(BaseSystem):
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
        return self.model(batch['rays_dict'])
    
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
            #Multi-ray shooting 

            #Randomly select some pixels during training, these pixels are only used to index the gt transients        
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            x = x.repeat(self.train_rep) #(n_rays*rep)
            y = y.repeat(self.train_rep) #(n_rays*rep)
            index = index.repeat(self.train_rep)
            rgb = self.dataset.all_images[index, y, x]
            
            if self.config.dataset.use_gt_depth_normal:
                ground_truth_depth = self.dataset.all_depths[index, y, x]
                ground_truth_normal = self.dataset.all_normals[index, y, x]
                ground_truth_depth = torch.from_numpy(ground_truth_depth).to(self.rank).float()
                ground_truth_normal = ground_truth_normal.to(self.rank).float()
                
            rgba_sum = rgb.sum(-2)
            rgba_sum_normalized = rgba_sum/self.dataset.dataset_scale
            rgba_sum_norm_gamma = rgba_sum_normalized**(1/2.2)
            rgb = rgba_sum_norm_gamma.float()
            
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x, s_y = x, y
            
            if self.config.model.use_reg_nerf:
                #NOTE: the regnerf patch rays are always concatenated at the end
                patch_size = self.config.model.train_patch_size 
                n_sparse_rays = self.config.model.sparse_rays_size
                img_wh = self.dataset.w, self.dataset.h
                regnerf_x, regnerf_y, unseen_rotation_matrix, unseen_translation_vector = generate_unseen_poses(patch_size, img_wh, self.dataset.all_c2w, n_sparse_rays=n_sparse_rays, 
                                                                                                bounding_form =self.config.model.bounding_form, mean_focus_point = self.dataset.mean_focus_point,
                                                                                                sample_boundary = self.config.model.sample_boundary, write_poses = self.config.model.write_poses, captured=True)
                regnerf_x, regnerf_y = regnerf_x.to(self.rank), regnerf_y.to(self.rank)
                unseen_rotation_matrix = torch.from_numpy(unseen_rotation_matrix).to(self.rank)
                #Calculate the rays for the unseen patch using the new rotation matrix and translation vector
                regnerf_sx, regnerf_sy = regnerf_x, regnerf_y

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
                #NOTE: the regnerf patch rays are always concatenated at the end
                patch_size = self.config.model.train_patch_size 
                n_sparse_rays = self.config.model.sparse_rays_size
                img_wh = self.dataset.w, self.dataset.h
                regnerf_x, regnerf_y, unseen_rotation_matrix, unseen_translation_vector = generate_unseen_poses(patch_size, img_wh, self.dataset.all_c2w, n_sparse_rays=n_sparse_rays, 
                                                                                                bounding_form =self.config.model.bounding_form, mean_focus_point = self.dataset.mean_focus_point,
                                                                                                sample_boundary = self.config.model.sample_boundary, write_poses = self.config.model.write_poses, captured=True)
                regnerf_x, regnerf_y = regnerf_x.to(self.rank), regnerf_y.to(self.rank)
                unseen_rotation_matrix = torch.from_numpy(unseen_rotation_matrix).to(self.rank)
                #Calculate the rays for the unseen patch using the new rotation matrix and translation vector
                regnerf_sx, regnerf_sy = regnerf_x, regnerf_y
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
            gt_transient = rgb #Need the gt transient to extract the gt depth
            
            rgba_sum = rgb.sum(-2)
            rgba_sum_normalized = rgba_sum/self.dataset.dataset_scale
            rgba_sum_norm_gamma = rgba_sum_normalized**(1/2.2)
            rgb = rgba_sum_norm_gamma.float()
            
            self.train_rep = torch.tensor(1, dtype=torch.long)
            x = x.to(self.rank)
            y = y.to(self.rank)
            s_x, s_y = x,y
            
        #Converts pixel coordinates to normalized camera coordinates    
        c2w = self.dataset.all_c2w[index]
        camera_dirs = self.dataset.K(s_x, s_y)
        
        if self.config.model.use_reg_nerf and stage not in ["test"]:
            patch_size = self.config.model.train_patch_size if stage == "train" else 512
            regnerf_rays_d = (camera_dirs[-patch_size**2:, None, :] * unseen_rotation_matrix.to(self.rank)).sum(dim=-1)
            unseen_rays_o = unseen_translation_vector.expand_as(regnerf_rays_d).to(self.rank).float()
            unseen_rays = torch.cat([unseen_rays_o, F.normalize(regnerf_rays_d, p=2, dim=-1)], dim=-1).float()
            rays_dict.update({"regnerf_patch": {'rays': unseen_rays, 'rotation_matrix': unseen_rotation_matrix}})
            if self.trainer.global_step == 1:
                self.print("Regnerf patch rays have been added!")
                
                
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
        
        
        
        if stage in ["train"]:
            batch.update({
                'rays_dict': rays_dict,
                'rgb': rgb,
                'ground_truth_depth': ground_truth_depth if self.config.dataset.use_gt_depth_normal else None,
                'ground_truth_normals': ground_truth_normal if self.config.dataset.use_gt_depth_normal else None,
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
                'rgb': rgb,
                'laser_kernel': self.dataset.laser_kernel,
                'gt_transient': gt_transient
            })
    
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.


        if self.config.model.use_reg_nerf:
            #NOTE: This loss is used when reproducing regnerf
            if out["num_samples_regnerf"] > 0:
                #Depth smoothness regularizer
                depth_patch = out["depth_patch"].reshape(self.config.model.train_patch_size, self.config.model.train_patch_size)
                v00 = depth_patch[:-1, :-1]
                v01 = depth_patch[:-1, 1:]
                v10 = depth_patch[1:, :-1]
                smoothness_loss = (torch.sum(((v00 - v01) ** 2) + ((v00 - v10) ** 2))).item()
                loss += smoothness_loss * self.C(self.config.system.loss.lambda_depth_smoothness)

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        
        

        #NOTE: This loss is used when reproducing MonoSDF   
        #1. Depth Consistency Loss
        valid_rays = out['rays_valid_full']
        predicted_depth = out['depth'] #(n_rays, 1)
        groundtruth_depth = batch["ground_truth_depth"] #(n_rays, 1)
        comp_normal = out['comp_normal']
        
        if self.config.dataset.use_gt_depth_normal:        
            loss_depth_consistency = F.mse_loss((predicted_depth[valid_rays[...,0]])[...,0], groundtruth_depth[valid_rays[...,0]])
            loss += loss_depth_consistency * self.C(self.config.system.loss.lambda_depth_consistency)
            first_term = F.l1_loss(comp_normal[valid_rays[...,0]], batch["ground_truth_normals"][valid_rays[...,0]])
            dot_prod = (comp_normal[valid_rays[...,0]]*batch["ground_truth_normals"][valid_rays[...,0]]).sum(-1)
            second_term = torch.mean(torch.abs(1 - dot_prod))
            loss_normal_consistency = first_term + second_term
            loss += loss_normal_consistency * self.C(self.config.system.loss.lambda_normal_consistency) 
        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        
        
        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            
            #NOTE: Linear warmup for the curvature loss following the learning rate scheduler 
            #Start with 0.01 times the lambda_curvature and then linearly increase to lambda_curvature after 5000 steps
            end_scale = 1.0
            start_scale = 0.01
            end_step = (self.config.model.geometry.xyz_encoding_config.n_levels - self.config.model.geometry.xyz_encoding_config.start_level)*self.config.model.geometry.xyz_encoding_config.update_steps
            if self.global_step < self.config.system.warmup_steps:
                self.curvature_scale = min(end_scale, start_scale + (end_scale - start_scale) * self.global_step / self.config.system.warmup_steps)
                
            else: #Decreasing by 0.01 every 5000 steps until global step equals end step
                if self.global_step % 5000 == 0 and self.global_step < end_step:
                    self.curvature_scale = max(self.curvature_scale - 0.01, 0)
                
            self.log('train/curvature_scale', self.curvature_scale, prog_bar=True)
            
            
            if self.global_step == self.config.system.warmup_steps:
                self.print(f"Curvature loss warmup finished at step {self.global_step}")
                self.print(f"starting scaling it down")
                
            loss_curvature = loss_curvature * self.curvature_scale
            
            #NOTE: after reaching the end of the warmup, we need to scale the loss back down
            
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

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
        W, H = self.dataset.img_wh
        np.save(self.get_save_path(f"it{self.global_step}-{batch['index'][0].item()}_normal"), out['comp_normal'].view(H, W,3).cpu().numpy())
        
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
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
        
        W, H = self.dataset.img_wh
        meta_data = self.dataset.meta
        out = self(batch)
        

        # #Initialize all gt tensors
        gt_pixs = batch["rgb"].view(H, W, 3).detach().cpu().numpy() #NOTE: this is the integrated image not the gt transient
        gt_transient = batch["gt_transient"].view(H,W,self.dataset.n_bins,3)
        
        
        lm = correlate1d(gt_transient[..., 0], self.dataset.laser.cpu().numpy(), axis=-1)
        exr_depth = np.argmax(lm, axis=-1)
        exr_depth = (exr_depth*2*299792458*4e-12)/2
        # #Get predicted and ground_truth depth 
        # #Initialize all arrays as np arrays to prevent needing to convert tensor -> np arrays and populate
        rgb = out["comp_rgb_full"].numpy().reshape(H,W,3)
        depth = out["depth"].numpy().reshape(H,W)
        opacity = out["opacity"].numpy().reshape(H,W)
        
        depth_image = depth
        
        mask = (gt_pixs.sum(-1) > 0) # (H,W
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
        iou = torch.tensor(self.criterions["Transient_IOU"](rgb, gt_pixs))
        
        rgb_image = rgb
        data_image = gt_pixs
        
    
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
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth.png"), exr_depth, cmap='inferno', vmin=0.8, vmax=1.5)
        plt.imsave(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_viz.png"), depth_image, cmap='inferno', vmin=0.8, vmax=1.5)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_depth_array"), depth)
        np.save(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_transient_array"), rgb)
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_predicted_RGB.png"), (rgb_image*255.0).astype(np.uint8))
        imageio.imwrite(self.get_save_path(f"it{self.global_step}-test/{batch['index'][0].item()}_gt_RGB.png"), (data_image*255.0).astype(np.uint8))
        
        
        self.save_image_plot_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': rgb_image, 'kwargs': {"title": "Predicted Integrated Transient"}},
            {'type': 'rgb', 'img': data_image, 'kwargs': {"title": "Ground Truth Integrated Transient"}},
            {'type': 'depth', 'img': depth_image, 'kwargs': {"title": "Predicted Depth", "cmap": "inferno", "vmin":0.8, "vmax":1.5},},
            {'type': 'depth', 'img': exr_depth, 'kwargs': {"title": "Ground Truth Depth", "cmap": "inferno", "vmin":0.8, "vmax":1.5},},
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
            
            metrics_dict = {"L1 Depth": l1_depth.item(), "PSNR": psnr.item(), "SSIM": ssim.item(), "LPIPS": lpips.item(), "IOU": iou.item()}
            
            self.save_metrics(f"it{self.global_step}-test.txt", metrics_dict)
            
            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=10
            )
            
            rank_zero_info("Exporting Mesh...")
            # self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        mesh_path = self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )
        return mesh_path        
