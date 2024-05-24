import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

import models
from models.base import BaseModel
from models.utils import chunk_batch, mapping_dist_to_bin_mitsuba, mapping_dist_to_bin, torch_laser_kernel, convolve_colour
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, accumulate_along_rays, render_weight_from_alpha

import numpy as np
import scipy.io as sio

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('captured-neus')
class CapturedNeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        self.register_buffer("cos_anneal_ratio",torch.tensor(1.0).to(self.scene_aabb))
        
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray) - 1. # approximate
            self.render_step_size_bg = 0.01 # render_step_size = max(distance_to_camera * self.cone_angle, self.render_step_size)

        self.variance = VarianceNetwork(self.config.variance)
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = "black"
        self.near_plane, self.far_plane = None, None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        self.n_bins = 1500
        self.exposure_time = 0.002398339664
            
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)
        
        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = torch.tensor(1.0) if cos_anneal_end == 0 else torch.tensor( min(1.0, global_step / cos_anneal_end)).to(self.scene_aabb)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            
            if global_step < 10000:
                occ_thre = 1e-6 
            else:
                occ_thre = 1e-3
                
            self.occupancy_grid.every_n_step(step=global_step, ema_decay=   0.95, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre)
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(step=global_step, ema_decay=0.95, occ_eval_fn=occ_eval_fn_bg, occ_thre=occ_thre)

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, intervals):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * intervals.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * intervals.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha


    def forward_regnerf(self, unseen_rays, regnerf_laser_kernel):
        n_rays = unseen_rays.shape[0]
        assert n_rays == self.config.train_patch_size ** 2
        rays_o, rays_d = unseen_rays[:, 0:3], unseen_rays[:, 3:6] # both (N_rays, 3)
        
        def alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, t_ends - t_starts)
            return alpha[...,None]
        
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
        )
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        intervals = t_ends - t_starts
        distances_from_origin = midpoints.squeeze(-1)
        
        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)

        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, intervals)[...,None] #NOTE: be careful here, think adjacent sample
        rgb = self.texture(feature, t_dirs, normal)
        # rgb = torch.exp(rgb) - 1 #taking exp again as double exp enforces non-negativity and helps with HDR
        weights_non_squared = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        actual_colors = accumulate_along_rays(weights_non_squared, ray_indices, values=rgb, n_rays=n_rays)
        only_weights = weights_non_squared

        #NOTE: if only one sample is found, right pad with zeros
        if t_starts.shape[0] == 1:
            depths = (t_starts+t_ends)/2
            left_padding = 0
            right_padding = self.config.train_patch_size ** 2 - 1
            depths = F.pad(depths, (left_padding, right_padding), "constant", 0).view(self.config.train_patch_size**2, -1)
        else:
            #Reason we adjust this is because some rays don't have samples so the argmax at those rays is the last index
            out, argmax = scatter_max(torch.squeeze(only_weights), ray_indices, out=torch.zeros(n_rays, device=ray_indices.device)) #both are (n_rays, )
            argmax[argmax==only_weights.shape[0]] = only_weights.shape[0]-1 #shifting the index to the last index
            if t_starts.shape[0] > 0:
                depths = (t_starts+t_ends)[argmax]/2 #(self.config.train_patch_size ** 2, 1)
            else:
                depths = out[:, None] 
                
        depth_variance = ((t_ends - depths[ray_indices]) ** 2 + (t_ends - depths[ray_indices])*(t_starts - depths[ray_indices]) + (t_starts - depths[ray_indices]) ** 2)/3
        weight_variance = accumulate_along_rays(weights_non_squared, ray_indices, depth_variance, n_rays)

        weights = (only_weights ** 2 / (alpha + 1e-9))
        src = weights * rgb #Alpha compose
        times = distances_from_origin
        try:
            src = src/(times[:, None].detach()**2 + 1e-10) #square falloff of light
        except:
            print("here")
        
        
        #Mapping the predicted colors to bins
        tfilter_sigma = 3
        bin_mapping, dist_weights = mapping_dist_to_bin_mitsuba(distances_from_origin, self.n_bins, self.exposure_time, c=1, sigma=tfilter_sigma) #(n_samples, 24)
        src = (dist_weights[..., None] * src[:, None, :]).flatten(0, 1)
        colors = torch.zeros((n_rays * self.n_bins, 3), device=weights.device)
        #Creates a bunch of indices for the scatter add
        #NOTE: those indices are not unique. Basically, for each channel, say, index = [1805, 1806, 1807, ...], then colors[1805] sums over all elements in src that are assigned the 1805 index
        index = ((torch.repeat_interleave(ray_indices, 8*tfilter_sigma) * self.n_bins) + bin_mapping.flatten().long())[:, None].expand(-1, 3).long() # (n_samples * 24, 3) 
        #Add all colors to the respective bins
        colors.scatter_add_(0, index, src)
        colors = colors.view(n_rays, self.n_bins, 3)
        colors = convolve_colour(colors, regnerf_laser_kernel, n_bins=self.n_bins)

        bin_numbers_floor, bin_numbers_ceil, alpha = mapping_dist_to_bin(distances_from_origin, self.n_bins, self.exposure_time)
        index_f = ((ray_indices * self.n_bins) + bin_numbers_floor.long())[:, None].expand(-1, 3).long()
        index = index_f
        
        opacity = accumulate_along_rays(only_weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
        patch_size = int(math.sqrt(actual_colors.shape[0]))
        out = {
            "color_patch": actual_colors.view(patch_size, patch_size, 3),
            "depth_patch": depths.view(patch_size, patch_size),
            "opacity_patch": opacity.view(patch_size, patch_size),
            "transient_patch": colors.view(-1, self.n_bins, 3),
            "depth_variance_patch": weight_variance.view(patch_size, patch_size),
            "sdf_grad_samples_patch": sdf_grad,
            "num_samples_regnerf": len(ray_indices)
        }
        
        return out
    
    
    
    def forward_(self, rays, laser_kernel):
        
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid,
                alpha_fn=None,
                near_plane=self.near_plane, far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )   
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints 
        intervals = t_ends - t_starts
        distances_from_origin = midpoints.squeeze(-1)
        
        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
    
        #TODO: Possible extension HERE, feed the transients into a ViT to extract features and then feed that as auxiliary input to self.geometry
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, intervals)[...,None] #NOTE: be careful here, think adjacent sample
        rgb = self.texture(feature, t_dirs, normal)
        rgb = torch.exp(rgb) - 1 #taking exp again as double exp enforces non-negativity and helps with HDR
        weights_non_squared = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        only_weights = weights_non_squared
        
        '''
        Transient NeRF territory below
        '''
        
        weights = (only_weights ** 2 / (alpha+ 1e-9))
        src = weights * rgb #Alpha compose
        times = 2*distances_from_origin
        
        src = src/(times[:, None].detach()**2 + 1e-10) #square falloff of light
            
        
        bin_numbers_floor, bin_numbers_ceil, _ = mapping_dist_to_bin(distances_from_origin, self.n_bins, self.exposure_time)    
        colors = torch.zeros((n_rays * self.n_bins, 3), device=weights.device)    
        index = ((ray_indices * self.n_bins) + bin_numbers_floor.long())[:, None].expand(-1, 3).long()   


        colors.scatter_add_(0, index, src)    
        colors = colors.view(n_rays, self.n_bins, 3)


        # do the same for the sigmas
        comp_weights = torch.zeros((n_rays * self.n_bins, 1), device=weights.device)
        comp_weights.scatter_add_(0, index[:, [0]], only_weights)
        comp_weights = comp_weights.reshape(n_rays, self.n_bins)


        colors = convolve_colour(colors, laser_kernel, n_bins=self.n_bins)

        opacity = accumulate_along_rays(
            only_weights, ray_indices=ray_indices, values=None, n_rays=n_rays
        )
        comp_normal = accumulate_along_rays(only_weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        
        if t_starts.shape[0] == 1:
            depths = (t_starts+t_ends)/2
            left_padding = 0
            right_padding = self.config.ray_chunk - 1
            depths = F.pad(depths, (left_padding, right_padding), "constant", 0).view(self.config.ray_chunk, -1)
        else:
            #Reason we adjust this is because some rays don't have samples so the argmax at those rays is the last index
            out, argmax = scatter_max(torch.squeeze(only_weights), ray_indices, out=torch.zeros(n_rays, device=ray_indices.device)) #both are (n_rays, )
            argmax[argmax==only_weights.shape[0]] = only_weights.shape[0]-1 #shifting the index to the last index
            if t_starts.shape[0] > 0:
                depths = (t_starts+t_ends)[argmax]/2
            else:
                depths = out[:, None]

        to_accum_var = ((t_starts + t_ends)[..., None] - depths[ray_indices][...,None])**2
        depths_variance = accumulate_along_rays(
            weights = only_weights,
            ray_indices= ray_indices,
            values=to_accum_var.view(to_accum_var.shape[0], 1),
            n_rays=n_rays,
        )
        depth_variance = ((t_ends - depths[ray_indices]) ** 2 + (t_ends - depths[ray_indices])*(t_starts - depths[ray_indices]) + (t_starts - depths[ray_indices]) ** 2)/3
        weight_variance = accumulate_along_rays(weights_non_squared, ray_indices, depth_variance, n_rays)

        out = {
            'opacity': opacity,
            'depth': torch.squeeze(depths),
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'rgb': colors #the predicted transients
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'comp_weights': comp_weights,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1),
                'weight_variance': weight_variance,
                'distances_from_origin': distances_from_origin
            })
            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': sdf_laplace
                })
                
        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                # 'actual_colors': self.background_color[None,:].expand(*actual_colors.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }
        out_full = {
            # 'actual_colors': out['actual_colors'] + out_bg['actual_colors'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }
        
        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }
        


    
    def forward(self, rays_dict, laser_kernel):
        '''
        Performs forward pass from rays dict from the system to the model
        the rays_dict contains the rays and the corresponding metadata
        in the form of {something_rays: {"rays": rays, "metadata": metadata}}
        '''
        training_rays = rays_dict["training_rays"]["rays"]
        stage = rays_dict["stage"]
        if self.training:
            out = {}
            if self.config.use_reg_nerf:
                regnerf_patch = rays_dict["regnerf_patch"]["rays"]
                regnerf_out = self.forward_regnerf(regnerf_patch, laser_kernel)
                out.update(regnerf_out)
                
            out.update(self.forward_(training_rays, laser_kernel))
        if stage == "validation":
            if self.config.use_reg_nerf:
                regnerf_patch = rays_dict["regnerf_patch"]["rays"]
                out = chunk_batch(self.forward_, self.config.ray_chunk, True, regnerf_patch, laser_kernel)
            else:
                out = chunk_batch(self.forward_, self.config.ray_chunk, True, training_rays, laser_kernel)
        elif stage =="test":
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, training_rays, laser_kernel)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            rgb = torch.sigmoid(torch.log(rgb)) #Remap to 0-1 in order to get the right colors
            mesh['v_rgb'] = rgb.cpu()
        return mesh