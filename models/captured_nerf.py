import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

import models
from models.base import BaseModel
from models.utils import chunk_batch, mapping_dist_to_bin_mitsuba, mapping_dist_to_bin, torch_laser_kernel, convolve_colour
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, accumulate_along_rays

import numpy as np
import scipy.io as sio

@models.register('captured-nerf')
class CapturedNeRFModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))

        if self.config.learned_background:
            self.occupancy_grid_res = 256
            self.near_plane, self.far_plane = 0.2, 1e4
            self.cone_angle = 10**(math.log10(self.far_plane) / self.config.num_samples_per_ray) - 1. # approximate
            self.render_step_size = 0.01 # render_step_size = max(distance_to_camera * self.cone_angle, self.render_step_size)
            self.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        else:
            self.occupancy_grid_res = 128
            self.near_plane, self.far_plane = None, None
            self.cone_angle = 0.0
            self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
            self.contraction_type = ContractionType.AABB
            self.n_bins = 1500
            self.exposure_time = 0.002398339664

        self.geometry.contraction_type = self.contraction_type

        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.occupancy_grid_res,
                contraction_type=self.contraction_type
            )
        self.randomized = self.config.randomized
        self.background_color = "black"
            
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)

        def occ_eval_fn(x):
            density, _ = self.geometry(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size) based on taylor series
            return density[...,None] * self.render_step_size
        
        if self.training and self.config.grid_prune:
            if global_step < 10000:
                thre = 1e-6
            else:
                thre = 1e-3
            self.occupancy_grid.every_n_step(step=global_step, ema_decay=0.95, occ_eval_fn=occ_eval_fn, occ_thre=thre)

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def forward_(self, rays, laser_kernel):
        
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry(positions)
            return density[...,None]
        
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, feature = self.geometry(positions) 
            rgb = self.texture(feature, t_dirs)
            return rgb, density[...,None]

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid,
                sigma_fn=sigma_fn,
                near_plane=self.near_plane, far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=self.cone_angle,
                alpha_thre=0.0
            )   
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints 
        distances_from_origin = midpoints
        intervals = t_ends - t_starts
        density, feature = self.geometry(positions) 
        rgb = self.texture(feature, t_dirs)
        rgb = torch.exp(rgb) - 1 #taking exp again as double exp enforces non-negativity
        weights_non_squared = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays) 
        actual_colors = accumulate_along_rays(weights_non_squared, values = rgb, ray_indices=ray_indices, n_rays=n_rays)
        only_weights = weights_non_squared
        alphas = (1 - torch.exp(-2*density[...,None] * (t_ends - t_starts)))/2 # new formula
        
        
        '''
        Transient NeRF territory below
        '''
        
        weights = (only_weights ** 2 / (alphas+ 1e-9))
        src = weights * rgb #Alpha compose
        times = distances_from_origin
        src = src/(times.detach()**2 + 1e-10) #square falloff of light
        
        bin_numbers_floor, bin_numbers_ceil, _ = mapping_dist_to_bin(distances_from_origin, self.n_bins, self.exposure_time)    
        colors = torch.zeros((n_rays * self.n_bins, 3), device=weights.device)    
        index = ((ray_indices * self.n_bins) + torch.squeeze(bin_numbers_floor.long()))[:, None].expand(-1, 3).long()   
        colors.scatter_add_(0, index, src)    
        colors = colors.view(n_rays, self.n_bins, 3)
        colors = convolve_colour(colors, laser_kernel, n_bins=self.n_bins)


        # do the same for the sigmas
        comp_weights = torch.zeros((n_rays * self.n_bins, 1), device=weights.device)
        comp_weights.scatter_add_(0, index[:, [0]], only_weights)
        comp_weights = comp_weights.reshape(n_rays, self.n_bins)

        opacity = accumulate_along_rays(
            only_weights, ray_indices=ray_indices, values=None, n_rays=n_rays
        )

        # depths = accumulate_along_rays(
        #     weights_non_squared,
        #     ray_indices,
        #     values=(t_starts + t_ends) / 2.0,
        #     n_rays=n_rays,
        # )
        # depths = depths/(opacities+1e-10)
        
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
        depths_variance = depths_variance/(opacity+1e-10)

        out = {
            "actual_colors": actual_colors,
            'opacity': opacity,
            'depth': torch.squeeze(depths),
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
            'rgb': colors #the predicted transients
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'comp_weights': comp_weights,
                'ray_indices': ray_indices.view(-1)
            })
        
        return out


    
    def forward(self, rays, laser_kernel):
        if self.training:
            out = self.forward_(rays, laser_kernel)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays, laser_kernel)
        return {
            **out,
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

    #TODO: viewing direction might be inverted for captured
    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank))
            viewdirs = torch.zeros(feature.shape[0], 3).to(feature)
            viewdirs[...,2] = -1. # set the viewing directions to be -z (looking down)
            rgb = self.texture(feature, viewdirs)
            rgb = torch.sigmoid(torch.log(rgb)).clamp(0,1)
            mesh['v_rgb'] = rgb.cpu()
        return mesh
