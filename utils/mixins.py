import os
import re
import shutil
import numpy as np
import cv2
import imageio
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import json
from prettytable import PrettyTable
import torch
import matplotlib.patches as patches

from utils.obj import write_obj


class SaverMixin():
    @property
    def save_dir(self):
        return self.config.save_dir
    
    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))
    
    def get_save_path(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    
    DEFAULT_RGB_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1)}
    DEFAULT_UV_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1), 'cmap': 'checkerboard'}
    DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}

    def get_rgb_image_(self, img, data_format, data_range):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = ((img - data_range[0]) / (data_range[1] - data_range[0]) * 255.).astype(np.uint8)
        imgs = [img[...,start:start+3] for start in range(0, img.shape[-1], 3)]
        imgs = [img_ if img_.shape[-1] == 3 else np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1) for img_ in imgs]
        img = np.concatenate(imgs, axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def save_rgb_image(self, filename, img, data_format=DEFAULT_RGB_KWARGS['data_format'], data_range=DEFAULT_RGB_KWARGS['data_range']):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(self.get_save_path(filename), img)
    
    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ['checkerboard', 'color']
        if cmap == 'checkerboard':
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[...,0] + mask[...,1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == 'color':
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img
    
    def save_uv_image(self, filename, img, data_format=DEFAULT_UV_KWARGS['data_format'], data_range=DEFAULT_UV_KWARGS['data_range'], cmap=DEFAULT_UV_KWARGS['cmap']):
        img = self.get_uv_image_(img, data_format, data_range, cmap)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img)
        img = np.nan_to_num(img)
        if data_range is None:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in [None, 'jet', 'magma']
        if cmap == None:
            img = (img * 255.).astype(np.uint8)
            img = np.repeat(img[...,None], 3, axis=2)
        elif cmap == 'jet':
            img = (img * 255.).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == 'magma':
            img = 1. - img
            base = cm.get_cmap('magma')
            num_bins = 256
            colormap = LinearSegmentedColormap.from_list(
                f"{base.name}{num_bins}",
                base(np.linspace(0, 1, num_bins)),
                num_bins
            )(np.linspace(0, 1, num_bins))[:,:3]
            a = np.floor(img * 255.)
            b = (a + 1).clip(max=255.)
            f = img * 255. - a
            a = a.astype(np.uint16).clip(0, 255)
            b = b.astype(np.uint16).clip(0, 255)
            img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
            img = (img * 255.).astype(np.uint8)
        return img

    def save_grayscale_image(self, filename, img, data_range=DEFAULT_GRAYSCALE_KWARGS['data_range'], cmap=DEFAULT_GRAYSCALE_KWARGS['cmap']):
        img = self.get_grayscale_image_(img, data_range, cmap)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_image_grid_(self, imgs):
        if isinstance(imgs[0], list):
            return np.concatenate([self.get_image_grid_(row) for row in imgs], axis=0)
        cols = []
        for col in imgs:
            assert col['type'] in ['rgb', 'uv', 'grayscale']
            if col['type'] == 'rgb':
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col['kwargs'])
                cols.append(self.get_rgb_image_(col['img'], **rgb_kwargs))
            elif col['type'] == 'uv':
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col['kwargs'])
                cols.append(self.get_uv_image_(col['img'], **uv_kwargs))
            elif col['type'] == 'grayscale':
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col['kwargs'])
                cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
        return np.concatenate(cols, axis=1)
    
    def save_image_grid(self, filename, imgs):
        img = self.get_image_grid_(imgs)
        cv2.imwrite(self.get_save_path(filename), img)
    
    def save_image_plot_grid(self, filename, imgs):
        '''Assume the imgs are already preprocessed and ready to be plotted.
        imgs is a list of dicts where each dict has some keys with plotting options such as title or cmap.
        
        '''
        fig, axes = plt.subplots(1, len(imgs), figsize=(20, 5))
        for i, img_dict in enumerate(imgs):
            if img_dict["type"] == "rgb":
                axes[i].imshow(img_dict["img"])
        
            elif img_dict["type"] == "depth":
                axes[i].imshow(img_dict["img"], cmap=img_dict["kwargs"]["cmap"], vmin=img_dict["kwargs"]["vmin"], vmax=img_dict["kwargs"]["vmax"])
                
            axes[i].set_title(img_dict["kwargs"]["title"])
            axes[i].axis('off')
            
        plt.savefig(self.get_save_path(filename))
        plt.close(fig)
    
    
    def save_image(self, filename, img):
        img = self.convert_data(img)
        assert img.dtype == np.uint8
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(self.get_save_path(filename), img)
    
    def save_cubemap(self, filename, img, data_range=(0, 1)):
        img = self.convert_data(img)
        assert img.ndim == 4 and img.shape[0] == 6 and img.shape[1] == img.shape[2]

        imgs_full = []
        for start in range(0, img.shape[-1], 3):
            img_ = img[...,start:start+3]
            img_ = np.stack([self.get_rgb_image_(img_[i], 'HWC', data_range) for i in range(img_.shape[0])], axis=0)
            size = img_.shape[1]
            placeholder = np.zeros((size, size, 3), dtype=np.float32)
            img_full = np.concatenate([
                np.concatenate([placeholder, img_[2], placeholder, placeholder], axis=1),
                np.concatenate([img_[1], img_[4], img_[0], img_[5]], axis=1),
                np.concatenate([placeholder, img_[3], placeholder, placeholder], axis=1)
            ], axis=0)
            img_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
            imgs_full.append(img_full)
        
        imgs_full = np.concatenate(imgs_full, axis=1)
        cv2.imwrite(self.get_save_path(filename), imgs_full)

    def save_data(self, filename, data):
        data = self.convert_data(data)
        if isinstance(data, dict):
            if not filename.endswith('.npz'):
                filename += '.npz'
            np.savez(self.get_save_path(filename), **data)
        else:
            if not filename.endswith('.npy'):
                filename += '.npy'
            np.save(self.get_save_path(filename), data)
        
    def save_state_dict(self, filename, data):
        torch.save(data, self.get_save_path(filename))
    
    def save_img_sequence(self, filename, img_dir, matcher, save_format='gif', fps=30):
        assert save_format in ['gif', 'mp4']
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        matcher = re.compile(matcher)
        img_dir = os.path.join(self.save_dir, img_dir)
        imgs = []
        for f in os.listdir(img_dir):
            if matcher.search(f):
                imgs.append(f)
        imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]
        
        if save_format == 'gif':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(self.get_save_path(filename), imgs, fps=fps, palettesize=256)
        elif save_format == 'mp4':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(self.get_save_path(filename), imgs, fps=fps)
    
    def save_mesh(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None, v_rgb=None):
        '''
        Return the path to the saved mesh file and also saves the mesh file.
        '''
        v_pos, t_pos_idx = self.convert_data(v_pos), self.convert_data(t_pos_idx)
        if v_rgb is not None:
            v_rgb = self.convert_data(v_rgb)

        import trimesh
        mesh = trimesh.Trimesh(
            vertices=v_pos,
            faces=t_pos_idx,
            vertex_colors=v_rgb
        )
        saved_path = self.get_save_path(filename)
        mesh.export(saved_path)
        return saved_path
    
    def save_file(self, filename, src_path):
        shutil.copyfile(src_path, self.get_save_path(filename))
    
    def save_json(self, filename, payload):
        with open(self.get_save_path(filename), 'w') as f:
            f.write(json.dumps(payload))
            
    def save_plot(self, filename, predicted, gt):
        assert predicted.dim() == 1
        assert predicted.shape == gt.shape
        plt.plot(predicted.cpu().detach().numpy(), label='predicted transient')
        plt.plot(gt.cpu().detach().numpy(), label='ground truth transient')
        plt.legend()
        plt.title("Channel-Averaged Ground Truth vs Predicted Transient")
        plt.savefig(self.get_save_path(filename))
        plt.close()
        
        
    def save_depth_plot(self, filename, predicted, gt, depth, exposure_time):
        '''
        Logic: pick the first gt transient of the first ray and average intensity across color channel and plot the depth values
        '''
        plt.plot(torch.mean(predicted[0], dim=-1).cpu().detach().numpy(), label = "Predicted transient")
        plt.plot(torch.mean(gt[0], dim=-1).cpu().detach().numpy(), label = "Ground truth transient")
        plt.axvline(x=(depth[0]*2/exposure_time).detach().cpu().numpy(), color='r', linestyle='--', label='Predicted depth')
        plt.legend(fontsize="small")
        plt.title("Ground Truth vs Predicted Transient with Depth for the first visible ray and first color channel", fontsize=10)
        
        plt.savefig(self.get_save_path(filename))
        plt.close()
        
        
    def save_sdf_plot(self, filename, sdf, exposure_time, distances_from_origin, depth, ray_indices, alive_ray_mask):
        '''
        Plot the sdfs plot as a function of the distances from the origin and depth
        '''
        #Find the first ray that is visible and plot the SDFs of that ray
        first_alive_ray = torch.nonzero(alive_ray_mask)[0]
        distances_from_origin = distances_from_origin[ray_indices == first_alive_ray]
        sdf = sdf[ray_indices == first_alive_ray]
        
        distances_from_origin = 2 * distances_from_origin / exposure_time
        plt.plot(distances_from_origin.cpu().detach().numpy(), sdf.cpu().detach().numpy(), label='SDFs')
        plt.axvline(x=(depth[0]*2/exposure_time).detach().cpu().numpy(), color='r', linestyle='--', label='Predicted depth')
        plt.legend()
        plt.title("Predicted SDFs along the first ray")
        
        plt.savefig(self.get_save_path(filename))
        plt.close()
        
        
    def save_sdf_normal_plot(self, filename, sdf, distances_from_origin, depth, ray_indices, alive_ray_mask):
        '''
        Plot the sdfs plot as a function of the distances from the origin and depth
        '''
        #Find the first ray that is visible and plot the SDFs of that ray
        first_alive_ray = torch.nonzero(alive_ray_mask.squeeze())[0]
        distances_from_origin = distances_from_origin[ray_indices == first_alive_ray]
        sdf = sdf[ray_indices == first_alive_ray]
        plt.plot(distances_from_origin.cpu().detach().numpy(), sdf.cpu().detach().numpy(), label='SDFs')
        plt.axvline(x=(depth[0]).detach().cpu().numpy(), color='r', linestyle='--', label='Predicted depth')
        plt.legend()
        plt.title("Predicted SDFs along the first ray")
        
        plt.savefig(self.get_save_path(filename))
        plt.close()



    def save_image_n_dots(self, filename, image, points_x, points_y):
        '''
        Plot the predicted image along with the three red dots on the image.
        image: (H,W,3)
        points_x: (3,1)
        points_y: (3,1)
        '''
        assert len(image.shape) == 3, "Image should be of shape (H,W,3)"
        assert image.shape[:-1] == self.config.dataset.img_wh
        plt.imshow(image)
        plt.plot(points_x, points_y,'.', markersize=10, color='red')
        plt.title("Image with 3 sampled dots")
        plt.savefig(self.get_save_path(filename))
        plt.close()
        
        
    def save_sdf_plots(self, filename, top_3, dists_from_origin, ray_indices, exposure_time, depth, sdf):
        '''
        Given three red dots on the image, plot the SDFs of each point 
        point_x: (3,1)
        point_y: (3,1)
        sdfs: (n_samples, 1)
        In the original script, they use point_y = plotting_subset_y
        '''
        sdf_figure, sdf_axes = plt.subplots(1, len(top_3), figsize=(7.5, 10), dpi=250)
        dists_from_origin = 2 * dists_from_origin / exposure_time
        
        for idx, ax in enumerate(sdf_axes):
            #PLotting the SDFs

            sdf_min = sdf[ray_indices == top_3[idx]].min().item()
            sdf_max = sdf[ray_indices == top_3[idx]].max().item()

            # If you want a buffer around the min and max, you can add it
            buffer = (sdf_max - sdf_min) * 0.1  # 10% buffer for example
            sdf_min -= buffer
            sdf_max += buffer
            ax.set_ylim([sdf_min, sdf_max])
            sdf_ticks = np.linspace(sdf_min, sdf_max, num=5)
            
            #append zero to the set of ticks
            # if current_min <= 0 and current_max >= 0:
            #     # Include 0 in the ticks
            #     sdf_ticks = np.append(sdf_ticks, 0)
            #     # Sort the ticks since append doesn't maintain order
            #     sdf_ticks = np.sort(sdf_ticks)
            
            ax.set_yticks(sdf_ticks)
            ax.yaxis.tick_right()
            ax.scatter(dists_from_origin[ray_indices == top_3[idx]].detach().cpu().numpy(), sdf[ray_indices == top_3[idx]].detach().cpu().numpy(), alpha=0.5, color="blue", label='SDFs')
            
            # x = dists_from_origin[ray_indices == top_3[idx]].detach().cpu().numpy()
            # y = sdf[ray_indices == top_3[idx]].detach().cpu().numpy()

            # # Perform linear regression
            # slope, intercept, _,_,_ = stats.linregress(x, y)

            # # Generate points for the fitted line
            # fitted_line = slope * x + intercept

            # Plot the fitted line
            # ax.plot(x, fitted_line, color="red", label='Fitted SDF')
            # ax.legend(loc='best')
            
            ax.axvline(x=(depth[top_3][idx] * 2 / exposure_time)[0].detach().cpu().numpy(), color='magenta', linestyle='--', linewidth=2, label='Depth Line')
            shape_info = ray_indices[ray_indices == top_3[idx]].shape[0]
            depth_info = (depth[top_3][idx] * 2 / exposure_time)[0].detach().cpu().numpy()
            #Displaying the number of samples belonging to those sampled rays
            ax.text(0.95, 0.95, f'Number of samples: {shape_info}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5))
            ax.text(0.95, 0.7, f'Depth: {depth_info:.2f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5))
            ax.set_xlabel('Distance from the origin')
            ax.set_ylabel('SDF values')
            ax.set_title(f'Plot {idx+1}')
            
        sdf_figure.savefig(self.get_save_path(filename))
        plt.close(sdf_figure)
            
    def save_metrics(self, filename, metrics_dict):
        '''
        Save the metrics in a nice table format.
        Expects the metrics_dict to contain the names of all the metrics
        '''
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for key in metrics_dict.keys():
            table.add_row([key, metrics_dict[key]])
        table.border = True
        save_path = self.get_save_path(filename)
        with open(save_path, 'w') as f:
            f.write(str(table))
    
    
    def save_plot_grid(self, filename, point_x, point_y, transient, global_step, channel=0):
        assert len(point_x) == len(point_y)
        
        n_points = len(point_x)
        fig = plt.figure(figsize=(20, 5))  # Define a figure
        
        # Create a GridSpec layout
        gs = GridSpec(1, n_points + 1, width_ratios=[3] + [1]*n_points)  # First subplot 3 times wider
        
        integrated_transient = transient.sum(-2)  # sums along the time dimension to get the integrated transient
        integrated_transient = (integrated_transient / integrated_transient.max()) ** (1 / 2.2)  # gamma correct it
        
        # Plot the integrated transient in a larger subplot
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(integrated_transient)
        for i in range(n_points):
            ax0.scatter(point_x[i], point_y[i], color="red")
        ax0.set_title(f'Integrated transient at {global_step} steps')
        
        # Plot the transients for the selected points
        for i in range(n_points):
            ax = fig.add_subplot(gs[0, i+1])  # Create each subsequent subplot
            selected_transient = transient[point_y[i], point_x[i], :, channel]
            argmax_index = np.argmax(selected_transient)
            ax.plot(selected_transient)
            ax.set_title(f'Plot {i+1} - {point_x[i], point_y[i]}')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Intensity for Channel {channel}')
            
            textstr = f'argmax: {argmax_index}'
            # These are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # Place the text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8.5, verticalalignment='top', bbox=props)
            
        plt.tight_layout()
        plt.savefig(self.get_save_path(filename))
        plt.close(fig)
        
        
    # def save_weight_grid(self, filename, transient, weights, ray_indices, global_step):
    #     '''
    #     Plot the weights of the selected points
    #     '''
    #     height = transient.shape[0]
    #     n_points = len(point_x)
    #     fig = plt.figure(figsize=(20, 5))
    #     gs = GridSpec(1, n_points + 1, width_ratios=[3] + [1]*n_points)  # First subplot 3 times wider
    #     integrated_transient = transient.sum(-2)  # sums along the time dimension to get the integrated transient
    #     integrated_transient = (integrated_transient / integrated_transient.max()) ** (1 / 2.2)  # gamma correct it
        
    #     # Plot the integrated transient in a larger subplot
    #     ax0 = fig.add_subplot(gs[0, 0])
    #     ax0.imshow(integrated_transient)
    #     for i in range(n_points):
    #         ax0.scatter(point_x[i], point_y[i], color="red")
    #     ax0.set_title(f'Integrated transient at {global_step} steps')
        
    #     for i in range(n_points):
    #         ax = fig.add_subplot(gs[0, i+1])  # Create each subsequent subplot
    #         #convert point_y[i], point_x[i] into the specific ray index
            
    #         #Find the ray index for the selected point
    #         linear_index=point_y[i]*height+point_x[i]
    #         ray_index = ray_indices[ray_indices == linear_index]
    #         selected_weight = weights[ray_index]
    #         ax.plot(selected_weight)
    #         ax.set_title(f'Plot {i+1} - {point_x[i], point_y[i]}')
    #         ax.set_xlabel('Time')
    #         ax.set_ylabel('Weights')
            
    #     plt.tight_layout()
    #     plt.savefig(self.get_save_path(filename))
    #     plt.close(fig)
        
        
    def save_weight_plot(self, filename, weights, exposure_time, distances_from_origin, depth, ray_indices, alive_ray_mask):
        '''
        Plot the weights plot as a function of the distances from the origin and depth
        '''
        #Find the first ray that is visible and plot the SDFs of that ray
        first_alive_ray = torch.nonzero(alive_ray_mask)[0]
        distances_from_origin = distances_from_origin[ray_indices == first_alive_ray]
        weights = weights[ray_indices == first_alive_ray]
        
        distances_from_origin = 2 * distances_from_origin / exposure_time
        plt.plot(distances_from_origin.cpu().detach().numpy(), weights.cpu().detach().numpy(), label='Weights')
        plt.axvline(x=(depth[0]*2/exposure_time).detach().cpu().numpy(), color='r', linestyle='--', label='Predicted depth')
        plt.legend()
        plt.title("Predicted weights along the first ray")
        
        plt.savefig(self.get_save_path(filename))
        plt.close()
        
        
        
    def save_transient_figures(self, filename, gt_transients, predicted_transients, number_of_transients=20):
        '''
        Plot a figure of all gt_transients and predicted_transients (assuming that they are already masked for the visible rays)
        Take the first color channel.
        '''
        mask = gt_transients.abs().sum(dim=(1, 2)) > 1e-5
        mask = torch.ones(mask.shape, dtype=torch.bool)
        non_zero_gt_transients = gt_transients[mask]
        non_zero_pred_transients = predicted_transients[mask]
        
        num_rows = (number_of_transients + 1) // 2  # Ensure there's enough rows        
        
        plt.figure(figsize=(15,20))
        number_of_plots = min(number_of_transients, len(non_zero_gt_transients))
        for i in range(number_of_plots):
            plt.subplot(num_rows,2,i+1)
            plt.plot(non_zero_gt_transients[i,:,0].cpu().detach().numpy(), label='Non-zero Ground truth transient')
            plt.plot(non_zero_pred_transients[i,:,0].cpu().detach().numpy(), label='Non-zero Predicted transient')
            plt.title(f"Transients for the {i+1}th ray")
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(self.get_save_path(filename))
        plt.close()
            
        
        
        
    def save_color_figures(self, filename, gt_transients, predicted_transients, number_of_transients=20):
        '''
        Plot a figure of all gt vs predicted integrated transients (assuming that they are already masked for the visible rays)
        '''
        #First need to normalize + gamma correct
        gt_transients = gt_transients.sum(-2)
        gt_transients = gt_transients / gt_transients.max()
        gt_transients = gt_transients ** (1/2.2)
        
        predicted_transients = predicted_transients.sum(-2)
        predicted_transients = predicted_transients / predicted_transients.max()
        predicted_transients = predicted_transients ** (1/2.2)
        
        
        # Create a figure with enough subplots in a single column
        fig, ax = plt.subplots(figsize=(4, 20))  # Adjust the size as needed

        for i in range(20):
            gt_transients_pixel = patches.Rectangle((0, i * 1.05), 1, 1, linewidth=1, edgecolor='none', facecolor=gt_transients[i].numpy())
            ax.add_patch(gt_transients_pixel)

            predicted_transients_pixel = patches.Rectangle((1.05, i * 1.05), 1, 1, linewidth=1, edgecolor='none', facecolor=predicted_transients[i].numpy())
            ax.add_patch(predicted_transients_pixel)

        ax.text(0.5, -0.5, 'gt_color', va='center', ha='center')
        ax.text(1.55, -0.5, 'predicted_color', va='center', ha='center')

        # Set plot limits and aspect
        ax.set_xlim(0, 2.1)
        ax.set_ylim(0, 21)
        ax.set_aspect('auto')  # Auto aspect to fit the layout
        plt.axis('off')  # Hide the axes

        # Show the plot
        plt.savefig(self.get_save_path(filename))
        plt.close()