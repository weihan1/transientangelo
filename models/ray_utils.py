import torch
import numpy as np
import scipy
import json
import torch.nn.functional as F
from scipy.optimize import minimize
import cv2


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def spatial_filter(x, y, sigma, rep, prob_dithering=False, normalize=True):
    """
    Use of a truncated Gaussian to model the spatial footprint of the laser spot and SPAD sensor projected onto the scene
    Rays are sampled at 4 standard deviations from the ray center weighting their contribution to the rendered transient by the corresponding Gaussian probability density function value.
    
    Args:
        x (numpy.ndarray): The x pixel coordinates (n_rays) containing elements in the range (0, WIDTH)
        y (numpy.ndarray): The y pixel coordinates (n_rays) containing elements in the range (0, HEIGHT)
        sigma (float): The standard deviation of the Gaussian distribution. (0.15)
        rep (int): The number of repetitions.
        prob_dithering (bool, optional): Whether to use probabilistic dithering. Defaults to True.
        normalize (bool, optional): Whether to normalize the weights. Defaults to True.


    """
    
    pdf_fn = lambda x: np.exp(-x/(2*sigma**2)) - np.exp(-16)
    if prob_dithering:
        #the bounds are determined by 4 standard deviations from ray center.
        bounds_max = [4*sigma]*x.shape[0]
        loc = 0
        # Generate s_x, s_y based on a truncated normal distribution with bounds as the first two arguments
        s_x = scipy.stats.truncnorm.rvs((-np.array(bounds_max)-loc)/sigma, (np.array(bounds_max)-loc)/sigma, loc=loc, scale=sigma)
        s_y = scipy.stats.truncnorm.rvs((-np.array(bounds_max)-loc)/sigma, (np.array(bounds_max)-loc)/sigma, loc=loc, scale=sigma)
        weights = np.ones_like(s_x)*1/rep
    
    else: 
        s_x = np.random.uniform(low=-4*sigma, high=4*sigma, size=(rep, x.shape[0]//rep)) #(self.rep, num_rays)
        s_y = np.random.uniform(low=-4*sigma, high=4*sigma, size=(rep, x.shape[0]//rep))#(self.rep, num_rays)
        dists = (s_x**2 + s_y**2) #Look at how far they are from the origin
        weights = pdf_fn(dists)
        if normalize:
            weights = weights/weights.sum(0)[None, :] #if self.rep = 1, then the weights are all the same
        s_x = s_x.flatten()
        s_y = s_y.flatten()
        weights = weights.flatten()
    
    #weights tell you the contribution of each of the repeated rays for a given pixel. For instance, if weights is of shape
    #(3,16), this means that in your batch of 16 rays, you are additionally sampling two other rays. weights[:, i] tells you the 
    #contribution of each of the three rays for the ith ray, contribution sums up to 1
    return s_x, s_y, weights #(all are self.rep * num_rays)

def normalize(v):
    return v / np.linalg.norm(v)

def look_at(pf, pu, t, captured=False):
    '''
    pf is the 3D that you're looking at 
    pu is the up vector
    
    Logic: since pf is not a direction vector, we calculate the forward vector by subtracting it with the camera origin (think -t + pf is the vector from origin to pf)
    '''
    f = normalize(pf - t) #calculate the foward vector by normalizing the difference between the point of interest and the camera origin
    r = np.cross(f, pu) #calculate the right vector by taking the cross product of the forward vector and the up vector
    r = normalize(r)    #normalize the right vector
    u = np.cross(r, f)  #calculate the up vector by taking the cross product of the right vector and the forward vector
    if not captured:
        R = np.stack([r, u, -f], axis=1)
    else:
        R = np.stack([r, u, f], axis=1)
    return R



def sample_point_in_box(x_bounds, y_bounds, z_bounds, sample_boundary=True):
    '''
    Sample a point on the boundary defined by x_bounds, y_bounds, z_bounds
    '''
    
    if sample_boundary:
        boundary_choice = torch.randint(0, 3, (1,)).item()  # 0 for x, 1 for y, 2 for z
        
        if boundary_choice == 0:  # Fix x
            x = x_bounds[torch.randint(0, 2, (1,)).item()].unsqueeze(0)
            y = torch.rand(1) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
            z = torch.rand(1) * (z_bounds[1] - z_bounds[0]) + z_bounds[0]
        elif boundary_choice == 1:  # Fix y
            y = y_bounds[torch.randint(0, 2, (1,)).item()].unsqueeze(0)
            x = torch.rand(1) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
            z = torch.rand(1) * (z_bounds[1] - z_bounds[0]) + z_bounds[0]
        else:  # Fix z
            z = z_bounds[torch.randint(0, 2, (1,)).item()].unsqueeze(0)
            x = torch.rand(1) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
            y = torch.rand(1) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
            
        #Verify the new camera location is within the box
        assert x >= x_bounds[0].item() and x <= x_bounds[1].item()
        assert y >= y_bounds[0].item() and y <= y_bounds[1].item()
        assert z >= z_bounds[0].item() and z <= z_bounds[1].item()
        
    else:
        x = torch.rand(1) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
        y = torch.rand(1) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
        z = torch.rand(1) * (z_bounds[1] - z_bounds[0]) + z_bounds[0]
        assert x >= x_bounds[0].item() and x <= x_bounds[1].item()
        assert y >= y_bounds[0].item() and y <= y_bounds[1].item()
        assert z >= z_bounds[0].item() and z <= z_bounds[1].item()
    return x, y, z

def sample_point_in_sphere(mean_focus_point, known_camera_locations, sample_boundary=True):
    '''
    Sample an unknown camera location on the sphere boundary using spherical coordinates defined as 
    x = radius*sin(\phi)cos(\theta) + center[0]
    y = radius*sing(\phi)sin(\theta) + center[1]
    z = radius*cos(\phi) + center[2]
    Idea: 
    1. randomly sample an azimuth angle from 0 to 2pi
    2. sample the cosine of the zenith angle from -1 to 1
    3. apply arccos to the zenith angle 
    4. plug in the values to the spherical coordinates
    '''
    center = mean_focus_point
    max_radius = torch.sum((known_camera_locations - center)**2, dim=1).mean().sqrt() 
    theta = torch.rand(1) * 2 * np.pi #sample an azimuth angle from 0 to 2pi
    cos_zenith = torch.rand(1) * 2 - 1 #sample the cosine of the zenith angle from -1 to 1
    zenith = torch.acos(cos_zenith) #apply arccos to the zenith angle
    
    if sample_boundary:
        x = max_radius*torch.sin(zenith)*torch.cos(theta) + center[0]
        y = max_radius*torch.sin(zenith)*torch.sin(theta) + center[1]
        z = max_radius*torch.cos(zenith) + center[2]
        squared_distance = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        assert torch.abs(squared_distance - max_radius**2) < 1e-4
        
    else:
        r = (torch.rand(1) ** (1/3)) * max_radius #volume scales with r^3
        x = r*torch.sin(zenith)*torch.cos(theta) + center[0]
        y = r*torch.sin(zenith)*torch.sin(theta) + center[1]
        z = r*torch.cos(zenith) + center[2]
        squared_distance = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        assert squared_distance < max_radius**2
    #Verify the new camera location is on the surface of the sphere
    return x, y, z

    
def vectorized_distance_point_from_line(point, origins, directions):
    """
    Vectorized computation of the smallest distance from a single point to multiple lines defined by origins + t*directions.
    """
    point = np.array(point)
    # Ensure the point is broadcastable over the origins and directions
    point = point.reshape(1, -1)
    # Compute projection factors
    projection_factors = np.sum((point - origins) * directions, axis=1) / np.sum(directions * directions, axis=1) #(n_cameras,)
    # Compute projections
    projections = origins + (projection_factors[:, np.newaxis] * directions) #(n_cameras, 3)
    # Calculate distances
    distances = np.linalg.norm(projections - point, axis=1)**2 #adding squared to help with convergence
    return distances
    

def find_mean_focus_point(known_camera_locations, optical_axes, initial_guess):
    '''
    Find the mean focus point from regnerf using known optical axes.
    Logic is as follows:
    
    1. Given known camera locations, define ith ray r_i(t) = o_i + t*_di
    2. Start with some initial guess for the focus point 
    3. Calculate the distances of the focus point to the optical axes by projecting the focus point onto each optical axes and calculating the distance
    4. Add all distances
    5. Update the focus point by minimizing the sum of distances using solver
    '''
    
    #transform the known camera locations and optical axes to numpy
    known_camera_locations = known_camera_locations.cpu().numpy()
    optical_axes = optical_axes.cpu().numpy()
    
    def objective_function(point):
        '''
        Calculate the total cost of the point to the optical axes
        '''
        distances = vectorized_distance_point_from_line(point, known_camera_locations, optical_axes)
        total_cost = np.sum(distances)
        return total_cost
        
    result = minimize(objective_function, initial_guess)
    return torch.tensor(result.x)
    
def generate_unseen_poses(patch_size, img_wh, c2w, n_sparse_rays, bounding_form, mean_focus_point, write_poses=False, captured=False, sample_boundary=True):
    '''
    Generate a random pose following https://arxiv.org/pdf/2112.00724v1.pdf RegNerf from known poses c2w.
    Args:
        patch_size: int, length the patch
        img_wh: (h,w), image shape
        c2w: (n_images, 4, 4), c2w matrix
        n_sparse_rays: number of sparse rays that we are shooting
        bounding_form: str, sampling from a box or a sphere
        mean_focus_point: (3, ), the mean focus point of the existing cameras 
        use_sparse: bool, whether to generate sparse rays using the unseen_poses. 
        write_poses: bool, whether to write the poses to a json file for viz
        captured: bool, if set to true, the generated look-at matrix will have forward axis inverted
        sample_boundary: bool, if set to true, the unseen poses will only be sampled from the boundary
    Output:
        x_indices xx: (patch_size**2)
        y_indices yy: (patch_size**2)
        new_rotation_matrix: (3, 3)
        new_translation_vector: (3, 1)
    '''
    image_width, image_height = img_wh
    
    known_rotation_matrices = c2w[:, :3, :3].cpu() # (n_images, 3, 3)
    
    #1. First generate a random patch of size patch_size x patch_size
    #Randomly generating indices and forming a patch
    start_y = torch.randint(0, image_height - patch_size + 1, (1,)).item()
    start_x = torch.randint(0, image_width - patch_size + 1, (1,)).item()

    # Generate the coordinates of the patch
    rows = torch.arange(start_y, start_y + patch_size)
    cols = torch.arange(start_x, start_x + patch_size)

    # Use meshgrid to create a grid of coordinates
    xx, yy = torch.meshgrid(rows, cols, indexing='xy')
    xx = xx.flatten()
    yy = yy.flatten()
    
    # 2. Generate sparse image coordinates of shape (n_sparse_rays,1)
    if n_sparse_rays > 0:
        x_sparse = torch.randint(0, image_height, size=(n_sparse_rays,)) #(n_sparse_rays, )
        y_sparse = torch.randint(0, image_width, size=(n_sparse_rays,)) #(n_sparse_rays, )
    
        #Combine both rays
        xx, yy = torch.cat([xx, x_sparse]), torch.cat([yy, y_sparse]) #(patch_size**2 + n_sparse_rays, )

    # 3. Compute pu
    #Computes the average of the up vectors (second column) of the known rotation matrices
    local_up = torch.tensor([0., 1., 0.]) 
    global_up = known_rotation_matrices.float() @ local_up #batch matrix multiplication (n_images, 3) transform in the global coordinate system, basically just taking the second column of the known camera pose
    averaged_up = torch.mean(global_up, dim=0) #Average them across the n_images dimension
    p_u = averaged_up/torch.linalg.norm(averaged_up) #Normalize the vector
    
    # 4. Compute pf
    # Computes the point of smallest average distance to the optical axes of the known rotation matrices
    optical_axes = known_rotation_matrices[..., 2] #each of the rows is normalized
    known_camera_locations = c2w[:, :3, -1] # (n_images, 3)
    
    
    p_f = mean_focus_point + torch.normal(0, 0.125, size=(mean_focus_point.shape[0],)) #adding random jitter to the focal point
    
    # 5. Compute the new camera origin
    x_bounds = torch.min(known_camera_locations[:, 0]), torch.max(known_camera_locations[:, 0])
    y_bounds = torch.min(known_camera_locations[:, 1]), torch.max(known_camera_locations[:, 1])
    z_bounds = torch.min(known_camera_locations[:, 2]), torch.max(known_camera_locations[:, 2])
    if bounding_form == "box":
        new_location_x, new_location_y, new_location_z = sample_point_in_box(x_bounds, y_bounds, z_bounds, sample_boundary)
    elif bounding_form == "sphere":
        new_location_x, new_location_y, new_location_z = sample_point_in_sphere(mean_focus_point, known_camera_locations,sample_boundary)
    else:
        raise ValueError("Bounding form not recognized")
    
    unseen_camera_origin = torch.cat((new_location_x, new_location_y, new_location_z), dim=0)
    
    #5. Compute the new rotation matrix
    unseen_rotation_matrix = look_at(p_f, p_u, unseen_camera_origin, captured)
    
    #Concatenate the rotation matrix and the translation vector
    # Uncomment the following lines to visualize the camera locations
    if write_poses:
        #NOTE: make sure to change the paths and the frame 
        unseen_c2w = np.concatenate((unseen_rotation_matrix, unseen_camera_origin[:, None]), axis=1)
        unseen_c2w = np.concatenate((unseen_c2w, torch.tensor([0., 0., 0., 1.]).reshape(1, 4)), axis=0)
        unseen_c2w = unseen_c2w.tolist()
        #Open
        json_file_path = '/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/lego/lego_jsons/two_views/unseen_train.json'
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        new_frame = {
        "file_path": "./train/r_new",  
        "rotation": data["frames"][0]["rotation"],
        "transform_matrix": unseen_c2w 
        }
        data["frames"].append(new_frame)
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)
            

    return xx, yy, unseen_rotation_matrix, unseen_camera_origin



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
    
    
def find_surface_points(ray_origins, ray_dirs, opacities, depths, geometry_net, use_opacities=False, validation=False):
    # considered_indices = torch.where(opacities>0.5)[0]
            
    considered_indices = torch.arange(depths.shape[0])
    surface_points = ray_origins[considered_indices] + depths[considered_indices]*ray_dirs[considered_indices] #(n_rays, 3), 
    #o + t_sd, where t_s is the depth
    surface_points = torch.clip(surface_points, -1.49, 1.49) #keep the range between -1.49, 1.49
    sdf_surface, sdf_grad = geometry_net(surface_points, with_grad=True, with_feature=False, with_laplace=False)
    
    # Analytical gradients seem to have good visuals    
    normals = F.normalize(sdf_grad, p=2, dim=-1)
    normals = normals.view(-1, 3)
    normals_len = torch.linalg.norm(normals, dim=-1, keepdims=True)[..., 0] #computes the l2 norm of normals

    # surface_points = surface_points[normals_len>1e-9]
    # considered_indices = considered_indices[normals_len>1e-9]
    # normals = normals[normals_len>1e-9]
    
    normals[normals_len<1e-9] = -ray_dirs[normals_len<1e-9]
    
    # depths = depths[considered_indices]
    
    # normals = normals/(torch.max(normals, dim=1, keepdim=True)[0]+1e-10)
    # normals = normals / (torch.linalg.norm(normals, dim=-1, keepdims=True)+1e-10)
    if torch.isnan(normals).any():
        print("stop")
        
    
    # if use_opacities:
    #     considered_indices_surface = torch.where(opacities>0.5)[0]
    #     return surface_points, depths[considered_indices], considered_indices, normals, considered_indices_surface

    return surface_points, sdf_surface, depths[considered_indices], considered_indices, normals
    
# def compute_normals(depth_map, intrinsic):
#     '''Compute the normals from the depth map using camera intrinsics.'''
#     n, h, w = depth_map.shape
#     u, v = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    
#     fx, fy = intrinsic[0, 0], intrinsic[1, 1]
#     cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
#     z = torch.from_numpy(depth_map)
#     x = (u - cx) * z / fx
#     y = (v - cy) * z / fy
    
#     points = torch.stack([x, y, z], dim=-1)
#     padded = torch.nn.functional.pad(points, (0, 0, 1, 1, 1, 1), mode='replicate')
#     g_x = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2
#     g_y = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2
#     N = torch.cross(g_x, g_y, dim=-1)
#     N_norm = torch.linalg.norm(N, dim=-1, keepdim=True)
#     N_normalized = N / torch.where(N_norm == 0, torch.ones_like(N_norm), N_norm)
    
    return N_normalized


def get_rays_from_images(K, c2w):
    '''
    K is a 3x3 intrinsic matrix
    c2w is a (n,4x4) extrinsic matrix for each of the n images
    '''
    
    x, y = torch.meshgrid(
        torch.arange(512),
        torch.arange(512),
        indexing="xy",
    )

    x = x.flatten()
    y = y.flatten()

    dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1] * -1.0,
            ], dim=-1), (0, 1), value=-1.0)

    #Dimension for camera_dirs[:x.shape[0], None, None, :]
    #Dimension for c2w[:, :3, :3]
    directions = (dirs[:,None, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    directions = directions.permute(1,0,2)
    origins = torch.broadcast_to(c2w[:, None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (c2w.shape[0], 512, 512, 3))
    viewdirs = torch.reshape(viewdirs, (c2w.shape[0], 512, 512, 3))

    return origins, viewdirs



def get_rays_captured_from_images(K, c2w):
    
    x, y = torch.meshgrid(
        torch.arange(512),
        torch.arange(512),
        indexing="xy",
    )
    
    x = x.flatten()
    y = y.flatten()
    x = x.to(K.rays)
    y = y.to(K.rays)
    dirs = K(x, y) 
    
    directions = (dirs[:,None, None, :] * c2w[:, :3, :3].to(dirs)).sum(dim=-1)
    directions = directions.permute(1,0,2)
    origins = torch.broadcast_to(c2w[:, None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (c2w.shape[0], 512, 512, 3))
    viewdirs = torch.reshape(viewdirs, (c2w.shape[0], 512, 512, 3))
    return origins, viewdirs

def compute_normals(depth_map, K, c2w):
    '''Compute the normals from the depth map using camera intrinsics.'''
    n, h, w = depth_map.shape
    # Compute ray origins and directions
    origins, viewdirs = get_rays_from_images(K, c2w)
    # Reshape depth_map for broadcasting
    depth_map = torch.from_numpy(depth_map).reshape(n, h, w, 1)

    # Compute 3D points
    points = (depth_map * viewdirs + origins).reshape(n, h, w, 3)
    padded = torch.nn.functional.pad(points, (0, 0, 1, 1, 1, 1), mode='replicate')
    g_x = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2 #each of these terms is of shape (n, 512,512,3)
    g_y = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2
    N = torch.cross(g_x, g_y, dim=-1)
    N_norm = torch.linalg.norm(N, dim=-1, keepdim=True)
    N_normalized = N / torch.where(N_norm == 0, torch.ones_like(N_norm), N_norm)
    
    return N_normalized



def compute_normals_from_transient_captured(depth_map, K, c2w):
    '''Compute the normals from the depth map for the captured dataset'''
    n, h, w = depth_map.shape
    origins, viewdirs = get_rays_captured_from_images(K, c2w)
    
    depth_map = torch.from_numpy(depth_map).reshape(n, h, w, 1)

    # Compute 3D points
    points = (depth_map * viewdirs.to(origins) + origins).reshape(n, h, w, 3)
    padded = torch.nn.functional.pad(points, (0, 0, 1, 1, 1, 1), mode='replicate')
    g_x = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2 #each of these terms is of shape (n, 512,512,3)
    g_y = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2
    N = torch.cross(g_x, g_y, dim=-1)
    N_norm = torch.linalg.norm(N, dim=-1, keepdim=True)
    N_normalized = N / torch.where(N_norm == 0, torch.ones_like(N_norm), N_norm)
    
    return N_normalized