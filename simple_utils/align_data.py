import numpy as np
import torch
import open3d as o3d
import skimage.morphology as morph
import matplotlib.pyplot as plt
from colmap_reader import read_images_text, read_points3D_text, read_cameras_text
import os
import plotly.graph_objs as go
import h5py
import scipy.io
from scipy.ndimage import correlate1d
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import nnls
from read_depth import read_array
from get_cameras import txt2json
import json
import torch.nn.functional as F
from transforms.so3 import so3_exp_map, so3_log_map
# from viz import visualize


K = torch.Tensor([[864.1862, 0, 221.0810], [0, 841.4213, 229.2477], [0, 0, 1.0000]])


def least_squares_scale_estimator(d, d_gt):
    d = d.view(-1)
    d_gt = d_gt.view(-1)

    # shift = torch.mean(d_gt - d)
    # scale = torch.Tensor([1.0])

    d = torch.stack((d, torch.ones_like(d)), dim=-1)
    out, _, _, _ = torch.linalg.lstsq(d, d_gt)


    # [d 1] * [scale; shift] = [d_gt]
    # out, res = nnls(d.numpy(), d_gt.numpy())
    # out = torch.from_numpy(out)

    scale = out[0]
    shift = out[1]

    d_shifted = d * scale + shift

    return scale, shift, d_shifted


def load_transient(fname):
        transient = torch.load(fname).to_dense()

        # transient = loadmat(os.path.join(transient_path, f'transient{i+1:03d}.mat'))["transients"]
        img = transient.sum(0).numpy()

        # transient = loadmat(os.path.join(transient_path, f'transient{i+1:03d}.mat'))["transients"]
        img = transient.sum(0).numpy()

        # threshold image and depth out
        th = 200
        morph_th = 12000
        mask = img > th
        mask = morph.area_opening(mask, morph_th)
        img = img*mask

        # threshold image and depth out
        pulse = scipy.io.loadmat('pulse.mat')['laser'].squeeze()
        pulse = np.log(pulse[np.argmax(pulse)-20:np.argmax(pulse)+20] + 1e-6)
        pulse = pulse / np.sum(pulse)

        # plt.plot(pulse)
        # plt.show()
        # exit()

        lm = correlate1d(transient, pulse, axis=0)
        # lm = transient
        depth = np.argmax(lm, axis=0)
        depth = (depth*299792458*4e-12)/2
        # depth = depth*mask

        # d = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        # image = o3d.geometry.Image(np.ascontiguousarray(img).astype(np.float32))
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image, d)
        # cam = o3d.camera.PinholeCameraIntrinsic()
        # cam.intrinsic_matrix = [[864.1862, 0, 221.0810], [0, 841.4213, 229.2477], [0, 0, 1.0000]]
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

        # K = np.array([[864.1862, 0, 221.0810], [0, 841.4213, 229.2477], [0, 0, 1.0000]])

        # x = np.arange(512)
        # cols, rows = np.meshgrid(x, x, indexing='xy')
        # rays = np.stack([(cols - K[0, 2])/K[0, 0],
        #                  (rows - K[1, 2])/K[1, 1],
        #                  np.ones((512, 512))], axis=2)
        # rays = rays / np.linalg.norm(rays, axis=2, keepdims=True)
        # rays = rays * depth[..., None] * mask[..., None]

        return fname, depth * mask


def dist2pc(dist):
    x = np.arange(512)
    cols, rows = np.meshgrid(x, x, indexing='xy')
    rays = np.stack([(cols - K.numpy()[0, 2])/K.numpy()[0, 0],
                    (rows - K.numpy()[1, 2])/K.numpy()[1, 1],
                    np.ones((512, 512))], axis=2)
    rays = rays / np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays * dist[..., None]
    rays = rays[dist != 0]
    return rays


def load_colmap_depth(folder, cam_id):
    depth = read_array(os.path.join(folder, 'stereo', 'depth_maps', f'{cam_id-1:03d}.png.geometric.bin'))
    return depth.astype(np.float32)

def load_colmap(folder, cam_id):
    # get colmap data
    # load 2D points
    meta_2d = read_images_text(os.path.join(f'{folder}', 'images.txt'))[cam_id]
    xy = meta_2d.xys
    point_ids = meta_2d.point3D_ids
    tvec = meta_2d.tvec

    mask = point_ids != -1
    xy = xy[mask]
    point_ids = point_ids[mask]

    # load 3d points
    meta_3d = read_points3D_text(os.path.join(f'{folder}', 'points3D.txt'))

    dist = []
    for pid in point_ids:
        dist.append(np.linalg.norm(np.array(meta_3d[pid].xyz - tvec)))
    colmap_dist = np.array(dist)

    # bin them into an image
    dist = np.zeros((512, 512))
    xy = np.round(xy).astype(np.int)
    for i in range(xy.shape[0]):
        dist[xy[i, 1], xy[i, 0]] = colmap_dist[i]

    return dist


def get_lidar_pointclouds():
    dists = Parallel(n_jobs=8)(delayed(load_transient)(f'./cinema_aligned_undistorted_pt/transient{cam_id:03d}.pt') for cam_id in tqdm(range(1, 91)))
    # dists = np.load('transient_dists.npy', allow_pickle=True)

    np.save('transient_dists.npy', dists)


def plot_sliders():
    colmap_dists = []
    for cam_id in range(90):
        tmp = load_colmap_depth('model_dense', cam_id+1)
        mask = tmp > 0
        colmap_dists.append(tmp*mask)

    # Add traces, one for each slider step
    x = np.arange(512)
    cols, rows = np.meshgrid(x, x, indexing='xy')

    fig = go.Figure()
    print(len(colmap_dists))

    for step in range(90):
        mask = colmap_dists[step] > 0
        trace1 = go.Scatter3d(x=cols[mask], y=rows[mask], z=colmap_dists[step][mask], mode='markers',
                             marker=dict(
                             size=2,
                             color='red'), visible=False)

        fig.add_trace(trace1)

    # Make 0th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 90},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders

    )
    fig.update_layout(
        scene_camera=dict(up=dict(x=0., y=-1, z=0),
                          eye=dict(x=1, y=-1, z=-1)),
        scene=dict(
            xaxis=dict(range=(128, 512-128)),
            yaxis=dict(range=(128, 512-128)),
            zaxis=dict(range=(3, 5))
            ),
    )

    fig.show()


def plot_pcs(pcs):

    fig = go.Figure()
    for pc, color in zip(pcs, ['blue', 'red']):

        trace1 = go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=1, color=pc[:, 2], colorscale='Viridis'))
        fig.add_trace(trace1)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-3, 3]),  # Adjust these ranges based on your data
            yaxis=dict(range=[-3, 3]),
            zaxis=dict(range=[-3, 3]),
            camera=dict(
                up=dict(x=0, y=0, z=1),  # Z-axis is up
                eye=dict(x=1.25, y=1.25, z=1.25)  # Perspective view
            )
        ),
        scene_camera_projection=dict(type="perspective")
    )

    # fig.write_image("/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/simple_utils/depth.png")
    fig.show()

def load_dists():

    dists = np.load('transient_dists.npy', allow_pickle=True)
    dists = sorted(dists, key=lambda x: x[0])
    dists = [t[1].astype(np.float32) for t in dists]

    colmap_dists = []
    transient_dists = []

    for cam_id in range(1, 91):
        transient_dists.append(dists[cam_id-1])
        colmap_dists.append(load_colmap_depth('model_dense', cam_id))
        colmap_dists[-1] = depth2dist(colmap_dists[-1], K).numpy()
        # colmap_dists.append(load_colmap('model3', cam_id))

    transient_dists = torch.from_numpy(np.array(transient_dists))
    colmap_dists = torch.from_numpy(np.array(colmap_dists))

    return transient_dists, colmap_dists


def read_json_poses(fname):
    with open(
            os.path.join(fname), "r"
    ) as fp:
        meta = json.load(fp)
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        camtoworlds.append(frame["transform_matrix"])

    camtoworlds = np.stack(camtoworlds, axis=0)

    return camtoworlds


def get_rays(K, c2w, camera_rays=False):
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

    if camera_rays:
        return dirs.reshape(512, 512, 3)

    directions = (dirs[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (512, 512, 3))
    viewdirs = torch.reshape(viewdirs, (512, 512, 3))

    return origins, viewdirs


def get_rays_shaped(K, c2w,  img_shape, camera_rays=False,):
    x, y = torch.meshgrid(
        torch.arange(img_shape[0]),
        torch.arange(img_shape[1]),
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

    if camera_rays:
        return dirs.reshape(img_shape[0], img_shape[1], 3)

    directions = (dirs[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (img_shape[0], img_shape[1], 3))
    viewdirs = torch.reshape(viewdirs, (img_shape[0], img_shape[1], 3))

    return origins, viewdirs


def depth2dist(depth, K):
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

    directions = dirs
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True)
    viewdirs = viewdirs.reshape(512, 512, 3)

    add_dim = False
    if depth.ndim == 2:
        depth = depth[..., None]
        add_dim = True

    dist = depth / torch.abs(viewdirs[..., [-1]])

    if add_dim:
        dist = dist.squeeze()

    return dist

def get_rays_simulated_camera_space(K, c2w, img_shape, camera_rays=False):
    x, y = torch.meshgrid(
        torch.arange(img_shape[0]),
        torch.arange(img_shape[1]),
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

    if camera_rays:
        return dirs.reshape(img_shape[0], img_shape[1], 3)

    directions = (dirs[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (img_shape[0], img_shape[1], 3))
    viewdirs = torch.reshape(viewdirs, (img_shape[0], img_shape[1], 3))

    return origins, viewdirs



def compare_pointclouds(scale_shift=None):

    # convert cameras to json from colmap
    # txt2json('model3/images.txt')
    # exit()

    # get scale and shift
    if not scale_shift:
        scale, shift = solve_for_scale_and_shift()
        print(scale, shift)
        f = np.load('optimized_scale_shift.npy', allow_pickle=True)[()]
        scale = f['scale']
        shift = f['shift']
    else:
        scale, shift = scale_shift

    # get the ray distances
    transient_dists, colmap_dists = load_dists()
    transient_dists = transient_dists[..., None]
    colmap_dists = colmap_dists[..., None]

    # flip coordinates?
    # print(transient_dists.shape)
    # transient_dists = torch.permute(transient_dists, [0, 2, 1, 3])
    # colmap_dists = torch.permute(colmap_dists, [0, 2, 1, 3])
    # colmap_dists = torch.flip(colmap_dists, [1, ])

    # json_file = 'model_0.33_flatter/images.json'
    # json_file = 'model3/images.json'
    json_file = 'optimized_cams.json'
    camtoworlds = torch.from_numpy(read_json_poses(json_file))
    # K = torch.Tensor(np.array([[839.8794, 0, 287.2693], [0, 825.6033, 215.5942], [0, 0, 1.0000]]))

    # camera id
    camid = 80
    colmap_pcs = []
    transient_pcs = []

    for camid in range(1, 90, 5):

        # scale up extrinics
        alpha = 0.1
        camtoworlds[camid][:3, -1] *= alpha

        # get rays
        origins, viewdirs = get_rays(K, camtoworlds[camid])

        # convert colmap depth to dist
        # Note: (this is done when loading now, comment out)
        # colmap_dists[camid] = depth2dist(colmap_dists[camid], K)

        # get point cloud for colmap and transient
        colmap_pc = (colmap_dists[camid] * viewdirs*alpha + origins).reshape(-1, 3)
        transient_pc = alpha * ((transient_dists[camid]-shift)/scale * viewdirs*alpha + origins).reshape(-1, 3)

        # get masks
        transient_mask = (transient_dists[camid] > 0).flatten()
        colmap_mask = (colmap_dists[camid] > 0).flatten()
        colmap_pcs.append(colmap_pc[colmap_mask, :])
        transient_pcs.append(transient_pc[transient_mask, :])

    colmap_pc = torch.concatenate(colmap_pcs, dim=0)
    transient_pc = torch.concatenate(transient_pcs, dim=0)

    # plot point clouds
    # plot_pcs([transient_pc, colmap_pc])
    plot_pcs([transient_pc, ])



def compare_pointclouds_transients(scale_shift=None):
    device = "cuda:1"
    cams_to_viz = np.array([1, 11]) # indexed with 0 
    alpha = 0.35 
    json_path  = "/home/anagh/PycharmProjects/multiview_transient/datasets/cinema_aligned_undistorted_pt/transforms.json"
    transients_path = "/home/anagh/PycharmProjects/multiview_transient/datasets/captured_data/cinema_aligned"
    camtoworlds = torch.from_numpy(read_json_poses(json_path))
    camtoworlds = camtoworlds[cams_to_viz]
    camtoworlds[:, :3, 3] = camtoworlds[:, :3, 3]*alpha
    K = torch.Tensor(np.array([[864.1862, 0, 221.0810], [0, 841.4213, 229.2477], [0, 0, 1.0000]]))/2


    scale = 0.2797779142856598/alpha
    shift = -0.0005884504644200206
    exposure_time = 299792458*4e-12

    x = (torch.arange(512, device=device)-256+0.5)/(256-0.5)
    y = (torch.arange(512, device=device)-256+0.5)/(256-0.5)
    z = torch.arange(3000, device=device).float()
    X, Y, Z = torch.meshgrid(x, y, z, indexing="xy")
    Z = Z*exposure_time/2
    Z = scale*Z + shift 
    Z = Z*2/exposure_time
    Z = (Z-1500+0.5)/(1500-0.5)
    grid = torch.stack((Z, X, Y), dim=-1)[None, ...]

    images = []
    for number in cams_to_viz:
        fname = os.path.join(transients_path, f"masked_transient{number+1:03d}" + ".pt")
        rgba = torch.load(fname).to_dense()
        rgba = torch.Tensor(rgba)[:3000].permute(1, 2, 0).to(device)     
        rgba = torch.nn.functional.grid_sample(rgba[None, None, ...], grid, align_corners=True).squeeze().cpu()
        rgba = torch.clip(rgba, 0, None)
        rgba = (rgba[::2, ::2] + rgba[::2, 1::2] +  rgba[1::2, ::2]+ rgba[1::2, +1::2] )/4

        images.append(rgba)
    
    images = torch.stack(images, 0)


    # camera id
    camid = 80
    colmap_pcs = []
    transient_pcs = []

    for camid, _ in enumerate(cams_to_viz):


        # get rays
        origins, viewdirs = get_rays_shaped(K, camtoworlds[camid], img_shape=(256, 256))

        transient_dist = torch.argmax(images[camid], -1)
        transient_dist = transient_dist*exposure_time/2


        # get point cloud for colmap and transient
        transient_pc = (transient_dist[..., None] * viewdirs + origins).reshape(-1, 3)

        # get masks
        transient_mask = (transient_dist > 0).flatten()
        transient_pcs.append(transient_pc[transient_mask, :])

    transient_pc = torch.concat(transient_pcs, dim=0)

    plot_pcs([transient_pc, ])



def solve_for_scale_and_shift():
    dists = np.load('transient_dists.npy', allow_pickle=True)
    dists = sorted(dists, key=lambda x:x[0])
    dists = [t[1].astype(np.float32) for t in dists]

    colmap_dists = []
    transient_dists = []

    # solve for scale and shift
    for cam_id in range(1, 91):
        transient_dists.append(dists[cam_id-1])
        colmap_dists.append(load_colmap_depth('model_dense', cam_id))
        # colmap_dists.append(load_colmap('model3', cam_id))

    transient_dists = np.array(transient_dists)
    colmap_dists = np.array(colmap_dists)

    # make array of distances (ignoring bad values from colmap
    mask = (colmap_dists > 0) & (transient_dists > 0)
    d = torch.from_numpy(colmap_dists[mask])
    d_gt = torch.from_numpy(transient_dists[mask])

    scale, shift, d_shifted = least_squares_scale_estimator(d, d_gt)
    scale = scale.numpy()
    shift = shift.numpy()

    # make array of distances (ignoring bad values from colmap
    for i in range(0):
        transient_dists_tmp = (transient_dists - shift) / scale

        mask = (colmap_dists > 0) & (transient_dists > 0)
        d = torch.from_numpy(colmap_dists[mask])
        d_gt = torch.from_numpy(transient_dists_tmp[mask])

        scale2, shift2, d_shifted = least_squares_scale_estimator(d, d_gt)

        scale2 = scale2.numpy()
        shift2 = shift2.numpy()

        scale = scale*scale2
        shift = shift + scale*shift2

    return scale, shift


def load_colmap_3d_points(folder, device='cpu'):
    meta_2d = read_images_text(os.path.join(f'{folder}', 'images.txt'))
    meta_3d = read_points3D_text(os.path.join(f'{folder}', 'points3D.txt'))

    # get the ray distances
    transient_view_dists, colmap_view_dists = load_dists()
    transient_view_dists = transient_view_dists.to(device)
    colmap_view_dists = colmap_view_dists.to(device)

    # load the rays in the camera coordinate system
    cam_viewdirs = get_rays(K, None, camera_rays=True).to(device)

    # load the extrinsics
    json_file = f'{folder}/images.json'
    camtoworlds = torch.from_numpy(read_json_poses(json_file))

    # for all points
    # colmap_dists: [N_points, max_views, distances]
    # transient_dists: [N_points, max_views, distances]
    # viewdirs: [N_points, max_views, 3]
    # origins: [N_points, max_views, 3]
    # c2ws: [N_points, max_views, 3, 3]
    # shift_offset
    # scale_offset

    error_threshold = 0.05
    max_matches = 30
    N_points = len(meta_3d)
    device = torch.device(device)

    colmap_dists = torch.zeros(N_points, max_matches, device=device)
    transient_dists = torch.zeros(N_points, max_matches, device=device)
    viewdirs = torch.zeros(N_points, max_matches, 3, device=device)
    c2ws = torch.zeros(N_points, max_matches, 4, 4, device=device)
    valid = torch.zeros(N_points, max_matches, device=device).bool()

    for idx, (k, pt) in enumerate(meta_3d.items()):

        # check if point is under error threshold
        if pt.error > error_threshold:
            continue

        # get image coordinates
        image_coords = torch.from_numpy(np.array([meta_2d[view_id].xys[pt2d_id]
                                                for view_id, pt2d_id in zip(pt.image_ids, pt.point2D_idxs)]).astype(np.float32)).to(device)
        image_coords = torch.round(image_coords).long()
        num_matches = image_coords.shape[0]

        """
        image_idxs = torch.from_numpy(pt.image_ids).to(device) - 1
        for match_idx, image_idx, image_coord in zip(range(num_matches), image_idxs, image_coords):

            # get distances
            colmap_dists[idx, match_idx] = colmap_view_dists[image_idx, image_coord[1], image_coord[0]]
            transient_dists[idx, match_idx] = transient_view_dists[image_idx, image_coord[1], image_coord[0]]

            # get viewdirs in camera coordinate system
            viewdirs[idx, match_idx, :] = cam_viewdirs[image_coord[1], image_coord[0], :]

            # get extrinsics
            c2ws[idx, match_idx, ...] = camtoworlds[image_idx, :, :]
        """

        # get distances
        colmap_dists[idx, :num_matches] = colmap_view_dists[pt.image_ids-1, image_coords[:, 1], image_coords[:, 0]]
        transient_dists[idx, :num_matches] = transient_view_dists[pt.image_ids-1, image_coords[:, 1], image_coords[:, 0]]

        # get viewdirs in camera coordinate system
        viewdirs[idx, :num_matches, :] = cam_viewdirs[image_coords[:, 1], image_coords[:, 0], :]

        # get extrinsics
        c2ws[idx, :num_matches, ...] = camtoworlds[pt.image_ids-1, :, :]

        # flag the valid points
        valid[idx, :num_matches] = True

    # trim out any empty rows (caused by matches below error threshold)
    mask = torch.sum(valid, dim=1) > 0

    colmap_dists = colmap_dists[mask]
    transient_dists = transient_dists[mask]
    viewdirs = viewdirs[mask]
    c2ws = c2ws[mask]
    valid = valid[mask]

    return {'colmap_dists': colmap_dists,
            'transient_dists': transient_dists,
            'viewdirs': viewdirs,
            'c2ws': c2ws,
            'valid': valid}


class MipLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(MipLRDecay, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp]


def register_pointclouds(num_iters=500, lr_init=1e-1, lr_final=1e-3, device='mps', opt_extrinsics=True):

    # get a list of corresponding 3d points, their camera views, pixel indices, and distances
    point_dict = load_colmap_3d_points('model3', device=device)

    # some shorthand
    colmap_dists = point_dict['colmap_dists']
    transient_dists = point_dict['transient_dists']
    viewdirs = point_dict['viewdirs']
    c2ws = point_dict['c2ws']
    valid = point_dict['valid']

    # get points in corresponding camera views between colmap and lidar
    per_view_transient_dists, per_view_colmap_dists = load_dists()
    mask = (per_view_colmap_dists > 0) & (per_view_transient_dists > 0)

    if not opt_extrinsics:
        per_view_colmap_dists = per_view_colmap_dists[mask].to(device)
        per_view_transient_dists = per_view_transient_dists[mask].to(device)
        mask = mask.to(device)
    else:
        max_valid_per_view = torch.max(torch.sum(mask, dim=(1, 2)))
        tmp_per_view_colmap_dists = torch.zeros(per_view_transient_dists.shape[0], max_valid_per_view)
        tmp_per_view_transient_dists = torch.zeros(per_view_transient_dists.shape[0], max_valid_per_view)
        tmp_cam_viewdirs = torch.zeros(per_view_transient_dists.shape[0], max_valid_per_view, 3)

        cam_viewdirs = get_rays(K, None, camera_rays=True).to(device)

        for i in range(per_view_transient_dists.shape[0]):
            tmp_per_view_colmap_dists[i, :mask[i].sum()] = per_view_colmap_dists[i][mask[i]]
            tmp_per_view_transient_dists[i, :mask[i].sum()] = per_view_transient_dists[i][mask[i]]
            tmp_cam_viewdirs[i, :mask[i].sum(), :] = cam_viewdirs[mask[i], :]

        cam_viewdirs = tmp_cam_viewdirs.to(device)
        cam_scale = 1 / torch.linalg.norm(cam_viewdirs, dim=-1, keepdims=True)
        cam_scale[torch.isinf(cam_scale)] = 1.

        per_view_colmap_dists = tmp_per_view_colmap_dists.to(device)
        per_view_transient_dists = tmp_per_view_transient_dists.to(device)
        mask = mask.to(device)

    json_file = 'model3/images.json'
    camtoworlds = torch.from_numpy(read_json_poses(json_file)).float().to(device)


    # optimization loop
    # compute xyz position as a function of ray, camera coordinates, distances, initial scale and shift + offsets
    # calculate error between matching points

    # initial scale and shift
    # scale, shift = solve_for_scale_and_shift()
    # scale = torch.from_numpy(scale)
    # shift = torch.from_numpy(shift)

    # optimize these offsets
    scale = torch.nn.Parameter(torch.ones(1, device=device))
    shift = torch.nn.Parameter(torch.zeros(1, device=device))
    params = [scale, shift]

    # get ray origins and directions in global coordinate system
    directions = (viewdirs[..., None, :] * c2ws[..., :3, :3]).sum(dim=-1)
    origins = c2ws[..., :3, -1]
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
    directions[torch.isnan(directions)] = 0.

    if opt_extrinsics:
        log_rot_init = so3_log_map(camtoworlds[..., :3, :3])
        log_rot = torch.nn.Parameter(so3_log_map(camtoworlds[..., :3, :3]))
        t_init = camtoworlds[..., :3, -1]
        t = torch.nn.Parameter(camtoworlds[..., :3, -1])

        params1 = [scale, shift]
        params2 = [log_rot, t]
        optim = torch.optim.Adam(params=[{'params': params1, 'lr': 1e-1}, {'params': params2, 'lr': 1}], lr=lr_init)

    else:
        optim = torch.optim.Adam(params=params, lr=lr_init)

    scheduler = MipLRDecay(optim, lr_init, lr_final, max_steps=num_iters, lr_delay_steps=200)


    for i in range(num_iters):

        # compute the xyz coordinates
        # colmap_pc = (colmap_dists[..., None] * directions + origins)

        if opt_extrinsics:

            directions = cam_scale * (cam_viewdirs[..., None, :] * so3_exp_map(log_rot)[:, None, ...]).sum(dim=-1)
            transient_pc = (per_view_transient_dists[..., None] - shift) / scale * directions + t[:, None, :]
            colmap_pc = per_view_colmap_dists[..., None] * directions.detach() + t[:, None, :].detach()

            # directions_init = cam_scale * (cam_viewdirs[..., None, :] * so3_exp_map(log_rot_init)[:, None, ...]).sum(dim=-1)
            # colmap_pc = per_view_colmap_dists[..., None] * directions_init + t_init[:, None, :]

            optim.zero_grad()
            # loss = torch.mean((per_view_colmap_dists - (per_view_transient_dists - (shift))/(scale))**2)
            loss = torch.mean((transient_pc - colmap_pc)**2)

        else:
            transient_pc = (transient_dists[..., None]-(shift))/(scale) * directions + origins

            # make penalty function such that all 3d points should be close together (minimize fro norm of distance matrices?)
            optim.zero_grad()

            loss = 0.01 * torch.mean(torch.abs((transient_pc[..., :, None, :] - transient_pc[..., None, :, :])))
            loss = loss + torch.mean((per_view_colmap_dists - (per_view_transient_dists - (shift))/(scale))**2)

        loss.backward()
        optim.step()
        # scheduler.step()

        print(f'{i:03d}: {loss.item():03e} | Scale: {scale.item():.04e} | Shift: {shift.item():.04e}')

    compare_pointclouds(scale_shift=(scale.item(), shift.item()))

    # save out new extrinsics
    R = so3_exp_map(log_rot).detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    c2w = np.zeros((90, 4, 4))
    c2w[:, :3, :3] = R
    c2w[:, :3, -1] = t
    c2w[:, -1, -1] = 1

    np.save('optimized_scale_shift.npy', {'scale': scale.cpu().item(), 'shift': shift.cpu().item()})

    print(camtoworlds[0][:3, :3].cpu().numpy() - c2w[0][:3, :3])
    print(camtoworlds[0][:3, -1].cpu().numpy() - t[0][:])

    write_json('optimized_cams.json', c2w)
    
    if False:
        # run visualization script
        fig = plt.figure(1)
        visualize_cams('model_0.33_flatter/images.json', fig=fig)
        plt.title('Before optimization')

        fig = plt.figure(2)
        visualize_cams('optimized_cams.json', fig=fig)
        plt.title('After optimization')
        plt.show()
    return


def visualize_cams(fname, fig=None):

    # visualize cameras
    def get_camera_poses():
        with open(fname, 'r') as f:
            meta = json.load(f)

        pose = []
        for frame in meta['frames']:
            pose.append(torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)))
        return torch.stack(pose)

    poses = get_camera_poses()
    visualize(poses, focal=8)


def write_json(outfile, c2w):
    out = {"camera_angle_x": 0.690976, "frames": []} 

    for i, mat in enumerate(c2w):
        frame = {"file_path":f"./train/r_{i}", "rotation": 0.012566370614359171, "transform_matrix": mat.tolist()}
        out["frames"].append(frame)

    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)

    return

if __name__ == '__main__':

    # register_pointclouds()
    # exit()

    compare_pointclouds_transients()
    exit()

    # plot_sliders()
    # exit()

    # get_lidar_pointclouds()
    # exit()

    dists = np.load('transient_dists.npy', allow_pickle=True)
    dists = sorted(dists, key=lambda x:x[0])
    dists = [t[1].astype(np.float32) for t in dists]

    colmap_dists = []
    transient_dists = []

    # solve for scale and shift
    for cam_id in range(1, 91):
        transient_dists.append(dists[cam_id-1])
        colmap_dists.append(load_colmap_depth('model_dense', cam_id))

    transient_dists = np.array(transient_dists)
    colmap_dists = np.array(colmap_dists)

    # make array of distances (ignoring bad values from colmap
    mask = (colmap_dists > 0) & (transient_dists > 0)
    d = torch.from_numpy(colmap_dists[mask])
    d_gt = torch.from_numpy(transient_dists[mask])

    scale, shift, d_shifted = least_squares_scale_estimator(d, d_gt)
    scale = scale.numpy()
    shift = shift.numpy()

    print(scale)
    print(shift)


    # plot pointclouds of both and compare
    # transient_xyz = dist2pc(transient_dist)
    # colmap_xyz = dist2pc(colmap_dist)

    # plot
    plotting_cam =9
    colmap_dist = colmap_dists[plotting_cam]
    transient_dist = transient_dists[plotting_cam]
    x = np.arange(512)
    cols, rows = np.meshgrid(x, x, indexing='xy')
    mask = (colmap_dist > 0) & (transient_dist > 0)
    mask = transient_dist > 0
    # mask = np.ones((512, 512)) > 0

    # scale * colmap + shift = lidar 
    # scale * colmap = lidar - shift
    trace1 = go.Scatter3d(x=cols[mask], y=rows[mask], z=(transient_dist[mask]-shift), mode='markers',
                         marker=dict(
                         size=2,
                         color='blue'))

    mask = colmap_dist > 0
    trace2 = go.Scatter3d(x=cols[mask], y=rows[mask], z=colmap_dist[mask]*scale, mode='markers',
                         marker=dict(
                         size=2,
                         color='red'))


    if False:
        trace1 = go.Scatter3d(x=transient_xyz[:, 0], y=transient_xyz[:, 1], z=transient_xyz[:, 2], mode='markers',
                             marker=dict(
                             size=2,
                             color=transient_xyz[:, 2],  # set color to an array/list of desired values
                             colorscale='Viridis'))

        trace2 = go.Scatter3d(x=colmap_xyz[:, 0], y=colmap_xyz[:, 1], z=colmap_xyz[:, 2], mode='markers',
                             marker=dict(
                             size=2,
                             color='red'))

    layout = go.Layout(title='3D Scatter plot')
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    fig.update_layout(
        scene_camera=dict(up=dict(x=0., y=-1, z=0),
                          eye=dict(x=1, y=-1, z=-1)),
    )
    fig.layout.scene.camera.projection.type = "orthographic"

    fig.write_image("disp.png")
    exit()

    # px.scatter_3(transient_xyz)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(transient_xyz[:, 0], transient_xyz[:, 1], transient_xyz[:, 2], color='k')
    ax.scatter(colmap_xyz[:, 0], colmap_xyz[:, 1], colmap_xyz[:, 2], color='r')
    plt.savefig("diso")



    # also, should correct the depth estimation to use log matched filtering



    # plt.subplot(121)
    # plt.imshow(transient_dist)
    # plt.subplot(122)
    # plt.imshow(colmap_dist)
    # plt.show()



    # plt.subplot(131)
    # plt.imshow(pcd[:, :, 0])
    # plt.subplot(132)
    # plt.imshow(pcd[:, :, 1])
    # plt.subplot(133)
    # plt.imshow(pcd[:, :, 2])
    # plt.show()