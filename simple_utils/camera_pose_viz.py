import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import json


def visualize(camera_poses, focal, view_pose=None, view_img=None, lims=((-4, 4), (-4, 4), (0, 4)), all_poses=None):
    '''Generates and returns a figure that illustrates camera & sampling geometry

    Parameters
    ----------
    camera_poses : array of size [batch_size, 3, 4]
        contains the camera rotation and translation matrix for each camera to be plotted in the
        visualization
    focal : float
        focal length of cameras in world units (not pixel)
    view_pose : 3 x 4 matrix (optional)
        contains the rotation and translation wrt world coordinates to show the scene
    view_img : array of shape [Nx, Ny, 4] (optional)
        an image shown at the origin of the coordinate system from the perspective defined by view_pose
    lims : 3-tuple of ((-xlim, xlim), (-ylim, ylim), (-zlim, zlim)) 
    all_poses : array of size [num_camera_poses, 3, 4]
        if not None, plot all camera poses with current poses indicated
    '''

    # make compound plot
    if all_poses is not None:
        matplotlib.rcParams['figure.figsize'] = [3, 3]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    camera_poses = [camera_poses[i] for i in range(camera_poses.shape[0])]

    width = 0.1  # these are always fixed for our models
    height = 0.1

    for idx, camera_pose in enumerate(camera_poses):
        X_cam = create_camera_model(focal, focal, width, height)
        X_cam = [torch.Tensor(X) for X in X_cam]
        color = next(ax._get_lines.prop_cycler)['color']
        for i in range(len(X_cam)):
            X = np.zeros(X_cam[i].shape)
            for j in range(X_cam[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(camera_pose, X_cam[i][0:4, j])

            ax.plot3D(X[0, :], X[1, :], X[2, :], color=color, linewidth=1, zorder=20)
            
            d = torch.Tensor([0, 0, -1, 1])
            d = camera_pose @ d
            d = np.array([[camera_pose[0, -1], d[0]], [camera_pose[1, -1], d[1]], [camera_pose[2, -1], d[2]]])
            print(d.shape)
            ax.plot3D(d[0], d[1], d[2], color=color, linewidth=1, zorder=20)

        # plot up vector
        # if ray.shape[-1] > 6:
            # up0 = camera_pose[:3, 3]
            # print(up0.shape)
            # print(ray.shape)
            # print(ray[0, 0, 6:].shape)
            # up1 = ray[0, 0, 6:] + up0
            # up = torch.stack((up0, up1), dim=-1)
            # ax.plot3D(up[0,:], up[1,:], up[2,:], color=color, linewidth=1, zorder=20)

    if view_img is not None:
        # generate ray directions
        x = torch.linspace(-0.5, 0.5, view_img.shape[0]) / focal
        y = -torch.linspace(-0.5, 0.5, view_img.shape[1]) / focal
        X, Y = torch.meshgrid(x, y)
        Z = -torch.ones_like(X)

        # send rays out a distance equal to the camera distance from the origin
        dist = torch.sqrt(torch.sum(camera_poses[0][:, 3]**2))
        img_coords = torch.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=0)
        img_coords = camera_poses[0][:3, :3].matmul(img_coords).permute(1, 0)
        img_coords = camera_poses[0][:3, 3][None, :] + dist * img_coords

        # plot the image as a pointcloud at that point
        ax.scatter(img_coords[:, 0], img_coords[:, 1], img_coords[:, 2], c=view_img.reshape(-1, 4), zorder=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    set_axes_equal(ax)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    for i in range(len(xticks)-2):
        ax.plot3D([xticks[i+1], xticks[i+1]], [ylim[0], ylim[1]], [zlim[0], zlim[0]], 'k', alpha=0.2, zorder=20)
    for i in range(len(yticks)-2):
        ax.plot3D([xlim[0], xlim[1]], [yticks[i+1], yticks[i+1]], [zlim[0], zlim[0]], 'k', alpha=0.2, zorder=20)

    def pose_to_xyz(pose, dir='z', vector=False):
        x = torch.Tensor([1.]) if dir == 'x' else torch.Tensor([0.])
        y = torch.Tensor([1.]) if dir == 'y' else torch.Tensor([0.])
        z = torch.Tensor([-1.]) if dir == 'z' else torch.Tensor([0.])
        view_dir = pose[:3, :3].matmul(torch.stack((x, y, z), dim=0))
        x, y, z = view_dir[0], view_dir[1], view_dir[2]

        if vector:
            x0, y0, z0 = pose[:3, 3]
            x1 = x0 + x/2
            y1 = y0 + y/2
            z1 = z0 + z/2
            return np.hstack((x0, x1)), np.hstack((y0, y1)), np.hstack((z0, z1))
        else:
            return x, y, z

    if view_pose is not None:

        # or get viewing direction from the rotation
        x, y, z = pose_to_xyz(view_pose)
        el = torch.atan2(z, torch.sqrt(x**2 + y**2)) / np.pi * 180
        az = torch.atan2(y, x) / np.pi * 180
    ax.view_init(elev=60, azim=60)

    if all_poses is not None:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        all_poses = [all_poses[i] for i in range(all_poses.shape[0])]

        for pose in all_poses:
            x, y, z = pose_to_xyz(pose, vector=True)
            ax2.plot3D(x, y, z, color='black', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='y', vector=True)
            ax2.plot3D(x, y, z, color='black', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='x', vector=True)
            ax2.plot3D(x, y, z, color='black', linewidth=1)

        for pose in camera_poses:
            x, y, z = pose_to_xyz(pose, vector=True)
            ax2.plot3D(x, y, z, color='red', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='y', vector=True)
            ax2.plot3D(x, y, z, color='red', linewidth=1)
            x, y, z = pose_to_xyz(pose, dir='x', vector=True)
            ax2.plot3D(x, y, z, color='red', linewidth=1)
        set_axes_equal(ax2)

    return fig


# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# use the following functions from OpenCV example code
# https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py
def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = torch.eye(4)
    return M.matmul(cMo.matmul(X))


def create_camera_model(fx, fy, width, height, scale_focal=True, draw_frame_axis=False):
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, -f_scale]
    X_img_plane[0:3, 1] = [width, height, -f_scale]
    X_img_plane[0:3, 2] = [width, -height, -f_scale]
    X_img_plane[0:3, 3] = [-width, -height, -f_scale]
    X_img_plane[0:3, 4] = [-width, height, -f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, height, -f_scale]
    X_triangle[0:3, 1] = [0, 2*height, -f_scale]
    X_triangle[0:3, 2] = [width, height, -f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, -f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, -f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, -f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, -f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, -f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def get_camera_poses(path):
    with open(path, 'r') as f:
        meta = json.load(f)

    pose = []
    for frame in meta['frames']:
        pose.append(torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)))
    return torch.stack(pose)


def demo(path):
    # load camera poses from file
    poses = [get_camera_poses(x) for x in path]
    # poses = get_camera_poses(path) 
    poses = torch.concat(poses, 0)
    
    # poses = np.load('poses.npy', allow_pickle=True)
    tmp = []
    poses = poses[:]
    for p in poses:
        tmp.append(p[:3])
        # tmp.append(np.concatenate((np.array(p['R']), np.array(p['T'])[:, None]), 1))
    poses = torch.from_numpy(np.stack(tmp)).float()

    tmp = torch.zeros(poses.shape[0], 4, 4)
    tmp[:, :3, :] = poses
    tmp[:, -1, -1] = 1
    poses = tmp

    poses[:, :2] *= -1
    poses[:2, -1] *= -1

    # run visualization script
    visualize(poses, focal=2)
    plt.savefig("disp")


if __name__ == '__main__':
    paths = ["/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/load/transient_nerf_synthetic/lego/lego_jsons/two_views/unseen_train.json"]
    
    demo(paths)