import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json

def get_rays(K, c2w):
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

    directions = (dirs[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (512, 512, 3))
    viewdirs = torch.reshape(viewdirs, (512, 512, 3))

    return origins, viewdirs



def compute_normals(depth_map, K, c2w):
    '''Compute the normals from the depth map using camera intrinsics.'''
    n, h, w = depth_map.shape
    # Compute ray origins and directions
    origins, viewdirs = get_rays(K, c2w)
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


if __name__ == "__main__":
    json_file = '/scratch/ondemand28/weihanluo/transientangelo/load/transient_nerf_synthetic/lego/lego_jsons/ten_views/transforms_train.json'
    with open(json_file) as f:
        meta = json.load(f)
    c2w_first_frame = torch.from_numpy(np.array(meta["frames"][1]["transform_matrix"]))
    w,h = 512,512
    camera_angle = 0.691111147403717
    focal = 0.5 * w / math.tan(0.5 * camera_angle)
    K = torch.tensor(
            [
                [focal, 0, w / 2.0],
                [0, focal, h / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
    depth_map = np.ones((1,512,512))
    normals = compute_normals(depth_map, K, c2w_first_frame)
    plt.imshow(normals[0])
    plt.savefig("normals_ones.png")