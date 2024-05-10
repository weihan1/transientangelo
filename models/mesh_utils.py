import argparse

import torch
import numpy as np

# pip install trimesh[all]
import trimesh
#Code from https://github.com/NVlabs/nvdiffrec/issues/71

# https://github.com/otaheri/chamfer_distance
from chamfer_distance import ChamferDistance

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")


def calculate_chamfer_distance(mesh_path, ref_mesh_path, n):
    '''
    Given two meshes, sample n points from each mesh and calculate the chamfer distance between the two sets of points.
    '''
    chamfer_dist = ChamferDistance()
    mesh = as_mesh(trimesh.load_mesh(mesh_path))
    ref = as_mesh(trimesh.load_mesh(ref_mesh_path))
    # Make sure l=1.0 maps to 1/10th of the AABB. https://arxiv.org/pdf/1612.00603.pdf
    scale = 10.0 / np.amax(np.amax(ref.vertices, axis=0) - np.amin(ref.vertices, axis=0))
    mesh.vertices = mesh.vertices * scale
    ref.vertices = ref.vertices * scale
    vpos_mesh = sample_mesh(mesh, n) #(n, 3)
    vpos_ref = sample_mesh(ref, n) #(n, 3)
    dist1, dist2,_,_ = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    return loss