'''
Author: error: git config user.name & please set dead value or install git && bigcileng@outlook.com
Date: 2023-05-25 20:02:53
LastEditors: error: git config user.name & please set dead value or install git && bigcileng@outlook.com
LastEditTime: 2023-05-25 23:08:20
FilePath: /NeuS/renderer.py
Description: 
    use function def validate_mesh(**kwargs) to generate mesh file
Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
'''
import torch
import mcubes
import trimesh
import numpy as np
import os
import argparse
from models.fields import SDFNetwork

def extract_fields(bound_min, bound_max, resolution=512, query_func=None):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


'''
description: generate mesh from SDF modal
param {*} bound_min: coordinates of the lower left corner of bounding box
param {*} bound_max: coordinates of the upper right corner of bounding box
param {*} mesh_name: the name of mesh
param {*} resolution: sample N points on each axis of the bounding box 
param {*} threshold: judge the surface
param {*} base_dir
param {*} checkpoint_dir
return {*}: file mesh.ply
'''
def validate_mesh(bound_min = None, bound_max = None, mesh_name = '', resolution=512, threshold=0.0, base_dir = '', checkpoint_dir = "exp/thin_catbus/womask_sphere/checkpoints/ckpt_300000.pth"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    sdfmodel = SDFNetwork(d_out = 257,
                            d_in = 3,
                            d_hidden = 256,
                            n_layers = 8,
                            skip_in = [4],
                            multires = 6,
                            bias = 0.5,
                            scale = 1.0,
                            geometric_init = True,
                            weight_norm = True
                            )
    checkpoint = torch.load(os.path.join(checkpoint_dir), map_location=torch.device('cuda'))
    sdf_f = checkpoint['sdf_network_fine']
    sdfmodel.load_state_dict(sdf_f)
    query_func = lambda pts: -sdfmodel.sdf(pts)

    bound_min = torch.ones(3, dtype=torch.float32) * (-1.01)
    bound_max = torch.ones(3, dtype=torch.float32) * (1.01)

    # bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
    # bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
    vertices, triangles =\
        extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=query_func)
    os.makedirs(os.path.join(base_dir), exist_ok=True)

    # if world_space:
    #     vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

    mesh = trimesh.Trimesh(vertices, triangles)
    filedir = base_dir + mesh_name + '.ply'
    
    mesh.export(filedir)


# example
if __name__ == "__main__":
    kwargs = {
        "bound_min": torch.tensor([-1.01, -1.01, -1.01]), 
        "bound_max": torch.tensor([1.01, 1.01, 1.01]), 
        "mesh_name": 'thin_catbus', 
        "resolution": 512, 
        "threshold": 0.0, 
        "base_dir": 'exp/meshes/', 
        "checkpoint_dir": "exp/thin_catbus/womask_sphere/checkpoints/ckpt_300000.pth"
    }
    validate_mesh(**kwargs)