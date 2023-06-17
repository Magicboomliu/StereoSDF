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

from PIL import Image
import cv2
from torchvision import transforms
import torch.nn.functional as F

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

def visualize_depth(disp, filedir='test',fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    height, width = disp.shape
    fx = fx_unit * width
    fy = fy_unit * height

    depths = fx * baseline / disp  # convert disparity to depth

    depths = np.nan_to_num(depths) # change nan to 0
    mi = np.min(depths) # get minimum depth
    ma = np.max(depths)
    depths_norm = (depths-mi)/(ma-mi+1e-8) # normalize to 0~1
    depths_color = (255*depths_norm).astype(np.uint8)
    depths_photo = Image.fromarray(cv2.applyColorMap(depths_color, cv2.COLORMAP_JET))
    filedir += '.jpg'
    depths_photo.save(filedir)

def visualize_mesh(disp, est_sdf, filedir='test', resolution=512, threshold=0.0, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    batch_num, depth_num, height, width = est_sdf.shape
    fx = fx_unit * width
    fy = fy_unit * height

    disp = disp[0].squeeze(0).detach().cpu().numpy()
    visualize_depth(disp, filedir, fx_unit, fy_unit, baseline)

    est_sdf = est_sdf[0]

    u = est_sdf.detach().cpu().numpy()
    # hypo_depths = torch.linspace(0, depth_num-1, depth_num).type_as(est_sdf)
    
    # hypo_depths = (fx * baseline / hypo_depths)  # convert disparity to depth
    
    
    # hypo_depths[0] = hypo_depths[1] # inf
    # hypo_depths = torch.flip(hypo_depths, dims=[0])
    # est_sdf = torch.flip(est_sdf, dims=[0])
    # unit = (torch.max(hypo_depths) - torch.min(hypo_depths)) / resolution

    # u = np.zeros([resolution, height, width], dtype=np.float32)

    # base = 0
    # for i in range(1, depth_num):
    #     N = ((hypo_depths[i] - hypo_depths[i-1]) / unit).int()
    #     sdf1 = est_sdf[i-1].flatten().unsqueeze(0).T
    #     sdf2 = est_sdf[i].flatten().unsqueeze(0).T
    #     sdf_inter_tmp = torch.cat([sdf1, sdf2], dim=-1)
    #     if N < 2:
    #         N = 1
    #         sdf_inter = torch.mean(sdf_inter_tmp, dim=1)
    #         sdf_inter = sdf_inter.reshape([height, width])
    #         if base < resolution:
    #             u[base] = sdf_inter.detach().cpu().numpy()
    #     else:
    #         sdf_inter_tmp = sdf_inter_tmp.unsqueeze(0).unsqueeze(0)
    #         sdf_inter = F.interpolate(sdf_inter_tmp, size=[height * width, N], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    #         sdf_inter = sdf_inter.reshape([N, height, width])
    #         if base + N > resolution:
    #             u[base: ] = sdf_inter.detach().cpu().numpy()[:resolution - base]
    #         else:
    #             u[base: base+N] = sdf_inter.detach().cpu().numpy()
    #     base = base + N
    u = u.transpose(2,1,0)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices = trans_mesh(vertices)
    mesh = trimesh.Trimesh(vertices, triangles)
    filedir += '.ply'
    mesh.export(filedir)
    return vertices, triangles
def trans_mesh(vertices, H=80, W=240, D=48, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    fx = fx_unit * W
    fy = fy_unit * H
    disp = np.linspace(0, 47, dtype=float)
    disp[0] = 1e-4
    depths = (fx * baseline / disp)[::-1]
    depths_diff = depths[1:] - depths[:-1]
    width_real_unit = depths / fx
    height_real_unit = depths / fy
    # print(depths)
    # print(width_real_unit)
    # print(depths[-1]-depths[-2])
    project_matrixs, scale_matrixs, translation_matrixs, trans_matrixs = [], [], [], []
    for idx, depth_diff in enumerate(depths_diff):
        near, far = depths[idx], depths[idx] + depth_diff
        project_matrixs.append(np.array([[near, 0, 0, 0], [0, near, 0, 0], [0, 0, near+far, -far*near], [0, 0, 1, 0]], dtype=float))
        translation_matrixs.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -near], [0, 0, 0, 1]], dtype=float))
        scale_matrixs.append(np.array([[1/(width_real_unit[idx]), 0, 0, W/2], [0, 1/(height_real_unit[idx]), 0, H/2], [0, 0, 1/depth_diff, idx], [0, 0, 0, 1]], dtype=float))
        trans_matrixs.append(np.linalg.inv(scale_matrixs[-1] @ translation_matrixs[-1] @ project_matrixs[-1]))
        # trans_matrixs.append(np.linalg.inv(np.matmul(scale_matrixs[-1], np.matmul(translation_matrixs_before[-1], project_matrixs[-1]))))
    import matplotlib.pyplot as plt

    vertices[:,2] = 47 -  vertices[:,2]

    for idx, p in enumerate(vertices):
        new_p = np.dot(trans_matrixs[int(p[2])], np.array(list(p)+[1]))
        new_p = (new_p / new_p[3])[:3]
        vertices[idx] = new_p
    return vertices
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