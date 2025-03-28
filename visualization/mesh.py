'''
Author: error: git config user.name & please set dead value or install git && bigcileng@outlook.com
Date: 2023-05-25 20:02:53
LastEditors: BigCiLeng && bigcileng@outlook.com
LastEditTime: 2023-06-26 16:56:50
FilePath: /stereoSDF-main/visualization/mesh.py
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
'''
description: 
param {*} est_sdf: (D * H * W)
param {*} filedir: 生成mesh的存放位置
param {*} threshold: sdf=0为表面, 用于marching_cubes
param {*} fx_unit: 
param {*} fy_unit: 
param {*} baseline: 
return {*}
'''
def visualize_mesh(est_sdf, exp='test', threshold=0.0, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    depth_num, height, width = est_sdf.shape
    u = est_sdf.detach().cpu().numpy()

    # 变换为(W, H, D)
    u = u.transpose(2,1,0)
    # marching cubes
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    # 还原为世界坐标
    vertices = trans_mesh(vertices, height, width, depth_num, fx_unit, fy_unit, baseline)

    # 生成mesh
    mesh = trimesh.Trimesh(vertices, triangles)
    filedir = 'visualization/output/' + exp + '_mesh.ply'
    mesh.export(filedir)
    return vertices, triangles

def trans_mesh(vertices, H=80, W=240, D=48, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    fx = fx_unit * W
    fy = fy_unit * H

    # disparity对应depth
    disp = np.linspace(0, D-1, D+1, dtype=float)
    disp[0] = 1e-4
    depths = (fx * baseline / disp)[::-1]
    depths_diff = depths[1:] - depths[:-1]

    # 计算每像素真实长宽
    width_real_unit = depths / fx
    height_real_unit = depths / fy
    project_matrixs, scale_matrixs, translation_matrixs, trans_matrixs = [], [], [], []

    for idx, depth_diff in enumerate(depths_diff):
        # near, far平面
        near, far = depths[idx], depths[idx] + depth_diff
        # 投影、变换、尺度变换，将中心轴位于h=0,w=0的视锥变换到定点在（0，0，0）的正方体
        project_matrixs.append(np.array([[near, 0, 0, 0], [0, near, 0, 0], [0, 0, near+far, -far*near], [0, 0, 1, 0]], dtype=float))
        translation_matrixs.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -near], [0, 0, 0, 1]], dtype=float))
        scale_matrixs.append(np.array([[1/(width_real_unit[idx]), 0, 0, W/2], [0, 1/(height_real_unit[idx]), 0, H/2], [0, 0, 1/depth_diff, idx], [0, 0, 0, 1]], dtype=float))
        # 矩阵逆运算，将sdf_grid转为视锥
        trans_matrixs.append(np.linalg.inv(scale_matrixs[-1] @ translation_matrixs[-1] @ project_matrixs[-1]))
    for idx, p in enumerate(vertices):
        new_p = np.dot(trans_matrixs[int(p[2])], np.array(list(p)+[1]))
        new_p = (new_p / new_p[3])[:3]
        vertices[idx] = new_p
    return vertices
# example
if __name__ == "__main__":
    pass