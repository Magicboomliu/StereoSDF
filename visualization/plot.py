'''
Author: BigCiLeng && bigcileng@outlook.com
Date: 2023-06-26 11:22:58
LastEditors: BigCiLeng && bigcileng@outlook.com
LastEditTime: 2023-06-26 11:27:22
FilePath: /StereoSDF/visualization/plot.py
Description: 

Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
'''

import numpy as np
import matplotlib.pyplot as plt
from visualization.depth import visualize_depth
'''
description: 绘制某一像素点的sdf曲线与预测深度
param {*} output: disparity (H * W)
param {*} est_sdf: sdf (D * H * W)
param {*} H: 像素点坐标0--319
param {*} W: 像素点坐标0--959
param {*} fx_unit
param {*} fy_unit
param {*} baseline
param {*} exp: 输出名字
return {*}
'''
def plot(output, est_sdf, H, W, fx_unit=0.58, fy_unit=1.92, baseline=0.54, exp='test'):
    filedir = 'visualization/output/' + exp + '_plot.jpg'
    depth_num, height, width = est_sdf.shape
    fx = fx_unit * width
    
    # 当前像素点， 1/4， sdf
    pt_sdf = est_sdf[:,H//4,W//4].cpu().detach().numpy()[::-1]

    # 当前像素点深度
    pt_depth = visualize_depth(output.cpu().detach().numpy(), exp=exp)[H,W]

    # 视差转深度坐标
    disp = np.linspace(0, depth_num - 1, depth_num, dtype=float)
    disp[0] = 1e-4
    depths = (fx * baseline / disp)[::-1]

    # 绘图
    plt.scatter(depths[:-1], pt_sdf[:-1], c='black', s=5)
    plt.axvline(pt_depth)
    plt.axhline(0)
    plt.savefig(filedir)
    plt.cla()