import numpy as np
from PIL import Image
import cv2
'''
description: 
param {*} disp: 视差(H * W)
param {*} filedir: 存放深度图位置
param {*} fx_unit
param {*} fy_unit
param {*} baseline
return {*}
'''
def visualize_depth(disp, exp='test',fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    height, width = disp.shape
    fx = fx_unit * width
    fy = fy_unit * height

    depths = fx * baseline / disp  # convert disparity to depth

    depths = np.nan_to_num(depths) # change nan to 0
    mi = np.min(depths) # get minimum depth
    ma = np.max(depths)
    depths_norm = (depths-mi)/(ma-mi+1e-8) # normalize to 0~1

    depths_color = (255*depths_norm).astype(np.uint8) # 转化为RGB
    depths_photo = Image.fromarray(cv2.applyColorMap(depths_color, cv2.COLORMAP_JET))
    filedir = 'visualization/output/' + exp + '_depth.jpg'
    depths_photo.save(filedir)
    return depths