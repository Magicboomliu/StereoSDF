import torch
import torch.nn  as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import re
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from losses.unsupervised_loss import warp_disp

import open3d as o3d
import numpy as np 


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)
    

def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img


def read_disp(filename, subset=False):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp

    return disp  # [H, W]



def tensor_to_vis(tensor):
    channel = tensor.shape[1]
    if channel==1:
        return tensor.squeeze(0).squeeze(0).cpu().numpy()
    else:
        return tensor.squeeze(0).permute(1,2,0).cpu().numpy()



def warp_right2left(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # 计算掩码：边界内的位置保持1，边界外的位置变为0
    mask = torch.ones_like(disp)
    mask = F.grid_sample(mask, 2 * flow_field - 1, mode='nearest', padding_mode='zeros', align_corners=False)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output,mask




def compute_right_disparity_and_mask(left_disparity):
    """
    使用grid_sample根据左图的视差图计算右图的视差图，并生成一个掩码标记边界外的像素。

    参数:
    left_disparity (torch.Tensor): 左图的视差图，形状为 [1, 1, H, W]。

    返回:
    torch.Tensor: 右图的视差图，形状为 [1, 1, H, W]。
    torch.Tensor: 边界外像素掩码，形状为 [1, 1, H, W]，边界内的像素为1，边界外的像素为0。
    """
    B, C, H, W = left_disparity.size()

    # 创建归一化的坐标网格
    xx = torch.linspace(0, W - 1, W).repeat(B, H, 1)
    yy = torch.linspace(0, H - 1, H).repeat(B, W, 1).transpose(1, 2)
    grid = torch.stack((xx, yy), dim=3).float()

    # 根据视差调整网格位置
    grid[:, :, :, 0] = 2.0 * (grid[:, :, :, 0] - left_disparity.squeeze(1)) / (W - 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0

    # 计算掩码：边界内的位置保持1，边界外的位置变为0
    mask = torch.ones_like(left_disparity)
    mask = F.grid_sample(mask, grid, mode='nearest', padding_mode='zeros', align_corners=False)

    # 使用grid_sample变换左视差图得到右视差图
    right_disparity = F.grid_sample(left_disparity, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return right_disparity, mask


def disparitry_to_point_clound(disp,focal_length,baseline):
    depth = focal_length * baseline/(disp+1e-6)
    b,c,h,w = depth.shape


    # Original coordinates of pixels
    xx = torch.arange(0, w).view(1, 1, w).expand(b, h, w).type_as(depth)  # [1, H, W]
    yy = torch.arange(0, h).view(1, h, 1).expand(b, h, w).type_as(depth)
    xx = xx.float()
    yy = yy.float()

    # grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    # grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]
    xx = (xx - w / 2) / focal_length
    yy = (yy - h / 2) / focal_length
    zz = depth.squeeze(1)
    
    xx = xx*zz
    yy = yy*zz

    
    point_cloud = torch.stack([xx, yy, zz], dim=-1) #[B,H,W,3]

    return point_cloud

def cam2word(cam_3d,pose):
    b,h,w,c = cam_3d.shape
    ones = torch.ones(b, h, w, 1)
    point_cloud_homogeneous = torch.cat((cam_3d, ones), dim=-1)  # [B, H, W, 4]

    # 投影点云到世界坐标系
    point_cloud_world_coords = torch.einsum('bhwi,bij->bhwj', point_cloud_homogeneous, pose)
    point_cloud_world_coords = point_cloud_world_coords[..., :3]
    
    return point_cloud_world_coords

def word2cam(cam_3d,pose_inv):
    b,h,w,c = cam_3d.shape
    ones = torch.ones(b, h, w, 1)
    point_cloud_homogeneous = torch.cat((cam_3d, ones), dim=-1)  # [B, H, W, 4]

    points_new_camera = torch.einsum('bhwk,bkl->bhwl', point_cloud_homogeneous, pose_inv)

    # 去除齐次坐标，留下3D坐标
    points_new_camera = points_new_camera[..., :3]
    
    
    return points_new_camera
    


if __name__=="__main__":
    
    example_case = "/home/zliu/ACMMM2024/UnsupervisedStereo/StereoSDF/test_images/example3"
    
    left_image_path = os.path.join(example_case,"left.png")
    right_image_path = os.path.join(example_case,"right.png")
    left_disp = os.path.join(example_case,"left_disp.pfm")
    
    
    left_image_data = read_img(left_image_path)
    right_image_data = read_img(right_image_path)
    left_disp_data = read_disp(left_disp)
    
    H,W = left_disp_data.shape[:2]
    
    
    left_image_tensor =torch.from_numpy(left_image_data/255.).permute(2,0,1).unsqueeze(0)
    right_image_tensor = torch.from_numpy(right_image_data/255.).permute(2,0,1).unsqueeze(0)
    left_disp_tensor = torch.from_numpy(left_disp_data).unsqueeze(0).unsqueeze(0)
    
    
    focal_length = 1000
    baseline = 1
    
    left_point_cloud = disparitry_to_point_clound(left_disp_tensor.repeat(2,1,1,1),focal_length=1000,baseline=baseline)



    # 相机的外参矩阵
    R = torch.eye(3)  # 3x3单位旋转矩阵
    T = torch.zeros(1, 3)  # 3x1零平移向量，调整为1x3以适配广播
    # 构建完整的外参矩阵 [B, 4, 4]
    extrinsics = torch.zeros(2, 4, 4)
    extrinsics[:, :3, :3] = R.unsqueeze(0).repeat(2, 1, 1)  # 复制旋转矩阵
    extrinsics[:, :3, 3] = T.repeat(2, 1)  # 复制平移向量
    extrinsics[:, 3, 3] = 1  # 设置齐次坐标行
    
    
    right_camera_extrinsics = extrinsics  # 沿x轴的平移
    right_camera_extrinsics[:,0, 3] = baseline  # 沿x轴的平移
    

    
    inverse_batch_matrices = right_camera_extrinsics
    
    
    world_point_cloud = cam2word(left_point_cloud,extrinsics)
    

    point_cloud_in_right = world_point_cloud
    point_cloud_in_right = point_cloud_in_right
    point_cloud_in_right[..., 0] +=baseline
    
    


    
    # 从新相机坐标系投影点云到2D图像平面
    x_new = point_cloud_in_right[..., 0]
    y_new = point_cloud_in_right[..., 1]
    z_new = point_cloud_in_right[..., 2]

    # 计算新相机视角下的像素坐标
    u_new = focal_length * x_new / z_new + W / 2
    v_new = focal_length * y_new / z_new + H / 2


    new_depth_maps = torch.zeros_like(left_disp_tensor).repeat(2,1,1,1)

        
    # beigin here
    u_new_rounded = torch.round(u_new).long()
    v_new_rounded = torch.round(v_new).long()

    # 检查像素坐标是否在图像尺寸范围内
    valid_indices = (u_new_rounded >= 0) & (u_new_rounded < W) & (v_new_rounded >= 0) & (v_new_rounded < H)

    # 更新新相机的深度图
    for b in range(2):
        valid_u = u_new_rounded[b][valid_indices[b]]
        valid_v = v_new_rounded[b][valid_indices[b]]
        valid_z = z_new[b][valid_indices[b]]
        
        # 这里我们简单地使用最近点的深度值来更新新相机的深度图
        # 更复杂的插值方法可以用于更平滑的深度图结果
        new_depth_maps[b, 0, valid_v, valid_u] = valid_z
    
    
    valid_mask = new_depth_maps>0
    valid_mask = valid_mask.float()
    
    
    
    
    disp = (baseline * focal_length)/(new_depth_maps+1e-6)
    disp = disp * valid_mask
    



    # raw_point= world_point_cloud.view(-1,3)
    
    # raw_point_2 = point_cloud_in_right.view(-1,3)

    # #创建窗口对象
    # vis = o3d.visualization.Visualizer()
    # #设置窗口标题
    # vis.create_window(window_name="kitti")
    # #设置点云大小
    # vis.get_render_option().point_size = 1
    # #设置颜色背景为黑色
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])

    # #创建点云对象
    # pcd=o3d.open3d.geometry.PointCloud()
    # #将点云数据转换为Open3d可以直接使用的数据类型
    # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    # pcd.paint_uniform_color([0,1,0])

    # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point_2)
    #     #设置点的颜色为白色
    # pcd.paint_uniform_color([1,1,1])
    

    # #将点云加入到窗口中
    # vis.add_geometry(pcd)

    # vis.run()
    # vis.destroy_window()
    # print((world_point_cloud-left_point_cloud).mean())
    # print(world_point_cloud.shape)
    
    
    # warped_right_disparity,warped_right_disparity_mask = compute_right_disparity_and_mask(left_disp_tensor)
    # warped_right_disparity = warped_right_disparity* warped_right_disparity_mask 
    
    # image_warped_left,rgb_mask = warp_right2left(right_image_tensor, -left_disp_tensor)
    # # image_warped_right = warp_right2left(left_image_tensor,left_disp_tensor)
    
    left_vis = tensor_to_vis(left_image_tensor)
    right_vis = tensor_to_vis(right_image_tensor)
    left_disp_vis = tensor_to_vis(left_disp_tensor)
    right_disp_vis = tensor_to_vis(disp[0:1,:,:,:])



    plt.figure(figsize=(10,4))
    plt.subplot(2,2,1)
    plt.axis("off")
    plt.imshow(left_vis)
    plt.subplot(2,2,2)
    plt.axis('off')
    plt.imshow(right_vis)
    plt.subplot(2,2,3)
    plt.axis('off')
    plt.imshow(left_disp_vis)
    plt.subplot(2,2,4)
    plt.axis('off')
    plt.imshow(right_disp_vis)
    plt.show()

    
    
    pass



