"""
SDF Grid: (B, D, H, W)

1. Obtain Color Grid: (B, 3, D, H, W)
2. VolSDF rendering
3. NeuS rendering
4. HF-NeuS rendering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sdf_mlp import SDF_MLP


KITTI_RAW_WIDTH = 1242
KITTI_RAW_HEIGHT = 375
CROP_WIDTH = 960
CROP_HEIGHT = 320


class BackprojectDepth(nn.Module):
    def __init__(self, batch_size, height, width, num_depths, depth_interval):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_depths = num_depths
        self.depth_interval = depth_interval

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                        requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                    requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                        requires_grad=False)

    def forward(self, depth_range, inv_K):
        # repeat inv_K
        inv_K = inv_K.repeat(self.batch_size, 1, 1)  # (B, 4, 4)

        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)  # (B, 3, HW)
        
        min_depth, max_depth = depth_range
        D_planes = torch.linspace(min_depth, max_depth, self.num_depths).to(device=cam_points.device)  # (D, )
        # random
        D_planes = D_planes + torch.rand(self.num_depths,).to(device=cam_points.device) * self.depth_interval

        cam_points = D_planes[None, None, :, None] * cam_points[..., None, :]  # (B, 3, D, HW)

        ones = self.ones[..., None, :].repeat(1, 1, self.num_depths, 1)  # (B, 1, D, HW)
        cam_points = torch.cat([cam_points, ones], 1)  # (B, 4, D, HW)

        cam_points = cam_points.view(self.batch_size, 4, -1)  # (B, 4, DHW)

        return cam_points, D_planes
    

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, source_height, source_width, num_depths, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_depths = num_depths
        self.source_height = source_height
        self.source_width = source_width
        self.eps = eps

    def forward(self, points, K, T):
        # repeat K, T
        K = K.repeat(self.batch_size, 1, 1)  # (B, 4, 4)
        T = T.repeat(self.batch_size, 1 ,1)  # (B, 4, 4)

        P = torch.matmul(K, T)[:, :3, :]  # (B, 3, 4)

        cam_points = torch.matmul(P, points)  # (B, 3, DHW)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.num_depths * self.source_height, self.source_width)  # (B, 2, DH, W)
        pix_coords = pix_coords.permute(0, 2, 3, 1)  # (B, DH, W, 2)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2  # (B, DH, W, 2)

        valid_mask = pix_coords.abs().max(dim=-1)[0] <= 1  # (B, DH, W)

        return pix_coords, valid_mask


class NewNeusRenderer(nn.Module):
    def __init__(self, batch_size, height, width, depth_range, num_depths,
                 inv_K, transform_RtoL, transform_RtoR, color_K, feat_K,
                 color_width, color_height, feat_width, feat_height,
                 feat_length, cv_feat_length) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth_range = depth_range
        min_depth, max_depth = depth_range
        self.depth_interval = (max_depth - min_depth) / (num_depths - 1)
        self.num_depths = num_depths
        self.transform_RtoL = transform_RtoL  # (4, 4)
        self.transform_RtoR = transform_RtoR

        self.inv_K = inv_K  # (4, 4)
        self.color_K = color_K  # (4, 4)
        self.feat_K = feat_K  # (4, 4)

        self.color_width = color_width
        self.color_height = color_height
        self.feat_width = feat_width
        self.feat_height = feat_height

        self.feat_length = feat_length
        self.cv_feat_length = cv_feat_length

        self.backproject = BackprojectDepth(self.batch_size, self.height, self.width, self.num_depths, self.depth_interval)
        self.project_color = Project3D(self.batch_size, self.color_height, self.color_width, self.height, self.width, self.num_depths)
        self.project_feat = Project3D(self.batch_size, self.feat_height, self.feat_width, self.height, self.width, self.num_depths)

        self.sdf_mlp = SDF_MLP(self.feat_length, self.cv_feat_length)

    def batchify(self, sampled_left_feat, sampled_right_feat, sampled_left_cv, xyz, chunk=1024*32):
        """
        (B, F, D, H, W)
        (B, F, D, H, W)
        (B, Cd, D, H, W)
        (B, 3, D, H, W)
        """
        sdf_list = []
        gradient_list = []

        sampled_left_feat = sampled_left_feat.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3)
        sampled_right_feat = sampled_right_feat.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3)
        sampled_left_cv = sampled_left_cv.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3)
        xyz = xyz.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3)

        for i in range(0, sampled_left_feat.shape[0], chunk):
            sdf = self.sdf_mlp(sampled_left_feat[i:i+chunk],
                               sampled_right_feat[i:i+chunk],
                               sampled_left_cv[i:i+chunk],
                               xyz[i:i+chunk])
            gradients = self.sdf_mlp.gradient(sampled_left_feat[i:i+chunk],
                               sampled_right_feat[i:i+chunk],
                               sampled_left_cv[i:i+chunk],
                               xyz[i:i+chunk])

            sdf_list.append(sdf)
            gradient_list.append(gradients)

        sdf_list = torch.concat(sdf_list, dim=0)
        sdf_grid = sdf_list.view(self.batch_size, self.num_depths, self.height, self.width, 1)
        gradient_list = torch.concat(gradient_list, dim=0)
        gradient_list = gradient_list.view(self.batch_size, self.num_depths, self.height, self.width, 1, 3)

        return sdf_grid, gradient_list

    def forward(self, left_feat, right_feat, left_volume, left_img):
        cam_points, D_planes = self.backproject(self.depth_range, self.inv_K)  # ((B, 4, DHW)), (D,)

        # get color
        pix_coords_color, valid_mask_color = self.project_color(cam_points, self.color_K, self.transform_RtoL)  # (B, DH, W, 2), (B, DH, W)
        sampled_color_grid = F.grid_sample(left_img, pix_coords_color, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 3, DH, W)
        sampled_color_grid = sampled_color_grid.view(self.batch_size, 3, self.num_depths, self.height, self.width)  # (B, 3, D, H, W)

        # get sdf
        pix_coords_feat, valid_mask_feat = self.project_feat(cam_points, self.feat_K, self.transform_RtoL)  # (B, DH, W, 2), (B, DH, W)
        sampled_left_feat = F.grid_sample(left_feat, pix_coords_feat, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, F, DH, W)
        sampled_left_feat = sampled_left_feat.view(self.batch_size, self.feat_length, self.num_depths, self.height, self.width)  # (B, F, D, H, W)

        # sampled_right_feat = right_feat[:, :, None, ...].repeat(1, 1, self.num_depths, 1, 1)  # (B, F, D, H, W)
        pix_coords_feat_right, _ = self.project_feat(cam_points, self.feat_K, self.transform_RtoR)
        sampled_right_feat = F.grid_sample(left_feat, pix_coords_feat_right, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, F, DH, W)
        sampled_right_feat = sampled_left_feat.view(self.batch_size, self.feat_length, self.num_depths, self.height, self.width)  # (B, F, D, H, W)

        sampled_left_cv = F.grid_sample(left_volume, pix_coords_feat, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, Cd, DH, W)
        sampled_left_cv = sampled_left_cv.view(self.batch_size, self.cv_feat_length, self.num_depths, self.height, self.width)  # (B, Cd, D, H, W)

        xyz = cam_points[:, :3, ...]  # (B, 3, DHW)
        xyz = xyz.view(self.batch_size, 3, self.num_depths, self.height, self.width)  # (B, 3, D, H, W)

        sdf_grid, gradient_list = self.batchify(sampled_left_feat, sampled_right_feat, sampled_left_cv, xyz)

        return sampled_color_grid, sdf_grid, gradient_list


def scale_K(K, scale=1):
    return np.array([[K[0,0]*scale, 0, K[0,2]*scale, 0],
                     [0, K[1,1]*scale, K[1,2]*scale, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


if __name__ == '__main__':
    batch_size = 8
    height = 320 // 4
    width = 960 // 4
    depth_range = [0.1, 80]
    num_depths = 4  # 128
    K = np.array([[192, 0, 200, 0],
                  [0, 192, 100, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    inv_K = torch.from_numpy(np.linalg.inv(K)).cuda()
    transform_RtoL = torch.from_numpy(np.array([[1, 0, 0, 0.54],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)).cuda()
    transform_RtoR = torch.from_numpy(np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)).cuda()
    color_K = torch.from_numpy(scale_K(K, 4)).cuda()
    feat_K = torch.from_numpy(scale_K(K, 0.5)).cuda()
    color_width = width * 4
    color_height = height * 4
    feat_width = width // 2
    feat_height = height // 2

    feat_length = 512
    cv_feat_length = 128

    left_feat = torch.randn(batch_size, feat_length, feat_height, feat_width).cuda()
    right_feat = torch.randn(batch_size, feat_length, feat_height, feat_width).cuda()
    left_volume = torch.randn(batch_size, cv_feat_length, feat_height, feat_width).cuda()
    left_img = torch.randn(batch_size, 3, color_height, color_width).cuda()

    renderer = NewNeusRenderer(batch_size, height, width, depth_range, num_depths,
                 inv_K, transform_RtoL, transform_RtoR, color_K, feat_K,
                 color_width, color_height, feat_width, feat_height,
                 feat_length, cv_feat_length).cuda()
    
    sampled_color_grid, sdf_grid, gradient_list = renderer(left_feat, right_feat, left_volume, left_img)
    print(sampled_color_grid.shape)
    print(sdf_grid.shape)
    print(gradient_list.shape)
    
    print('over')

    # right_img = torch.randn(B, 3, H, W).cuda()
    # warper = DispWarper(image_size=[H, W], disp_range=torch.arange(0, D, device='cuda', dtype=torch.float))
    # xxx = warper.get_warped_frame(right_img, -1)
    # print(xxx.shape)
