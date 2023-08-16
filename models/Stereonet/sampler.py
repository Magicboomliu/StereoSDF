import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .sdf_mlp import SDF_MLP
from .render import NeuSRenderer


class BackprojectDepth:
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
        inv_K = inv_K.cpu()

        # inv_K = inv_K.repeat(self.batch_size, 1, 1)  # (B, 4, 4)

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
    

class Project3D:
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

        self.x_offset = None
        self.y_offset = None

    def forward(self, points, K, T):
        points = points.cpu()
        K = K.cpu()
        T = T.cpu()
        # repeat K, T
        # K = K.repeat(self.batch_size, 1, 1)  # (B, 4, 4)
        T = T.repeat(self.batch_size, 1 ,1)  # (B, 4, 4)

        P = torch.matmul(K, T)[:, :3, :]  # (B, 3, 4)

        cam_points = torch.matmul(P, points)  # (B, 3, DHW)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.num_depths, self.source_height, self.source_width)  # (B, 2, D, H, W)
        pix_coords = pix_coords.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 2)
        pix_coords[..., 0] = (pix_coords[..., 0]) / (self.width - 1)
        pix_coords[..., 1] = (pix_coords[..., 1]) / (self.height - 1)
        pix_coords = (pix_coords - 0.5) * 2  # (B, D, H, W, 2)

        valid_mask = pix_coords.abs().max(dim=-1)[0] <= 1  # (B, D, H, W)

        return pix_coords, valid_mask


class NeuSSampler(nn.Module):
    def __init__(self, batch_size, height, width, depth_range, num_depths,
                 inv_K, transform_LtoR, transform_LtoL, color_K, feat_K,
                 color_width, color_height, feat_width, feat_height,
                 feat_length, cv_feat_length, N_rand=1024) -> None:
        """
        当前版本, height==color_height, width==color_width
        """
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth_range = depth_range
        min_depth, max_depth = depth_range
        self.depth_interval = (max_depth - min_depth) / (num_depths - 1)
        self.num_depths = num_depths
        self.transform_LtoR = transform_LtoR  # (4, 4)
        self.transform_LtoL = transform_LtoL

        self.inv_K = inv_K  # (4, 4)
        self.color_K = color_K  # (4, 4)
        self.feat_K = feat_K  # (4, 4)

        self.color_width = color_width
        self.color_height = color_height
        self.feat_width = feat_width
        self.feat_height = feat_height

        self.feat_length = feat_length
        self.cv_feat_length = cv_feat_length

        self.N_rand = N_rand

        self.backproject = BackprojectDepth(self.batch_size, self.height, self.width, self.num_depths, self.depth_interval)
        self.project_color = Project3D(self.batch_size, self.color_height, self.color_width, self.height, self.width, self.num_depths)
        self.project_feat = Project3D(self.batch_size, self.feat_height, self.feat_width, self.height, self.width, self.num_depths)

        self.sdf_mlp = SDF_MLP(self.feat_length, self.cv_feat_length)
        self.render = NeuSRenderer()

    def batchify(self, sampled_left_feat, sampled_right_feat, sampled_left_cv, xyz, chunk=1024*32):
        """
        (B, F, D, N)
        (B, F, D, N)
        (B, Cd, D, N)
        (B, 3, D, N)
        """
        sdf_list = []
        gradient_list = []

        sampled_left_feat = sampled_left_feat.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        sampled_right_feat = sampled_right_feat.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        sampled_left_cv = sampled_left_cv.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        xyz = xyz.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)

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
        sdf_grid = sdf_list.view(self.batch_size, self.num_depths, self.N_rand, 1)
        gradient_list = torch.concat(gradient_list, dim=0)
        gradient_list = gradient_list.view(self.batch_size, self.num_depths, self.N_rand, 1, 3)

        return sdf_grid[..., 0], gradient_list

    def batchify_whole_image(self, sampled_left_feat, sampled_right_feat, sampled_left_cv, xyz, chunk=1024*32):
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

        return sdf_grid[..., 0], gradient_list

    def forward(self, left_feat, right_feat, left_cv_feat, left_img, right_img):
        cam_points, D_planes = self.backproject.forward(self.depth_range, self.inv_K)  # ((B, 4, DHW)), (D,)


        select_inds = np.random.choice(self.height * self.width, size=[self.N_rand], replace=False)  # (N_rand,)

        # get color for ground-truth
#        left_img_gt = left_img.view(self.batch_size, 3, self.height * self.width)
#        left_img_gt = left_img_gt[..., select_inds]  # (B, 3, N)
#
        pix_coords_color, valid_mask_color = self.project_color.forward(cam_points, self.color_K, self.transform_LtoL)  # (B, D, H, W, 2), (B, D, H, W)
        pix_coords_color, valid_mask_color = select_data(pix_coords_color, valid_mask_color, select_inds)  # (B, D, N, 2), (B, D, N)
        pix_coords_color = pix_coords_color.cuda()
        valid_mask_color = valid_mask_color.cuda()
        weight = torch.sum(valid_mask_color, dim=1) / self.num_depths
        left_img_gt = F.grid_sample(left_img, pix_coords_color, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 3, D, N)
        left_img_gt = left_img_gt[:, :, 0]

        # get color for rendering
        pix_coords_color, valid_mask_color = self.project_color.forward(cam_points, self.color_K, self.transform_LtoR)  # (B, D, H, W, 2), (B, D, H, W)
        pix_coords_color, valid_mask_color = select_data(pix_coords_color, valid_mask_color, select_inds)  # (B, D, N, 2), (B, D, N)
        pix_coords_color = pix_coords_color.cuda()
        valid_mask_color = valid_mask_color.cuda()
        weight = torch.sum(valid_mask_color, dim=1) / self.num_depths
        sampled_color_grid = F.grid_sample(right_img, pix_coords_color, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 3, D, N)

        # get sdf feature (left feature)
        pix_coords_left_feat, valid_mask_left_feat = self.project_feat.forward(cam_points, self.feat_K, self.transform_LtoL)  # (B, D, H, W, 2), (B, D, H, W)
        pix_coords_left_feat, valid_mask_left_feat = select_data(pix_coords_left_feat, valid_mask_left_feat, select_inds)  # (B, D, N, 2), (B, D, N)
        pix_coords_left_feat = pix_coords_left_feat.cuda()
        valid_mask_left_feat = valid_mask_left_feat.cuda()
        sampled_left_feat = F.grid_sample(left_feat, pix_coords_left_feat, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, F, D, N)

        # get sdf feature (right feature)
        pix_coords_right_feat, valid_mask_right_feat = self.project_feat.forward(cam_points, self.feat_K, self.transform_LtoR)  # (B, D, H, W, 2), (B, D, H, W)
        pix_coords_right_feat, valid_mask_right_feat = select_data(pix_coords_right_feat, valid_mask_right_feat, select_inds)  # (B, D, N, 2), (B, D, N)
        pix_coords_right_feat = pix_coords_right_feat.cuda()
        valid_mask_right_feat = valid_mask_right_feat.cuda()
        sampled_right_feat = F.grid_sample(right_feat, pix_coords_right_feat, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, F, D, N)

        # get sdf feature (left volume)
        sampled_left_cv = F.grid_sample(left_cv_feat, pix_coords_left_feat, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, Cd, D, N)

        # get sdf feature (xyz)
        xyz = cam_points[:, :3, ...].view(self.batch_size, 3, self.num_depths, self.height * self.width)  # (B, 3, D, HW)
        xyz = xyz[..., select_inds].cuda()
        xyz = xyz.view(self.batch_size, 3, self.num_depths, self.N_rand)  # (B, 3, D, N)

        # print('sampled_left_feat', sampled_left_feat.shape)
        # print('sampled_left_cv', sampled_left_cv.shape)
        # print('xyz', xyz.shape)
        # exit()
        sdf_grid, gradient_list = self.batchify(sampled_left_feat, sampled_right_feat, sampled_left_cv, xyz)

        # render
        left_img_render, _ = self.render(sdf_grid, sampled_color_grid)
        return {
            'left_ray_render': left_img_render,
            'left_ray_gt': left_img_gt,
            'left_ray_weight': weight,
            'sdf_gradient': gradient_list
        }

    @torch.no_grad
    def render_whole_image(self, left_feat, right_feat, left_cv_feat, left_img, right_img):
        cam_points, D_planes = self.backproject.forward(self.depth_range, self.inv_K)  # ((B, 4, DHW)), (D,)
        pix_coords_color, valid_mask_color = self.project_color.forward(cam_points, self.color_K, self.transform_LtoL)  # (B, D, H, W, 2), (B, D, H, W)
        left_img_gt = F.grid_sample(left_img, pix_coords_color[:, 0, ...], mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 3, H, W)

        # get color for rendering
        pix_coords_color, valid_mask_color = self.project_color.forward(cam_points, self.color_K, self.transform_LtoR)  # (B, D, H, W, 2), (B, D, H, W)
        sampled_color_grid = grid_sample_helper(right_img, pix_coords_color)  # (B, 3, D, H, W)

        # get sdf feature (left feature)
        pix_coords_left_feat, valid_mask_left_feat = self.project_feat.forward(cam_points, self.feat_K, self.transform_LtoL)  # (B, D, H, W, 2), (B, D, H, W)
        sampled_left_feat = grid_sample_helper(left_feat, pix_coords_left_feat)  # (B, F, D, H, W)

        # get sdf feature (right feature)
        pix_coords_right_feat, valid_mask_right_feat = self.project_feat.forward(cam_points, self.feat_K, self.transform_LtoR)  # (B, D, H, W, 2), (B, D, H, W)
        sampled_right_feat = grid_sample_helper(right_feat, pix_coords_right_feat)  # (B, F, D, H, W)

        # get sdf feature (left volume)
        sampled_left_cv = grid_sample_helper(left_cv_feat, pix_coords_left_feat)  # (B, Cd, D, H, W)

        # get sdf feature (xyz)
        xyz = cam_points[:, :3, ...].view(self.batch_size, 3, self.num_depths, self.height, self.width)  # (B, 3, D, H, W)

        sdf_grid, gradient_list = self.batchify_whole_image(sampled_left_feat, sampled_right_feat, sampled_left_cv, xyz)

        left_img_render, _ = self.render.forward_whole_image(sdf_grid, sampled_color_grid)

        return {
            'left_ray_render': left_img_render,
            'left_ray_gt': left_img_gt,
            'sdf_gradient': gradient_list
        }

def select_data(pix_coords, valid_mask, select_inds):
    """
    """
    B, D, H, W, _ = pix_coords.shape
    
    pix_coords = pix_coords.flatten(start_dim=2, end_dim=3)  # (B, D, HW, 2)
    valid_mask = valid_mask.flatten(start_dim=2, end_dim=3)  # (B, D, HW)

    pix_coords = pix_coords[..., select_inds, :]  # (B, D, N, 2)
    valid_mask = valid_mask[..., select_inds]  # (B, D, N)

    return pix_coords, valid_mask


def grid_sample_helper(image, sample_points):
    """
    image: (B, 3, H, W)
    sample_points: (B, D, H, W, 2)
    """
    sampled_color_grid_list = []
    for i in range(sample_points.shape[1]):
        sampled_color_grid = F.grid_sample(image, sample_points[:, i], mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_color_grid_list.append(sampled_color_grid)

    sampled_color_grid_list = torch.stack(sampled_color_grid_list, dim=2)

    return sampled_color_grid_list


def scale_K(K, scale=1):
    return np.array([[K[0,0]*scale, 0, K[0,2]*scale, 0],
                     [0, K[1,1]*scale, K[1,2]*scale, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


if __name__ == '__main__':
    batch_size = 4
    height = 320
    width = 960
    depth_range = [0.1, 80]
    num_depths = 128  # 128
    K = np.array([[200, 0, 480, 0],
                  [0, 200, 160, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    inv_K = torch.from_numpy(np.linalg.inv(K)).cuda()
    transform_LtoR = torch.from_numpy(np.array([[1, 0, 0, -0.54],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)).cuda()
    transform_LtoL = torch.from_numpy(np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)).cuda()
    color_K = torch.from_numpy(scale_K(K, 1)).cuda()
    feat_K = torch.from_numpy(scale_K(K, 1/8)).cuda()
    color_width = width
    color_height = height
    feat_width = width // 8
    feat_height = height // 8

    feat_length = 512
    cv_feat_length = 128

    left_feat = torch.randn(batch_size, feat_length, feat_height, feat_width).cuda()
    right_feat = torch.randn(batch_size, feat_length, feat_height, feat_width).cuda()
    left_volume = torch.randn(batch_size, cv_feat_length, feat_height, feat_width).cuda()
    left_img = torch.randn(batch_size, 3, color_height, color_width).cuda()
    right_img = torch.randn(batch_size, 3, color_height, color_width).cuda()

    renderer = NeuSSampler(batch_size, height, width, depth_range, num_depths,
                 inv_K, transform_LtoR, transform_LtoL, color_K, feat_K,
                 color_width, color_height, feat_width, feat_height,
                 feat_length, cv_feat_length, N_rand=256).cuda()
    
    sampled_color_grid, sdf_grid, gradient_list = renderer(left_feat, right_feat, left_volume, left_img, right_img)
    print(sampled_color_grid.shape)
    print(sdf_grid.shape)
    print(gradient_list.shape)

    a = input()
    
    print('over')

    # right_img = torch.randn(B, 3, H, W).cuda()
    # warper = DispWarper(image_size=[H, W], disp_range=torch.arange(0, D, device='cuda', dtype=torch.float))
    # xxx = warper.get_warped_frame(right_img, -1)
    # print(xxx.shape)
