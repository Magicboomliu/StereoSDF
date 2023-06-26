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


KITTI_RAW_WIDTH = 1242
KITTI_RAW_HEIGHT = 375
CROP_WIDTH = 960
CROP_HEIGHT = 320


class DispWarper(object):
    def __init__(self, image_size, disp_range, device='cuda'):
        """
        image_size: (2,)
        disp_range: (D,)
        """
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        normal_coord = F.affine_grid(i_tetha,
                                     [1, 1, image_size[0], image_size[1]],
                                     align_corners=True)
        self.base_coord = normal_coord.to(device)  # (1, height, width, 2), range from -1 to 1

        # add disparity
        # D个 (1, 192, 640, 2)
        self.normal_disp_bunch = []
        zeros = torch.zeros_like(self.base_coord)
        for disp in disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] = disp_map[..., 0] + disp
            normal_disp_map = self._get_normalize_coord(disp_map, image_size)
            normal_disp_map = normal_disp_map.to(device=device)
            self.normal_disp_bunch.append(normal_disp_map)

        self.ch_num = len(disp_range)

    def _get_normalize_coord(self, coord, image_size):
        coord[..., 0] /= (image_size[1] / 2)
        coord[..., 1] /= (image_size[0] / 2)

        return coord

    def get_warped_frame(self, image, direct):
        base_coord = self.base_coord

        # D个 (B, 3, H, W)
        frame_volume = []
        B = image.shape[0]
        for ch_idx in range(self.ch_num):
            normal_disp = self.normal_disp_bunch[ch_idx]
            # 注意采样方向
            # 左图采样右图去重构左图 direct=-1, image为右图
            # 右图采样左图去重构右图 direct=1, image为左图
            grid_coord = normal_disp * direct + base_coord
            image = image.float()
            grid_coord = grid_coord.repeat(B, 1, 1, 1)
            warped_frame = F.grid_sample(image,
                                         grid_coord,
                                         mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)  # (B, 3, H, W)
            frame_volume.append(warped_frame.unsqueeze(2))

        frame_volume = torch.cat(frame_volume, dim=2)  # (B, 3, D, H, W)

        return frame_volume


def get_normalize_coord(coord, image_size):
    coord[..., 0] /= (image_size[1] / 2)
    coord[..., 1] /= (image_size[0] / 2)

    return coord


def disp_warp(right, disparity):
    B, _, H, W = right.shape
    device = right.device

    i_tetha = torch.zeros(1, 2, 3)
    i_tetha[:, 0, 0] = 1
    i_tetha[:, 1, 1] = 1
    normal_coord = F.affine_grid(i_tetha,
                                 [1, 1, H, W],
                                 align_corners=True)
    base_coord = normal_coord.to(device)  # (1, height, width, 2), range from -1 to 1
    base_coord = base_coord.repeat(B, 1, 1, 1)

    zeros = torch.zeros_like(base_coord)
    disp_map = zeros.clone()
    disp_map[..., 0:1] = disp_map[..., 0:1] - disparity.permute(0, 2, 3, 1)  # 左图采样右图去重构左图
    normal_disp_map = get_normalize_coord(disp_map, [H, W])
    normal_disp_map = normal_disp_map.to(device=device)

    grid_coord = normal_disp_map + base_coord
    warped_frame = F.grid_sample(right,
                                 grid_coord,
                                 mode='bilinear',
                                 padding_mode='border',
                                 align_corners=True)  # (B, 3, H, W)

    return warped_frame


class NeuSRenderer(nn.Module):
    def __init__(self, init_val=0.3, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
        super(NeuSRenderer, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

        self.fx_unit = fx_unit
        self.fy_unit = fy_unit
        self.baseline = baseline

    def render_core(self, sdf_batch, color_batch, hypo_depths):
        """
        sdf_batch: (512, D)
        color_batch: (512, D, 3)
        """
        batch_size, n_samples = sdf_batch.shape

        # 转换函数的超参
        inv_s = torch.ones([batch_size * (n_samples - 1), 1],
                           dtype=sdf_batch.dtype, device=sdf_batch.device) \
                * torch.exp(self.variance * 10.0)  # (512 * (D-1), 1)

        # 不是NeuS的mid_point采样，所以next_sdf和prev_sdf在sample维度少1
        next_sdf = sdf_batch[:, 1:].reshape(-1, 1)  # (512 * (D-1), 1)
        prev_sdf = sdf_batch[:, :-1].reshape(-1, 1)  # (512 * (D-1), 1)

        next_cdf = torch.sigmoid(next_sdf * inv_s)  # (512 * (D-1), 1)
        prev_cdf = torch.sigmoid(prev_sdf * inv_s)  # (512 * (D-1), 1)

        p = prev_cdf - next_cdf  # equation 13 分子
        c = prev_cdf  # equation 13 分母
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples - 1).clip(0.0, 1.0)  # equation 13, (512, D-1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1]  # (512, D-1)
        weights_sum = weights.sum(dim=-1, keepdim=True)  # (512, 1)

        # sampled_color: (512, D-1, 3)
        # weights: (512, D-1)
        color = (color_batch[:, :-1, :] * weights[:, :, None]).sum(dim=1)  # (512, 3)
        depth = (hypo_depths[None, :-1, None] * weights[:, :, None]).sum(dim=1)  # (512, 1)

        return color, weights, weights_sum, depth

    def batchify(self, sdf_grid, color_grid, hypo_depths, chunk=1024*32):
        color_list = []
        weights_list = []
        weights_sum_list = []
        depth_list = []
        for i in range(0, sdf_grid.shape[0], chunk):
            color_batch, weights_batch, weights_sum_batch, depth_batch = self.render_core(sdf_grid[i:i+chunk], color_grid[i:i+chunk], hypo_depths)
            color_list.append(color_batch)
            weights_list.append(weights_batch)
            weights_sum_list.append(weights_sum_batch)
            depth_list.append(depth_batch)

        color_list = torch.concat(color_list, dim=0)
        weights_list = torch.concat(weights_list, dim=0)
        weights_sum_list = torch.concat(weights_sum_list, dim=0)
        depth_list = torch.concat(depth_list, dim=0)

        return color_list, weights_list, weights_sum_list, depth_list


    def forward(self, sdf_grid, color_grid, right, is_debug=True):
        """
        sdf_grid: (B, D, H, W)
        color_grid: (B, 3, D, H, W)
        """
        sdf_grid = torch.flip(sdf_grid, dims=[1])  # 沿着depth从小到大的方向，排列sdf的格点
        color_grid = torch.flip(color_grid, dims=[2])  # 沿着depth从小到大的方向，排列color的格点

        B, D, H, W = sdf_grid.shape
        fx = self.fx_unit * W * (KITTI_RAW_WIDTH / CROP_WIDTH)
        fy = self.fy_unit * H * (KITTI_RAW_HEIGHT / CROP_HEIGHT)
        hypo_depths = torch.linspace(0, D-1, D).type_as(sdf_grid)
        hypo_depths[0] += 1e-2  # precision issue
        hypo_depths = fx * self.baseline / hypo_depths  # convert disparity to depth
        hypo_depths = torch.flip(hypo_depths, dims=[0])  # 沿着depth从小到大的方向

        sdf_grid = sdf_grid.permute(0, 2, 3, 1)
        sdf_grid = sdf_grid.flatten(start_dim=0, end_dim=2)  # ((B*H*W), D)

        color_grid = color_grid.permute(0, 3, 4, 2, 1)
        color_grid = color_grid.flatten(start_dim=0, end_dim=2)  # ((B*H*W), D, 3)

        # color, weights, weights_sum = self.render_core(sdf_grid, color_grid)
        color, weights, weights_sum, depth = self.batchify(sdf_grid, color_grid, hypo_depths)
        color = color.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        weights_sum = weights_sum.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        depth = depth.reshape(B, H, W, 1).permute(0, 3, 1, 2)

        warped_color = None
        if is_debug:
            disparity = fx * self.baseline / depth

            warped_color = disp_warp(right, disparity)

        return color, weights_sum, depth, warped_color


class VolSDFRenderer(object):
    def __init__(self, image_size, disp_range):
        pass


    def render(self, sdf_grid, color_grid):
        pass


class HFNeuSRenderer(object):
    pass


if __name__ == '__main__':
    B = 7
    D = 10
    H = 3
    W = 4
    sdf_grid = torch.randn(B, D, H, W).cuda()
    color_grid = torch.randn(B, 3, D, H, W).cuda()
    right = torch.randn(B, 3, H, W).cuda()

    renderer = NeuSRenderer().cuda()

    color, weights_sum, depth, warped = renderer(sdf_grid, color_grid, right)
    print(color.shape)
    print(weights_sum.shape)
    print(depth.shape)
    print(warped.shape)

    # right_img = torch.randn(B, 3, H, W).cuda()
    # warper = DispWarper(image_size=[H, W], disp_range=torch.arange(0, D, device='cuda', dtype=torch.float))
    # xxx = warper.get_warped_frame(right_img, -1)
    # print(xxx.shape)
