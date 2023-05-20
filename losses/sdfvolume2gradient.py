import torch
import torch.nn as nn
import torch.nn.functional as F


def SDF2Graident(sdf_volume, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    '''
    SDF_volume: (B, D, H, W)
    K: (3, 3)
    hypo_depths: (D, )

    SDF_volume为1/8分辨率下
    注意K也要被scale到1/8分辨率
    '''
    batch_num, depth_num, height, width = sdf_volume.shape
    # i, j = torch.meshgrid(torch.linspace(0, width-1, width), torch.linspace(0, height-1, height), indexing='xy')
    # coords_2d = torch.stack([i, j, torch.ones_like(i)], dim=0)  # (3, H, W)
    # inv_K = torch.linalg.inv(K)
    # coords_3d = torch.einsum('ij,jhw->ihw', inv_K, coords_2d)  # (3, H, W)

    # hypo_depths = hypo_depths[None, :, None, None]  # (1, D, 1, 1)
    # coords_3d = coords_3d[:, None, ...] * hypo_depths  # (3, D, H, W)
    fx = fx_unit * width
    fy = fy_unit * height
    hypo_depths = torch.linspace(0, depth_num-1, depth_num).type_as(sdf_volume)
    hypo_depths[0] += 1e-4  # precision issue
    hypo_depths = fx * baseline / hypo_depths  # convert disparity to depth

    z_diffs = torch.abs(hypo_depths[:-1] - hypo_depths[1:])  # (D-1, )
    hypo_xs = 1 / fx * hypo_depths  # (D, )
    hypo_ys = 1 / fy * hypo_depths  # (D, )

    grad_x = torch.abs(sdf_volume[:, :, :, :-1] - sdf_volume[:, :, :, 1:]) / hypo_xs[None, :, None, None]  # (B, D, H, W-1)
    grad_y = torch.abs(sdf_volume[:, :, :-1, :] - sdf_volume[:, :, 1:, :]) / hypo_ys[None, :, None, None]  # (B, D, H-1, W)
    grad_z = torch.abs(sdf_volume[:, :-1, :, :] - sdf_volume[:, 1:, :, :]) / z_diffs[None, :, None, None]  # (B, D-1, H, W)

    sdf_gradient = torch.stack([grad_x[:, :depth_num-1, :height-1, :width-1],
                                grad_y[:, :depth_num-1, :height-1, :width-1],
                                grad_z[:, :depth_num-1, :height-1, :width-1]], dim=1)  # (B, 3, D-1, H-1, W-1)

    sdf_gradient_norm = torch.norm(sdf_gradient, dim=1)  # (B, D-1, H-1, W-1)

    return sdf_gradient_norm




    




if __name__=="__main__":
    sample = torch.rand(2,24,40,80)
    sdf_gradient_norm = SDF2Graident(sample)
    print(sdf_gradient_norm.shape)
    eki_loss = ((sdf_gradient_norm - torch.ones_like(sdf_gradient_norm))**2).mean()
    print(eki_loss)
