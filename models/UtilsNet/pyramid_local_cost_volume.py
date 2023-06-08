import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid


def build_local_cost_volume_based_on_feature(left_feat,right_feat,cur_disp,searching_radius,sample_nums=None):
    
    W_C = left_feat.shape[-1]
    W_D = cur_disp.shape[-1]

    scale = W_C//W_D
    if scale!=1:
        cur_disp = F.interpolate(cur_disp,scale_factor=scale,mode='bilinear',align_corners=False) * scale
    
    if sample_nums==None:
        sample_nums = 2 * searching_radius
    
    lower_bound = cur_disp - searching_radius
    upper_bound = cur_disp + searching_radius
    sample_intervals = (upper_bound-lower_bound) *1.0/(sample_nums)    
    addition_summation = torch.arange(sample_nums+1).type_as(cur_disp)
    addition_summation=addition_summation.view(1,sample_nums+1,1,1)
    sampling_candiate_intervals = addition_summation * sample_intervals
    
    sampling_candidates =lower_bound + sampling_candiate_intervals

    # possible disparities: should bigger than zero and smaller than w
    sample_candidate_ceil = ste_ceil(sampling_candidates)
    sample_candidate_floor = ste_floor(sampling_candidates)

    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=W_C-1)
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=W_C-1)


    
    sample_candidates = sample_candidate_ceil.shape[1]
    print(sample_candidates)


    # grid = meshgrid(left_feat)  # [B, 2, H, W] in image scale
    # # Note that -disp here
    # offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    # sample_grid = grid + offset
    # sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    # warped_feature = F.grid_sample(right_feat, sample_grid, mode='bilinear', padding_mode='border')


    print(sample_candidate_ceil.shape)
    print(sample_candidate_floor.shape)




def build_local_cost_volume_fixed(cost_volume,cur_disp,searching_radius,sample_nums):
    
    # Cost Volume Shape
    B,D,H,W = cost_volume.shape
    
    # Get sample candidates
    lower_bound = cur_disp - searching_radius
    upper_bound = cur_disp + searching_radius
    sample_intervals = (upper_bound-lower_bound) *1.0/(sample_nums)    
    addition_summation = torch.arange(sample_nums+1).type_as(cur_disp)
    addition_summation=addition_summation.view(1,sample_nums+1,1,1)
    sampling_candiate_intervals = addition_summation * sample_intervals
    sampling_candidates =lower_bound + sampling_candiate_intervals
    
    # valid mask
    sample_candidate_ceil = ste_ceil(sampling_candidates)
    sample_candidate_floor = ste_floor(sampling_candidates)
    
    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=D-1)
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=D-1)
    
    # Linear interplotation
    floor_rate =(sample_candidate_ceil- sampling_candidates)
    ceil_rate = 1.0 - floor_rate
    
    ceil_volume = torch.gather(cost_volume,dim=1,index=sample_candidate_ceil.long())
    floor_volume = torch.gather(cost_volume,dim=1,index=sample_candidate_floor.long())
    
    final_volume = ceil_volume*ceil_rate+ floor_volume*floor_rate
    
    return final_volume


class PyrmaidCostVolume(nn.Module):
    def __init__(self,radius,nums_levels,
                 sample_points):
        super(PyrmaidCostVolume,self).__init__()
        self.radius = radius
        self.nums_levels = nums_levels
        self.sample_points = sample_points
        
    
    def forward(self,cost_volume,radius,cur_disp):
        
        # Get the Cost Volume.
        cost_volume_pyramid = []
        cost_volume_pyramid.append(cost_volume)
        # from full searching range to 1/2 searching range.
        for i in range(self.nums_levels-1):
            B,D,H,W = cost_volume.shape
            cost_volume = cost_volume.view(B,D,-1).permute(0,2,1)
            cost_volume = F.avg_pool1d(cost_volume,2,stride=2)
            cost_volume = cost_volume.permute(0,2,1).contiguous().view(B,D//2,H,W)
            cost_volume_pyramid.append(cost_volume)
        
        # Index the Cost Volume.
        
        out_pyramid = []
        for i in range(self.nums_levels):
            corr = cost_volume_pyramid[i]
            ref_disp = cur_disp*1.0 /(2**i)
            local_cost_volume = build_local_cost_volume_fixed(corr,ref_disp,radius,self.sample_points)
            out_pyramid.append(local_cost_volume)
        
        out = torch.cat(out_pyramid,dim=1)
    
        return out


if __name__=="__main__":
    feat1 = torch.randn(1,64,100,100)
    feat2 = torch.randn(1,64,100,100)
    raduius = 3
    cur_disparity = torch.abs(torch.randn(1,1,100,100)*10-6)
    cur_disparity = torch.clamp(cur_disparity,min=0,max=99)

    build_local_cost_volume_based_on_feature(feat1,feat2,cur_disp=cur_disparity,searching_radius=3)