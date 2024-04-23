import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from losses.metirc import *


def L1Loss_without_mean(input, target):
    return (input - target).abs()

def L1Loss(input, target):
    return (input - target).abs().mean()

def loss_disp_unsupervised(img_left, img_right, disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped = warp_disp(img_right, -disp)

    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
    loss = 0.15 * L1Loss(image_warped * valid_mask, img_left * valid_mask) + \
           0.85 * (valid_mask * (1 - ssim(img_left, image_warped)) / 2).mean()
    return loss

# left-left and right-right
def loss_disp_unsupervised_multi(img_left, 
                                 img_right, 
                                 img_left_left,
                                 img_right_right,
                                 disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_right_right,-2*disp)
    image_warped_left3 = warp_disp(img_left_left,disp)
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   
    loss_2 = 0.15 * L1Loss_without_mean(image_warped_left3*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left3)) / 2)
                   
    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)
    loss_2_abs = torch.sum(loss_2,dim=1,keepdim=True)

    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,loss_1_abs,loss_2_abs], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_2 = torch.where(min_tensor_idx == 2, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    
    mask_0 = mask_0.type_as(img_left)
    mask_1 = mask_1.type_as(img_left)
    mask_2 = mask_2.type_as(img_left)
    mask_0= mask_0.float()
    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    mask = mask_0 + mask_1 + mask_2
    loss = loss_0 * mask_0 + loss_1 * mask_1  + loss_2 * mask_2
    loss = loss.mean()
                   
    return loss

def loss_disp_smoothness(disp, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
            ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss

def loss_disp_unsupervised_center(img_left, 
                                 img_right, 
                                 img_med,
                                 disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_med,-0.5*disp)
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   

    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)

    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,loss_1_abs], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
   
    
    mask_0 = mask_0.type_as(img_left)
    mask_1 = mask_1.type_as(img_left)

    mask_0= mask_0.float()
    mask_1 = mask_1.float()

    mask = mask_0 + mask_1 
    loss = loss_0 * mask_0 + loss_1 * mask_1 
    loss = loss.mean()
                   
    return loss

def loss_disp_unsupervised_center_outside(
                                 img_left, 
                                 img_right, 
                                 img_med,
                                 img_left_left,
                                 img_right_right,
                                 disp, 
                                 valid_mask=None, 
                                 mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_med,-0.5*disp)
    
    image_warped_left3 = warp_disp(img_right_right,-2*disp)
    image_warped_left4 = warp_disp(img_left_left,disp)
    
    
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   
    loss_2 = 0.15 * L1Loss_without_mean(image_warped_left3*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left3)) / 2)
                   
    loss_3 = 0.15 * L1Loss_without_mean(image_warped_left4*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left4)) / 2)          
    
                   

    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)
    loss_2_abs = torch.sum(loss_2,dim=1,keepdim=True)
    loss_3_abs = torch.sum(loss_3,dim=1,keepdim=True)
    
    

    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,
                                             loss_1_abs,
                                             loss_2_abs,
                                             loss_3_abs                                             
                                             ], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    
    mask_2 = torch.where(min_tensor_idx == 2, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_3 = torch.where(min_tensor_idx == 3, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
   
    
    mask_0 = mask_0.type_as(img_left)
    mask_1 = mask_1.type_as(img_left)
    
    mask_2 = mask_2.type_as(img_left)
    mask_3 = mask_3.type_as(img_left)
    
    

    mask_0= mask_0.float()
    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    mask_3 = mask_3.float()
    

    mask = mask_0 + mask_1 + mask_2 + mask_3
    
    
    loss = loss_0 * mask_0 + loss_1 * mask_1 + loss_2 * mask_2 + loss_3 * mask_3 
    loss = loss.mean()
                   
    return loss






def Simple_MultiScaleLoss(weights,disp_pyramid,left_img,right_img):
    total_loss = 0
    for idx, disp in enumerate(disp_pyramid):
        supervised_loss = loss_disp_unsupervised(left_img,right_img,disp)
        smooth_loss = loss_disp_smoothness(disp,left_img)
        cur_loss = supervised_loss *1.0 + smooth_loss*0.1
        
        total_loss+=cur_loss*weights[idx]
    
    return total_loss


def MultiScaleLoss_Plus_Center(weights,
                                disp_pyramid,
                                left_img,
                                right_img,
                                med_img,
                                ):
    total_loss = 0
    for idx, disp in enumerate(disp_pyramid):
        supervised_loss = loss_disp_unsupervised_center(img_left=left_img,
                                                       img_right=right_img,
                                                       img_med=med_img,
                                                       disp=disp)
        
        smooth_loss = loss_disp_smoothness(disp,left_img)
        cur_loss = supervised_loss *1.0 + smooth_loss*0.1
        
        total_loss+=cur_loss*weights[idx]
    
    return total_loss


def MultiScaleLoss_Plus_Outside(weights,
                                disp_pyramid,
                                left_img,
                                right_img,
                                left_left,
                                right_right
                                ):
    total_loss = 0
    for idx, disp in enumerate(disp_pyramid):
        supervised_loss = loss_disp_unsupervised_multi(img_left=left_img,
                                                       img_right=right_img,
                                                       img_left_left=left_left,
                                                       img_right_right=right_right,
                                                       disp=disp)
        
        smooth_loss = loss_disp_smoothness(disp,left_img)
        cur_loss = supervised_loss *1.0 + smooth_loss*0.1
        
        total_loss+=cur_loss*weights[idx]
    
    return total_loss


def MultiScaleLoss_Plus_Outside_Center(weights,
                                disp_pyramid,
                                left_img,
                                right_img,
                                left_left,
                                right_right,
                                img_center,
                                ):
    total_loss = 0
    for idx, disp in enumerate(disp_pyramid):
        supervised_loss = loss_disp_unsupervised_center_outside(img_left=left_img,
                                                                img_right=right_img,
                                                                img_med=img_center,
                                                                img_left_left=left_left,
                                                                img_right_right=right_right,
                                                                disp=disp
                                                                )
        
        smooth_loss = loss_disp_smoothness(disp,left_img)
        cur_loss = supervised_loss *1.0 + smooth_loss*0.1
        
        total_loss+=cur_loss*weights[idx]
    
    return total_loss




def warp_disp(img, disp):
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

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

def ssim(img1, img2, window_size=11):
    _, channel, h, w = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)