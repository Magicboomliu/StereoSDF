import torch.nn as nn
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
from models.PAMStereo.pam_utils import *
from models.PAMStereo.modules import *
from models.Stereonet.render import NeuSRenderer, DispWarper


def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, s, d, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


def conv_1x1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_3x3(c0, c0, d=dilation),
            conv_3x3(c0, c0, d=dilation),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input


class ResBlock1x1(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_1x1(c0, c0),
            conv_1x1(c0, c0),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input


class PASMnetSDFRender(nn.Module):
    def __init__(self,sdf_type='MLP',max_disp=192, use_sdf_render=False):
        super(PASMnetSDFRender, self).__init__()
        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################
        self.sdf_type = sdf_type
        self.use_sdf_render = use_sdf_render

        self.max_disp = max_disp

        # Feature Extraction
        self.hourglass = Hourglass([32, 64, 96, 128, 160])

        # Cascaded Parallax-Attention Module
        self.cas_pam = CascadedPAM([128, 96, 64])

        # Output Module
        self.output = Output()

        # Disparity Refinement
        self.refine = Refinement([64, 96, 128, 160, 160, 128, 96, 64, 32, 16])

        if self.sdf_type=='2D_conv':
            self.sdf_filter = nn.Sequential(
                ResBlock(self.max_disp//4),
                ResBlock(self.max_disp//4),
                nn.Conv2d(self.max_disp//4,self.max_disp//4,kernel_size=3,stride=1,padding=1)
            )
        elif self.sdf_type =="MLP":
            self.sdf_filter = nn.Sequential(
                ResBlock1x1(self.max_disp//4),
                ResBlock1x1(self.max_disp//4),
                nn.Conv2d(self.max_disp//4,self.max_disp//4,kernel_size=3,stride=1,padding=1)
            )

        if self.use_sdf_render:
            self.render = NeuSRenderer()

    def forward(self, x_left, x_right, max_disp=192):
        # input left and right images
        b, _, h, w = x_left.shape
        # Feature Extraction: 1/16, 1/8 , 1/4 from decoder, 1/4 from encoder
        (fea_left_s1, fea_left_s2, fea_left_s3), fea_refine = self.hourglass(x_left)
        (fea_right_s1, fea_right_s2, fea_right_s3), _       = self.hourglass(x_right)

        # Cascaded Parallax-Attention Module
        cost_s1, cost_s2, cost_s3 = self.cas_pam([fea_left_s1, fea_left_s2, fea_left_s3],
                                                 [fea_right_s1, fea_right_s2, fea_right_s3])

        # Output Module
        if self.training:
            disp_s1, att_s1, att_cycle_s1, valid_mask_s1 = self.output(cost_s1, max_disp // 16)
            disp_s2, att_s2, att_cycle_s2, valid_mask_s2 = self.output(cost_s2, max_disp // 8)
            disp_s3, att_s3, att_cycle_s3, valid_mask_s3 = self.output(cost_s3, max_disp // 4)
        else:
            disp_s3 = self.output(cost_s3, max_disp // 4)

        # Disparity SDF
        # First Get the Cost Volime
        disp_cross_attn, occ_cross_attn = cost_s3
        disp_cross_attn = torch.tril(disp_cross_attn) # [B,H,W,W]
        disp_cross_attn = disp_cross_attn.permute(0,3,1,2)

        cur_h = disp_s3.shape[-2]
        cur_w = disp_s3.shape[-1]
        # get the gather index.
        disparity_candidates_index = torch.arange(max_disp//4).view(1,max_disp//4,1,1).type_as(disp_s3) #[1,D,1,1]
        x_range = torch.arange(0, cur_w).view(1, 1, cur_w).expand(1, cur_h, cur_w).unsqueeze(1).type_as(disp_s3)  # [1, 1,H, W]

        x_range = x_range.repeat(disp_s3.shape[0],1,1,1)

        right_sample_point = x_range - disparity_candidates_index
        valid_sample = right_sample_point>=0
        valid_sample = valid_sample.float()
        abs_right_sample_point = torch.clamp(right_sample_point,min=0,max=cur_w)

        sampled_cost_volume = torch.gather(disp_cross_attn,dim=1,index=abs_right_sample_point.long())
        sampled_cost_volume = sampled_cost_volume * valid_sample

        if self.sdf_type in ["2D_conv","MLP"]:
            est_sdf = self.sdf_filter(sampled_cost_volume)

        rendered_left_depth = None
        right_14 = F.interpolate(x_right, scale_factor=1/4, mode='bilinear', align_corners=False)
        d = torch.arange(0, max_disp // 4, device=right_14.device, dtype=right_14.dtype)
        warper = DispWarper(image_size=right_14.shape[2:], disp_range=d, device=right_14.device)  # FIXME: Added!!!
        color_grid = warper.get_warped_frame(right_14, direct=-1)
        rendered_left, weights_sum, rendered_left_disparity, warped_left = self.render(est_sdf, color_grid, right_14)

        # Disparity Refinement: A sample layer with initial 1/4 level feature as the guidance.
        disp = self.refine(fea_refine, disp_s3)

        if self.training:
            return disp, \
                   [att_s1, att_s2, att_s3], \
                   [att_cycle_s1, att_cycle_s2, att_cycle_s3], \
                   [valid_mask_s1, valid_mask_s2, valid_mask_s3],est_sdf, \
                   rendered_left, weights_sum, rendered_left_disparity, warped_left

        else:
            return disp


# Hourglass Module for Feature Extraction
class Hourglass(nn.Module):
    def __init__(self, channels):
        super(Hourglass, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.E0 = EncoderB(1,           3, channels[0], downsample=True)               # scale: 1/2
        self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)               # scale: 1/4
        self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)               # scale: 1/8
        self.E3 = EncoderB(1, channels[2], channels[3], downsample=True)               # scale: 1/16
        self.E4 = EncoderB(1, channels[3], channels[4], downsample=True)               # scale: 1/32

        self.D0 = EncoderB(1, channels[4], channels[4], downsample=False)              # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[3], channels[3])                  # scale: 1/16
        self.D2 = DecoderB(1, channels[3] + channels[2], channels[2])                  # scale: 1/8
        self.D3 = DecoderB(1, channels[2] + channels[1], channels[1])                  # scale: 1/4

    def forward(self, x):
        # return upsample feature 1/16,1/8,1/4 and initial 1/4
        fea_E0 = self.E0(x)                                                            # scale: 1/2
        fea_E1 = self.E1(fea_E0)                                                       # scale: 1/4
        fea_E2 = self.E2(fea_E1)                                                       # scale: 1/8
        fea_E3 = self.E3(fea_E2)                                                       # scale: 1/16
        fea_E4 = self.E4(fea_E3)                                                       # scale: 1/32

        fea_D0 = self.D0(fea_E4)                                                       # scale: 1/32
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E3), 1))                # scale: 1/16
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E2), 1))                # scale: 1/8
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E1), 1))                # scale: 1/4

        return (fea_D1, fea_D2, fea_D3), fea_E1


# Cascaded Parallax-Attention Module
class CascadedPAM(nn.Module):
    def __init__(self, channels):
        super(CascadedPAM, self).__init__()
        self.stage1 = PAM_stage(channels[0])
        self.stage2 = PAM_stage(channels[1])
        self.stage3 = PAM_stage(channels[2])

        # bottleneck in stage 2
        self.b2 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, 1, 1, 0, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # bottleneck in stage 3
        self.b3 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, 1, 1, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, fea_left, fea_right):
        '''
        :param fea_left:    feature list [fea_left_s1, fea_left_s2, fea_left_s3]
        :param fea_right:   feature list [fea_right_s1, fea_right_s2, fea_right_s3]
        '''
        fea_left_s1, fea_left_s2, fea_left_s3 = fea_left
        fea_right_s1, fea_right_s2, fea_right_s3 = fea_right

        # 1/16 level
        b, _, h_s1, w_s1 = fea_left_s1.shape

        # 1/8 level
        b, _, h_s2, w_s2 = fea_left_s2.shape

        # stage 1: 1/16
        cost_s0 = [
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device),
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device)
        ]

        fea_left, fea_right, cost_s1 = self.stage1(fea_left_s1, fea_right_s1, cost_s0)

        # stage 2: 1/8
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b2(torch.cat((fea_left, fea_left_s2), 1))
        fea_right = self.b2(torch.cat((fea_right, fea_right_s2), 1))

        cost_s1_up = [
            F.interpolate(cost_s1[0].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s1[1].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s2 = self.stage2(fea_left, fea_right, cost_s1_up)

        # stage 3: 1/4
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b3(torch.cat((fea_left, fea_left_s3), 1))
        fea_right = self.b3(torch.cat((fea_right, fea_right_s3), 1))

        cost_s2_up = [
            F.interpolate(cost_s2[0].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s2[1].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        # Final 1/4 level
        fea_left, fea_right, cost_s3 = self.stage3(fea_left, fea_right, cost_s2_up)

        # return 1/16, 1/8, 1/4 level disparity cross attention volume.
        return [cost_s1, cost_s2, cost_s3]


class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right, cost):
        # update the left-right / right-left cost volume 4 times.
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        # return the left feature, right feature, and aggregated cost volume list.
        return fea_left, fea_right, cost


# Disparity Refinement Module
class Refinement(nn.Module):
    def __init__(self, channels):
        super(Refinement, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)

        self.E0 = EncoderB(1, channels[0] + 1, channels[0], downsample=False)   # scale: 1/4
        self.E1 = EncoderB(1, channels[0],     channels[1], downsample=True)    # scale: 1/8
        self.E2 = EncoderB(1, channels[1],     channels[2], downsample=True)    # scale: 1/16
        self.E3 = EncoderB(1, channels[2],     channels[3], downsample=True)    # scale: 1/32

        self.D0 = EncoderB(1, channels[4],     channels[4], downsample=False)   # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[5], channels[5])           # scale: 1/16
        self.D2 = DecoderB(1, channels[5] + channels[6], channels[6])           # scale: 1/8
        self.D3 = DecoderB(1, channels[6] + channels[7], channels[7])           # scale: 1/4
        self.D4 = DecoderB(1, channels[7],               channels[8])           # scale: 1/2
        self.D5 = DecoderB(1, channels[8],               channels[9])           # scale: 1

        # regression
        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
        )

    def forward(self, fea, disp):
        # scale the input disparity
        disp = disp / (2 ** 5)

        fea_E0 = self.E0(torch.cat((disp, fea), 1))                         # scale: 1/4
        fea_E1 = self.E1(fea_E0)                                            # scale: 1/8
        fea_E2 = self.E2(fea_E1)                                            # scale: 1/16
        fea_E3 = self.E3(fea_E2)                                            # scale: 1/32

        fea_D0 = self.D0(fea_E3)                                            # scale: 1/32
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E2), 1))     # scale: 1/16
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E1), 1))     # scale: 1/8
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E0), 1))     # scale: 1/4
        fea_D4 = self.D4(self.upsample(fea_D3))                             # scale: 1/2
        fea_D5 = self.D5(self.upsample(fea_D4))                             # scale: 1

        # regression
        confidence = self.confidence(fea_D5)
        disp_res = self.disp(fea_D5)
        disp_res = torch.clamp(disp_res, 0)

        disp = F.interpolate(disp, scale_factor=4, mode='bilinear') * (1-confidence) + disp_res * confidence

        return disp * 2 ** 7


if __name__=="__main__":

    left_sample = torch.randn(1,3,320,640).cuda()
    right_sample = torch.randn(1,3,320,640).cuda()
    target_smaple = torch.randn(1,1,320,640).cuda()

    pasmet = PASMnetSDFRender(use_sdf_render=True).cuda()

    disp,attn_list,att_cycle,valid_mask,est_sdf,rendered_left, weights_sum = pasmet(left_sample,right_sample)
    print(rendered_left.shape)
    print(weights_sum.shape)
