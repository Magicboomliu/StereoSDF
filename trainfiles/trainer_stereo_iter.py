import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.preprocess import scale_disp
import matplotlib.pyplot as plt

from utils.AverageMeter import AverageMeter
from utils.common import logger, check_path, write_pfm,count_parameters

from models.Stereonet.stereonet import StereoNet
from models.Stereonet.stereonet_sdf import StereoNetSDF
from models.Stereonet.stereonet_sdf_render import StereoNetSDFRender
from models.Stereonet.stereonet_new_sdf_render import StereoNetNewSDFRender
from models.PAMStereo.PASMnet import PASMnet
from models.PAMStereo.PASMNet_SDF import PASMnetSDF
from models.PAMStereo.PAMNet_SDF_render import PASMnetSDFRender
from dataloader.kitti_loader import StereoDataset
import cv2

from dataloader import kitti_transform

# metric
from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss
from utils.metric import compute_iou,Occlusion_EPE

import time
import os
from torch.autograd import Variable

from losses.unsupervised_loss import loss_disp_unsupervised,MultiScaleLoss,Eki_Loss
from losses.pam_loss import PAMStereoLoss, loss_ssim_l1, loss_disp_smoothness


# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DisparityTrainer(object):
    def __init__(self, lr, devices, dataset, trainlist, vallist, datapath, 
                 batch_size, maxdisp,use_deform=False, pretrain=None, 
                        model='StereoNet', test_batch=4,initial_pretrain=None,
                        wandb=None,
                        **kwargs):
        super(DisparityTrainer, self).__init__()
        
        self.lr = lr
        self.initial_pretrain = initial_pretrain
        self.current_lr = lr
        self.devices = devices
        self.devices = [int(item) for item in devices.split(',')]
        ngpu = len(devices)
        self.ngpu = ngpu

        self.summary_freq = kwargs['opt'].summary_freq
        self.sdf_weight = kwargs['opt'].sdf_weight
        self.sdf_type = kwargs['opt'].sdf_type
        self.optimizer = kwargs['opt'].optimizer 
        self.num_steps = kwargs['opt'].num_steps
        self.weight_decay = kwargs['opt'].weight_decay
        self.val_freq = kwargs['opt'].val_freq
        self.datathread = kwargs['opt'].datathread
        self.outf = kwargs['opt'].outf
    
        self.trainlist = trainlist
        self.vallist = vallist
        self.dataset = dataset
        self.datapath = datapath
        self.batch_size = batch_size
        self.test_batch = test_batch
        self.pretrain = pretrain 
        self.maxdisp = maxdisp
        self.use_deform= use_deform
        self.criterion = None
        self.epe = Disparity_EPE_Loss
        self.p1_error = P1_metric
        self.d1_error = D1_metric
        self.model = model
        self.wandb = wandb
        #assert self.wandb is not None
        self.initialize()

    
    # Get Dataset Here
    def _prepare_dataset(self):
        # KITTI MIX
        if self.dataset =='KITTI':
            train_transform_list = [kitti_transform.RandomCrop(320, 960),
                                    kitti_transform.ToTensor(),
                                    kitti_transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
            train_transform = kitti_transform.Compose(train_transform_list)
            val_transform_list = [kitti_transform.ToTensor(),
                                kitti_transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

            val_transform = kitti_transform.Compose(val_transform_list)
            test_dataset = StereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,
                                        test_datalist=self.vallist,
                                dataset_name='KITTI_mix',mode='test',transform=val_transform)
        
            train_dataset = StereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='KITTI_mix',mode='train',transform=train_transform)
            
        self.img_height, self.img_width = train_dataset.get_img_size()

        self.scale_height, self.scale_width = test_dataset.get_scale_size()

        logger.info("Use %d processes to load data..." % self.datathread)

        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
                                shuffle = True, num_workers = self.datathread, \
                                pin_memory = True)

        self.test_loader = DataLoader(test_dataset, batch_size = self.test_batch, \
                                shuffle = False, num_workers = self.datathread, \
                                pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)

    def _build_net(self):
        # Build the Network architecture according to the model name
        if self.model == 'StereoNet':
            self.net = StereoNet()
        elif self.model=='StereoNetSDF':
            self.net = StereoNetSDF(sdf_type=self.sdf_type)
        elif self.model == 'StereoNetSDFRender':
            self.net = StereoNetSDFRender(self.batch_size, sdf_type=self.sdf_type, use_sdf_render=True)
        elif self.model == 'StereoNetNewSDFRender':
            self.net = StereoNetNewSDFRender(self.batch_size, sdf_type=self.sdf_type, use_sdf_render=True)
        elif self.model == "PAM":
            self.net = PASMnet()
        elif self.model =="PAMSDF":
            self.net = PASMnetSDF(sdf_type='MLP',max_disp=192)
        elif self.model == 'PAMSDFRender':
            self.net = PASMnetSDFRender(sdf_type='MLP',max_disp=192,use_sdf_render=True)
        else:
            raise NotImplementedError
        
        self.is_pretrain = False
        if self.ngpu > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        else:
            # self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.net.parameters()])))

        if self.pretrain == 'none':
            logger.info('Initial a new model...')
            if self.initial_pretrain !='none':
                pretrain_ckpt = self.initial_pretrain
                print("Loading the Model with Some initial Weights........")
                ckpt = torch.load(pretrain_ckpt)
                current_model_dict = self.net.state_dict()
                useful_dict ={k:v for k,v in ckpt['state_dict'].items() if k in current_model_dict.keys()}
                print("{}/{} has been re-used in this training".format(len(useful_dict) ,len(ckpt['state_dict'])))
                current_model_dict.update(useful_dict)
                self.net.load_state_dict(current_model_dict)
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)
                
    def _build_optimizer(self):
        if self.optimizer == 'Adam':
            print('load Adam as optimizer')
            beta = 0.999
            momentum = 0.9
            last_epoch = -1
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, self.lr,
                self.num_steps + 10,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy='cos',
                last_epoch=last_epoch,
            )
        elif self.optimizer == 'AdamW':
            print('load AdamW as optimizer')
            # this is a hardcode here
            last_epoch = -1
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                            weight_decay=self.weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, self.lr,
                self.num_steps + 10,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy='cos',
                last_epoch=last_epoch,
            )

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()

    def launch_training(self):
        
        # Data Summary
        batch_time = AverageMeter()
        data_time = AverageMeter()    
        losses = AverageMeter()
        eki_loss_meter = AverageMeter()
        render_loss_meter = AverageMeter()

        # non-detection
        torch.autograd.set_detect_anomaly(True)
        
        # switch to train mode
        self.net.train()
        end = time.time()
        sdf_loss = None
        sdf_render_loss = None
        sdf_warped_loss = None
        render_depth_smooth_loss = None

        # load trainer
        total_steps = 0
        while total_steps < self.num_steps:
            for i_batch, sample_batched in enumerate(self.train_loader):
                left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
                right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
                intrinsic_K = torch.autograd.Variable(sample_batched['K'].cuda(), requires_grad=False)

                data_time.update(time.time() - end)
                self.optimizer.zero_grad()
                
                if self.model =="StereoNet":
                    pyramid_disp = self.net(left_input,right_input)["multi_scale"]
                    output = pyramid_disp[-1]
                elif self.model=="StereoNetSDF":
                    output_staff = self.net(left_input,right_input)
                    pyramid_disp = output_staff["multi_scale"]
                    est_sdf = output_staff['sdf']
                elif self.model == 'StereoNetSDFRender':
                    output_staff = self.net(left_input,right_input)
                    pyramid_disp = output_staff["multi_scale"]
                    est_sdf = output_staff['sdf']
                    rendered_left = output_staff['rendered_left']
                    weights_sum = output_staff['weights_sum']
                elif self.model == 'StereoNetNewSDFRender':
                    output_staff = self.net(left_input, right_input, intrinsic_K, run_sdf=True)
                    pyramid_disp = output_staff["multi_scale"]
                elif self.model == "PAM":
                    output,attn_list,att_cycle,valid_mask = self.net(left_input,right_input,192)
                elif self.model =="PAMSDF":
                    output,attn_list,att_cycle,valid_mask,est_sdf= self.net(left_input,right_input,192)
                elif self.model == 'PAMSDFRender':
                    output,attn_list,att_cycle,valid_mask,est_sdf,rendered_left,weights_sum,rendered_left_disparity,warped_left= self.net(left_input,right_input,192)

                if self.model=='StereoNet':
                    # Loss Here
                    photo_loss = MultiScaleLoss(weights=[0.8,1.0,1.0],disp_pyramid=pyramid_disp,
                                    left_img=left_input,right_img=right_input)
                    loss = photo_loss

                elif self.model=='StereoNetSDF':
                    photo_loss = MultiScaleLoss(weights=[0.8,1.0,1.0],disp_pyramid=pyramid_disp,
                                    left_img=left_input,right_img=right_input)
                    sdf_loss = Eki_Loss(est_sdf=est_sdf)
                    loss = photo_loss + sdf_loss * self.sdf_weight 
                    eki_loss_meter.update(sdf_loss.data.item(),left_input.size(0))
                elif self.model == 'StereoNetSDFRender':
                    photo_loss = MultiScaleLoss(weights=[0.8,1.0,1.0],disp_pyramid=pyramid_disp,
                                    left_img=left_input,right_img=right_input)
                    sdf_loss = Eki_Loss(est_sdf=est_sdf)
                    left_18 = F.interpolate(left_input, scale_factor=1/8, mode='bilinear', align_corners=False)
                    sdf_render_loss = F.l1_loss(left_18, rendered_left, size_average=True, reduction='mean')

                    # FIXME : beta
                    # beta = 0.01
                    loss = photo_loss + sdf_loss * self.sdf_weight + sdf_render_loss * 0.1
                    
                    render_loss_meter.update(sdf_render_loss.data.item(), left_input.size(0))
                    eki_loss_meter.update(sdf_loss.data.item(),left_input.size(0))
                elif self.model == 'StereoNetNewSDFRender':
                    photo_loss = MultiScaleLoss(weights=[0.8,1.0,1.0],disp_pyramid=pyramid_disp,
                                    left_img=left_input,right_img=right_input)

                    ret_dict = output_staff['ret_dict']
                    left_ray_render = ret_dict['left_ray_render']
                    left_ray_gt = ret_dict['left_ray_gt']
                    left_ray_weight = ret_dict['left_ray_weight']
                    sdf_gradient = ret_dict['sdf_gradient']
                    b_size, n_samples_per_ray, n_ray, _, _ = sdf_gradient.shape
                    sdf_gradient = sdf_gradient[..., 0, :] # (B, N_sample, N_ray, 3)

                    # print('left_ray_render', left_ray_render.shape)
                    # print('left_ray_gt', left_ray_gt.shape)
                    # print('left_ray_weight', left_ray_weight.shape)
                    # print('sdf_gradient', sdf_gradient.shape)
                    
                    # sdf render loss
                    sdf_render_loss = torch.abs(left_ray_render - left_ray_gt).mean(dim=1)
                    sdf_render_loss = torch.mean(left_ray_weight * sdf_render_loss)

                    # sdf gradient loss
                    sdf_gradient = (torch.linalg.norm(sdf_gradient.reshape(b_size, n_samples_per_ray * n_ray, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
                    sdf_loss = sdf_gradient.sum() / (b_size * n_samples_per_ray * n_ray)

                    loss = photo_loss + sdf_loss * self.sdf_weight + sdf_render_loss

                    render_loss_meter.update(sdf_render_loss.data.item(), left_input.size(0))
                    eki_loss_meter.update(sdf_loss.data.item(),left_input.size(0))
                elif self.model =='PAM':
                    loss, loss_P, loss_S, loss_PAM = PAMStereoLoss(left_input,right_input,disp=output,att=attn_list,
                                                                att_cycle=att_cycle,valid_mask=valid_mask,disp_gt=None)
                    photo_loss = loss
                elif self.model == "PAMSDF":
                    loss, loss_P, loss_S, loss_PAM = PAMStereoLoss(left_input,right_input,disp=output,att=attn_list,
                                                                att_cycle=att_cycle,valid_mask=valid_mask,disp_gt=None)
                    sdf_loss = Eki_Loss(est_sdf=est_sdf)
                    photo_loss = loss
                    # FIXME : beta
                    # beta = 0.01
                    loss = loss + sdf_loss * self.sdf_weight
                    eki_loss_meter.update(sdf_loss.data.item(),left_input.size(0))
                elif self.model == 'PAMSDFRender':
                    loss, loss_P, loss_S, loss_PAM = PAMStereoLoss(left_input,right_input,disp=output,att=attn_list,
                                                                att_cycle=att_cycle,valid_mask=valid_mask,disp_gt=None)
                    sdf_loss = Eki_Loss(est_sdf=est_sdf)
                    photo_loss = loss
                    # FIXME : beta
                    # beta = 0.01
                    left_14 = F.interpolate(left_input, scale_factor=1/4, mode='bilinear', align_corners=False)
                    right_14 = F.interpolate(right_input, scale_factor=1/4, mode='bilinear', align_corners=False)
                    # sdf_render_loss = F.l1_loss(left_14, rendered_left, size_average=True, reduction='mean')
                    sdf_render_loss = loss_ssim_l1(left_14, rendered_left, valid_mask[-1][0])
                    # print(valid_mask[-1][0].shape)
                    # sdf_warped_loss = loss_ssim_l1(left_14, warped_left, valid_mask[-1][0])

                    # render_depth_smooth_loss = loss_disp_smoothness(rendered_left_disparity, left_14)

                    # loss = loss + sdf_loss * self.sdf_weight + sdf_render_loss * 0.1 + sdf_warped_loss * 0.1 + render_depth_smooth_loss * 1e-4
                    loss = loss + sdf_loss * self.sdf_weight + sdf_render_loss * 10

                    render_loss_meter.update(sdf_render_loss.data.item(), left_input.size(0))
                    eki_loss_meter.update(sdf_loss.data.item(),left_input.size(0))
                else:
                    raise NotImplementedError
                
                if type(loss) is list or type(loss) is tuple:
                    
                    loss = np.sum(loss)
                
                losses.update(loss.data.item(), left_input.size(0))
                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                total_steps = total_steps + 1

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()                
                
                # update training logs
                if self.wandb is not None and total_steps % self.summary_freq == 0:
                    logger.info('total step is %d, photometric loss is %.3f' % (total_steps, photo_loss.data.cpu().numpy()))
                    self.wandb.log({'photometric_loss': photo_loss.data.cpu().numpy()})
                    if sdf_loss is not None:
                        self.wandb.log({'eikonal_loss': (sdf_loss * self.sdf_weight).data.cpu().numpy()})
                    if sdf_render_loss is not None:
                        self.wandb.log({'sdf render loss': (sdf_render_loss * 0.1).data.cpu().numpy()})
                    if sdf_warped_loss is not None:
                        self.wandb.log({'sdf warped loss': (sdf_warped_loss * 0.1).data.cpu().numpy()})
                    if render_depth_smooth_loss is not None:
                        self.wandb.log({'sdf depth smooth loss': (render_depth_smooth_loss * 1e-4).data.cpu().numpy()})
                    # self.wandb.log({'learning_rate': cur_lr})
                    self.wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})

                if self.wandb is not None and total_steps % 100 * self.summary_freq == 0 and self.model == 'PAMSDFRender':
                    fig, ax = plt.subplots(nrows=4, ncols=2)

                    left_14_vis = left_14[0].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)
                    h, w, _ = left_14_vis.shape
                    ax[0, 0].imshow(cv2.resize(left_14_vis, (2*w, 2*h)))
                    ax[0, 0].set_title('Original Left')
                    rendered_left_vis = rendered_left[0].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)
                    ax[1, 0].imshow(cv2.resize(rendered_left_vis, (2*w, 2*h)))
                    ax[1, 0].set_title('Rendered Left')

                    if rendered_left_disparity is not None:
                        rendered_left_disparity_vis = rendered_left_disparity[0].permute(1, 2, 0).detach().cpu().numpy()
                        ax[0, 1].imshow(cv2.resize((rendered_left_disparity_vis / np.percentile(rendered_left_disparity_vis, 90)).clip(0.0, 1.0), (2*w, 2*h)), cmap='jet')
                        ax[0, 1].set_title('Rendered Left Disparity')

                    if warped_left is not None:
                        warped_left_vis = warped_left[0].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)
                        ax[2, 0].imshow(cv2.resize(warped_left_vis, (2*w, 2*h)))
                        ax[2, 0].set_title('Warped Left')

                    valid_mask_vis = valid_mask[-1][0][0].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)
                    ax[1, 1].imshow(cv2.resize(valid_mask_vis, (2*w, 2*h)), cmap='gray')
                    ax[1, 1].set_title('Mask')

                    right_14_vis = right_14[0].permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)
                    ax[2, 1].imshow(cv2.resize(right_14_vis, (2*w, 2*h)))
                    ax[2, 1].set_title('Original Right')

                    left_disparity_vis = output[0].permute(1, 2, 0).detach().cpu().numpy()
                    ax[3, 1].imshow(cv2.resize(left_disparity_vis / np.max(left_disparity_vis), (2*w, 2*h)), cmap='jet')
                    ax[3, 1].set_title('Left Disparity')

                    self.wandb.log({'Visulization': self.wandb.Image(fig)})

                    # fig, ax = plt.subplots()
                    # ax.imshow(rendered_left[0].permute(1, 2, 0).detach().cpu().numpy())
                    # self.wandb.log({'rendered_14': self.wandb.Image(fig)})

                    # fig, ax = plt.subplots()
                    # ax.imshow(valid_mask[-1][0][0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                    # self.wandb.log({'valid_mask': self.wandb.Image(fig)})

                # launch evaluation
                if total_steps % self.val_freq == 0:
                    val_EPE = self.validate()
                    torch.save({
                        'step': total_steps,
                        'state_dict': self.get_model(),
                        'val_epe': val_EPE,
                    }, '%s/step_%d_%.3f.pth' % (self.outf, total_steps, val_EPE))
                    self.net.train()
        
        return None

    def validate(self):
        
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        P1_errors = AverageMeter()
        D1_errors = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        end = time.time()
        inference_time = 0
        img_nums = 0
        for i, sample_batched in enumerate(self.test_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            input = torch.cat((left_input, right_input), 1)
            input_var = torch.autograd.Variable(input, requires_grad=False)
            target_disp = sample_batched['gt_disp'].unsqueeze(1)
            target_disp = target_disp.cuda()
            target_disp = torch.autograd.Variable(target_disp, requires_grad=False)
            
            # pading the left and right input image as well as the gt_disparity and pseudo left/right disparity
            b,c,h,w = left_input.shape
            w_pad = 1280 - w
            h_pad = 384 -h
            pad = (w_pad,0,
                   h_pad,0)
            left_input_pad = F.pad(left_input,pad=pad)
            right_input_pad = F.pad(right_input,pad=pad)


            with torch.no_grad():
                
                start_time = time.perf_counter()
                # Get the predicted disparity
                if self.model in ["StereoNet","StereoNetSDF","StereoNetSDFRender"]:
                    output = self.net(left_input_pad,right_input_pad)['disp']

                elif self.model in ["PAM","PAMSDF","PAMSDFRender"]:
                    output = self.net(left_input_pad,right_input_pad)
                
                elif self.model in ["StereoNetNewSDFRender"]:
                    output = self.net(left_input_pad,right_input_pad, None, False)['disp']

                output = output[:,:,h_pad:,w_pad:]
                assert output.shape ==target_disp.shape

                
                inference_time += time.perf_counter() - start_time
                img_nums += left_input.shape[0]
                

                flow2_EPE = self.epe(output, target_disp)
                P1_error = self.p1_error(output, target_disp)
                D1_error = self.d1_error(output, target_disp)
                
                
            if flow2_EPE.data.item() == flow2_EPE.data.item():
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if P1_error.data.item() == P1_error.data.item():
                P1_errors.update(P1_error.data.item(), input_var.size(0))
            if D1_error.data.item() == D1_error.data.item():
                D1_errors.update(D1_error.data.item(), input_var.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.summary_freq == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val))
            
            
        logger.info(' * DISP EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * P1_error {:.3f}'.format(P1_errors.avg))
        logger.info(' * D1_error {:.3f}'.format(D1_errors.avg))
        
        logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))

        if self.wandb is not None:
            self.wandb.log({'val_epe': flow2_EPEs.avg})
            self.wandb.log({'val_p1': P1_errors.avg})
            self.wandb.log({'val_d1': D1_errors.avg})

        return flow2_EPEs.avg
        

    def get_model(self):
        return self.net.state_dict()

