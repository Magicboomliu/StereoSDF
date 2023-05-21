import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.preprocess import scale_disp

from utils.AverageMeter import AverageMeter
from utils.common import logger, check_path, write_pfm,count_parameters

from models.Stereonet.stereonet import StereoNet
from models.Stereonet.stereonet_sdf import StereoNetSDF
from dataloader.kitti_loader import StereoDataset
from dataloader import kitti_transform

# metric
from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss
from utils.metric import compute_iou,Occlusion_EPE


import time
import os
from torch.autograd import Variable

from losses.unsupervised_loss import loss_disp_unsupervised,MultiScaleLoss,Eki_Loss


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
        self.model = model
        self.initialize()
        self.wandb = wandb
        # additional parameters
        self.summary_freq = kwargs['opt']['summary_freq']
        self.sdf_weight = kwargs['opt']['sdf_weight']
    
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

        datathread=4
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        logger.info("Use %d processes to load data..." % datathread)

        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
                                shuffle = True, num_workers = datathread, \
                                pin_memory = True)

        self.test_loader = DataLoader(test_dataset, batch_size = self.test_batch, \
                                shuffle = False, num_workers = datathread, \
                                pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)

    def _build_net(self):
        # Build the Network architecture according to the model name
        if self.model == 'StereoNet':
            self.net = StereoNet()
        elif self.model=='StereoNetSDF':
            self.net = StereoNetSDF(sdf_type='2D_conv')
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
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()


    def adjust_learning_rate(self, epoch):
        if epoch>=0 and epoch<=10:
            cur_lr = 3e-4
        elif epoch > 10 and epoch<45:
            cur_lr = 1e-4
        elif epoch>=40 and epoch<50:
            cur_lr = 5e-5
        elif epoch>=50 and epoch<60:
            cur_lr = 3e-5
        elif epoch>=60:
            cur_lr =1.5e-5
        else:
            cur_lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr


    def train_one_epoch(self, epoch, round, iterations, summary_writer):
        
        # Data Summary
        batch_time = AverageMeter()
        data_time = AverageMeter()    
        losses = AverageMeter()
        eki_loss_meter = AverageMeter()

        nums_samples = len(self.train_loader)
        train_count = 0
        
        # non-detection
        torch.autograd.set_detect_anomaly(True)
        
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
        summary_writer.add_scalar("Learning_Rate",cur_lr,epoch+1)

        for i_batch, sample_batched in enumerate(self.train_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            
            
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            
            if self.model =="StereoNet":
                pyramid_disp = self.net(left_input,right_input)["multi_scale"]
                output = pyramid_disp[-1]
            elif self.model=="StereoNetSDF":
                output_staff = self.net(left_input,right_input)
                pyramid_disp = output_staff["multi_scale"]
                est_sdf = output_staff['sdf']
            
            if self.model=='StereoNet':
                # Loss Here
                loss = MultiScaleLoss(weights=[0.8,1.0,1.0],disp_pyramid=pyramid_disp,
                                  left_img=left_input,right_img=right_input)
            
            elif self.model=='StereoNetSDF':
                photo_loss = MultiScaleLoss(weights=[0.8,1.0,1.0],disp_pyramid=pyramid_disp,
                                  left_img=left_input,right_img=right_input)
                sdf_loss = Eki_Loss(est_sdf=est_sdf)
                
                # FIXME : beta 
                # beta = 0.01
                loss = photo_loss + sdf_loss * self.sdf_weight
                
                eki_loss_meter.update(sdf_loss.data.item(),left_input.size(0))
                summary_writer.add_scalar("eki_loss",eki_loss_meter.val,iterations+1)
                
            
            if type(loss) is list or type(loss) is tuple:
                
                loss = np.sum(loss)
            
            losses.update(loss.data.item(), left_input.size(0))
            summary_writer.add_scalar("total_loss",losses.val,iterations+1)

            # compute gradient and do SGD step
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()
            iterations = iterations+1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            if self.model=='StereoNet':
                if i_batch % self.summary_freq == 0:
                    logger.info('this is round %d', round)
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                    epoch, i_batch, self.num_batches_per_epoch, 
                    batch_time=batch_time,
                    data_time=data_time, loss=losses))
            elif self.model=='StereoNetSDF':
                if i_batch % self.summary_freq == 0:
                    logger.info('this is round %d', round)
                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Sdf_loss {sdf_losses.val:.3f} ({sdf_losses.avg:.3f})\t'
                    'Loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                    epoch, i_batch, self.num_batches_per_epoch, 
                    batch_time=batch_time,
                    sdf_losses = eki_loss_meter,
                    data_time=data_time, loss=losses))
            
        # update training logs
        self.wandb.log('photometric_loss': photo_loss.data.cpu().numpy())
        self.wandb.log('eikonal_loss': (sdf_loss * self.sdf_weight).data.cpu().numpy())
        self.wandb.log('learning_rate': cur_lr)

        return losses.avg, losses.avg, iterations

    def validate(self,summary_writer,epoch,vis=False):
        
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        P1_errors = AverageMeter()    
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
                if self.model=="StereoNet":
                    output = self.net(left_input_pad,right_input_pad)['disp']
                elif self.model =='StereoNetSDF':
                    output = self.net(left_input_pad,right_input_pad)['disp']

                output = output[:,:,h_pad:,w_pad:]
                assert output.shape ==target_disp.shape

                
                inference_time += time.perf_counter() - start_time
                img_nums += left_input.shape[0]
                

                flow2_EPE = self.epe(output, target_disp)
                P1_error = self.p1_error(output, target_disp)
                
                
            if flow2_EPE.data.item() == flow2_EPE.data.item():
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if P1_error.data.item() == P1_error.data.item():
                P1_errors.update(P1_error.data.item(), input_var.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.summary_freq == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val))
            
            
        logger.info(' * DISP EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * P1_error {:.3f}'.format(P1_errors.avg))
        
        
        logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))

        self.wandb.log('val_epe': flow2_EPEs.avg)
        self.wandb.log('val_p1': P1_error.avg)

        return flow2_EPEs.avg
        

    def get_model(self):
        return self.net.state_dict()

