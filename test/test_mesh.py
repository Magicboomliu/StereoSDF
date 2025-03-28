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
from models.PAMStereo.PASMnet import PASMnet
from models.PAMStereo.PASMNet_SDF import PASMnetSDF
from models.PAMStereo.PAMNet_SDF_render import PASMnetSDFRender
from dataloader.kitti_loader import StereoDataset

from dataloader import kitti_transform

# metric
from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss
from utils.metric import compute_iou,Occlusion_EPE

import time
import os
from torch.autograd import Variable

from losses.unsupervised_loss import loss_disp_unsupervised,MultiScaleLoss,Eki_Loss
from losses.pam_loss import PAMStereoLoss, loss_ssim_l1

import os
import argparse
import datetime
import random
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.common import *
from trainfiles.trainer_stereo_iter import DisparityTrainer
from utils import renderer
import wandb

from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(opt.outf,filename))
    if is_best:
        torch.save(state, os.path.join(opt.outf,'model_best.pth'))

'''  Main Function to train the model'''
def main(opt):

    # load the training loss scheme
    loss_json = load_loss_scheme(opt.loss)
    train_round = loss_json["round"]
    loss_scale = loss_json["loss_scale"]
    loss_weights = loss_json["loss_weights"]
    epoches = loss_json["epoches"]
    logger.info(loss_weights)
    
    
    # initialize a trainer
    tester1 = DisparityTrainer(opt.lr, opt.devices, 
                               opt.dataset, opt.trainlist, opt.vallist, 
                               opt.datapath, opt.batch_size, opt.maxdisp,opt.use_deform,'models_saved/PAMSDF/pamsdf_1.590.pth','PAMSDF', 
                               test_batch=opt.test_batch,initial_pretrain=opt.initial_pretrain, wandb=wandb,opt=opt)
    tester2 = DisparityTrainer(opt.lr, opt.devices, 
                               opt.dataset, opt.trainlist, opt.vallist, 
                               opt.datapath, opt.batch_size, opt.maxdisp,opt.use_deform,'models_saved/PAMSDF/pamsdfrender_1.641.pth','PAMSDFRender', 
                               test_batch=opt.test_batch,initial_pretrain=opt.initial_pretrain, wandb=wandb,opt=opt)
    # checkpoint1 = torch.load('checkpoints/ckpt_100.pth') # 加载断点
    # tester1.net.load_state_dict(checkpoint1['state_dict']) # 加载模型可学习参数

    # checkpoint2 = torch.load('checkpoints/ckpt_100.pth') # 加载断点
    # tester2.net.load_state_dict(checkpoint2['state_dict']) # 加载模型可学习参数
    
    sample_batched = next(iter(tester1.train_loader))
    left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
    right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
    
    img = left_input[0]
    to_image = transforms.ToPILImage()
    img = to_image(img)
    img.save('test_ori.jpg')
    if tester1.model =="PAMSDF":
        output,attn_list,att_cycle,valid_mask,est_sdf= tester1.net(left_input,right_input,192)
        renderer.visualize_mesh(output, est_sdf, 'test_PAMSDF')
    if tester2.model == 'PAMSDFRender':
        output,attn_list,att_cycle,valid_mask,est_sdf,rendered_left,weights_sum= tester2.net(left_input,right_input,192)
        renderer.visualize_mesh(output, est_sdf, 'test_PAMSDFRender')

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicates the name of net', default='simplenet')
    parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='config/loss_config_disp.json')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--test_batch', type=int, default=4, help='test batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd, alpha parameter for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=0.999, help='beta parameter for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables, cuda')
    parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--model', type=str, help='model for finetuning', default='PAMSDF')
    parser.add_argument('--startRound', type=int, help='the round number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--startEpoch', type=int, help='the epoch number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--logFile', type=str, help='logging file', default='./train.log')
    parser.add_argument('--showFreq', type=int, help='display frequency', default='100')
    parser.add_argument('--flowDiv', type=float, help='the number by which the flow is divided.', default='1.0')
    parser.add_argument('--maxdisp', type=int, help='disparity search range.', default='-1')
    parser.add_argument('--dataset', type=str, help='provide the dataset name', default='KITTI')
    parser.add_argument('--datapath', type=str, help='provide the root path of the data', default='/data/public_dataset_111/KITTI_Raw_Data')
    parser.add_argument('--trainlist', type=str, help='provide the train file (with file list)', default='filenames/kitti_raw_complete.txt')
    parser.add_argument('--vallist', type=str, help='provide the val file (with file list)', default='filenames/KITTI_2015_train.txt')
    parser.add_argument('--augment', type=int, help='if augment data in training', default=0)
    parser.add_argument('--trainmodel', type=str, help='denote the specific model to train')
    parser.add_argument('--use_deform',action='store_true',help='use deform conv or not in features extraction period')
    parser.add_argument('--save_logdir',type=str,help='tensorboard log files saved path',default='experiments_logdirs')
    parser.add_argument('--pretrain',type=str,help='Load pretrain model for fine-tuning',default='models_saved/PAMSDF/model_best.pth')
    parser.add_argument('--initial_pretrain',type=str,help='Load Part of the weight',default='None')

    # additional parameters
    parser.add_argument('--summary_freq', type=int, default=100, help='summary frequency')
    parser.add_argument('--sdf_weight', type=float, default=0.01, help='loss weight for sdf eikonal regularization')
    parser.add_argument('--sdf_type', type=str, default='MLP', help='cost volume process method for sdf regression')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='indicates the optimizer for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dynamic_lr', action='store_true', help='launch dynamic lr strategy')
    parser.add_argument('--load_wandb', action='store_true', help='apply wandb for summary')
    parser.add_argument('--num_steps', type=int, default=100000, help='total steps for training')
    parser.add_argument('--val_freq', type=int, default=10000, help='validation frequency')
    parser.add_argument('--datathread', type=int, default=4, help='numworkers for dataloader')
    opt = parser.parse_args()

    print("Use Deformable Conv ? :",opt.use_deform)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    hdlr = logging.FileHandler(opt.logFile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', opt)
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logger.info("Random Seed: %s", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if torch.cuda.is_available() and not opt.cuda:
        logger.warning("WARNING: You should run with --cuda since you have a CUDA device.")
    main(opt)
