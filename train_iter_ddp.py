from __future__ import print_function
import os
import argparse
import datetime
import random
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.common import *
from trainfiles.trainer_stereo_iter_ddp import DisparityTrainer

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
    
    use_wandb = opt.load_wandb
    
    if use_wandb==True:
        import wandb
        # init wandb
        group_name = 'model_%s_sdf_%s_cos_lr' % (opt.model, opt.sdf_type)
        name = 'sdf_weight_%.2f_lr_%f_optim_%s' % (opt.sdf_weight, opt.lr, opt.optimizer)
        wandb.init(
            project = 'stereo_sdf',
            group = group_name,
            name = name,
            config = {
                'learning_rate' : opt.lr,
                'training_epochs' : train_round * epoches,
                'sdf_weight' : opt.sdf_weight,
                'model' : opt.model,
                'entity': 'sterer-sdf'
            }
        )
    else:
        wandb = None
    
    # initialize a trainer
    trainer = DisparityTrainer(opt.lr, opt.devices, 
                               opt.dataset, opt.trainlist, opt.vallist, 
                               opt.datapath, opt.batch_size, opt.maxdisp,opt.use_deform,opt.pretrain,opt.model, 
                               test_batch=opt.test_batch,initial_pretrain=opt.initial_pretrain, wandb=wandb,
                               local_rank=opt.local_rank,
                               opt=opt)
    
    # validate the pretrained model on test data
    best_EPE = -1
    best_index = 0
    summary_writer = SummaryWriter(opt.save_logdir)
    start_epoch = opt.startEpoch
    
    # load training process
    trainer.launch_training(opt.local_rank)
    wandb.finish()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicates the name of net', default='simplenet')
    parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='simplenet_flying')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--test_batch', type=int, default=4, help='test batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd, alpha parameter for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=0.999, help='beta parameter for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables, cuda')
    parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--model', type=str, help='model for finetuning', default='')
    parser.add_argument('--startRound', type=int, help='the round number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--startEpoch', type=int, help='the epoch number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--logFile', type=str, help='logging file', default='./train.log')
    parser.add_argument('--showFreq', type=int, help='display frequency', default='100')
    parser.add_argument('--flowDiv', type=float, help='the number by which the flow is divided.', default='1.0')
    parser.add_argument('--maxdisp', type=int, help='disparity search range.', default='-1')
    parser.add_argument('--dataset', type=str, help='provide the dataset name', default='sceneflow')
    parser.add_argument('--datapath', type=str, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--trainlist', type=str, help='provide the train file (with file list)', default='FlyingThings3D_release_TRAIN.list')
    parser.add_argument('--vallist', type=str, help='provide the val file (with file list)', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--augment', type=int, help='if augment data in training', default=0)
    parser.add_argument('--trainmodel', type=str, help='denote the specific model to train')
    parser.add_argument('--use_deform',action='store_true',help='use deform conv or not in features extraction period')
    parser.add_argument('--save_logdir',type=str,help='tensorboard log files saved path',default='experiments_logdirs')
    parser.add_argument('--pretrain',type=str,help='Load pretrain model for fine-tuning',default='None')
    parser.add_argument('--initial_pretrain',type=str,help='Load Part of the weight',default='None')


    # local rank
    parser.add_argument("--local_rank",help='local device id on current node',type=int)


    # additional parameters
    parser.add_argument('--summary_freq', type=int, default=10, help='summary frequency')
    parser.add_argument('--sdf_weight', type=float, default=0.01, help='loss weight for sdf eikonal regularization')
    parser.add_argument('--sdf_type', type=str, default='2D_conv', help='cost volume process method for sdf regression')
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
