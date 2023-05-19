import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from dataloader.kitti_loader import StereoDataset
from dataloader import kitti_transform 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from utils.common import logger, check_path, write_pfm,count_parameters
import matplotlib.pyplot as plt



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def prepare_dataset(dataset,datapath,trainlist,vallist):
    if dataset == 'kitti2015':
        
        train_transform_list = [kitti_transform.RandomCrop(320, 640),
                            kitti_transform.ToTensor(),
                            kitti_transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
        train_transform = kitti_transform.Compose(train_transform_list)

        val_transform_list = [kitti_transform.ToTensor(),
                                kitti_transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        val_transform = kitti_transform.Compose(val_transform_list)
        test_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,
                                        test_datalist=vallist,
                                dataset_name='KITTI_mix',mode='test',transform=val_transform)
        
        train_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                    dataset_name='KITTI_mix',mode='train',transform=train_transform)

    datathread=4
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    logger.info("Use %d processes to load data..." % datathread)

    test_loader = DataLoader(test_dataset, batch_size = 1, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    
    train_loader = DataLoader(train_dataset, batch_size =1, \
                                shuffle = True, num_workers = datathread, \
                                pin_memory = True)
    return test_loader,train_loader



if __name__=="__main__":
    
    datapath = "/media/zliu/datagrid1/liu/kitti_stereo/kitti_2015"
    trainfile  = "../filenames/KITTI_2015_train.txt"
    testfile ="../filenames/KITTI_2015_val.txt"
    
    
    test_loader,train_loader = prepare_dataset(datapath=datapath,dataset='kitti2015',trainlist=trainfile,vallist=testfile)
    
    

    for i, sample_batched in enumerate(train_loader):
        left_input = Variable(sample_batched['img_left'].cuda(), requires_grad=False)
        right_input = Variable(sample_batched['img_right'].cuda(), requires_grad=False)

        target_disp = sample_batched['gt_disp'].unsqueeze(1)
        target_disp = target_disp.cuda()
        target_disp =Variable(target_disp, requires_grad=False) #[B,C,H,W]
        
        
        print(left_input.shape)
        print(target_disp.shape)