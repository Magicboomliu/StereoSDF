import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from torch.utils.data import DataLoader 
from dataloader_kitti.KITTI_Loader import KITTI_Multi_Baseline_Dataset
from dataloader_kitti import kitti_transforms
import numpy as np
from tqdm import tqdm
from utils.devtools import Convert_IMGTensor_To_Numpy
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm



# Get Dataset Here
def prepare_dataset(datapath,
                    trainlist,
                    vallist,
                    batch_size,
                    datathread,
                    visible_list=None):
    
    train_transform_list = [kitti_transforms.ToTensor(),
                            kitti_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    train_transform = kitti_transforms.Compose(train_transform_list)
    
    
    val_transform_list = [kitti_transforms.ToTensor(),
                        kitti_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    val_transform = kitti_transforms.Compose(val_transform_list)
    
    
    train_dataset = KITTI_Multi_Baseline_Dataset(datapath=datapath,
                                                 train_datalist=trainlist,
                                                 test_datalist=vallist,
                                                 mode='train',
                                                 transform=train_transform,
                                                 visible_list=visible_list
                                                 )
    
    test_dataset = KITTI_Multi_Baseline_Dataset(datapath=datapath,
                                                 train_datalist=trainlist,
                                                 test_datalist=vallist,
                                                 mode='test',
                                                 transform=val_transform,
                                                 visible_list=visible_list
                                                 )
    

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = 1, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)

    return train_loader,test_loader,num_batches_per_epoch





if __name__=="__main__":
    
    datapath = "/data1/KITTI/KITTI_Raw/"
    trainlist = "/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTISmallSet/kitti_raw_train.txt"
    vallist = "/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTI_2015_train.txt"
    
    
    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(datapath=datapath,
                                                                     trainlist=trainlist,
                                                                     vallist=vallist,
                                                                     batch_size=1,
                                                                     datathread=1,
                                                                     visible_list=['left','right','left_left','right_right',
                                                                                  'center'])
    # # The training dataset
    # for idx, sample in enumerate(train_loader):
        
    #     left_image = sample['img_left']  # [B,3,H,W]
    #     right_image = sample['img_right'] # [B,3,H,W]
    #     left_left_image = sample['img_left_left'] # [B,3,H,W]
    #     right_right_image = sample['img_right_right'] # [B,3,H,W] 
    #     center_image = sample['img_center'] # [B,3,H,W]
    #     resolution = sample['resolution'] # [B,2]
        
        
    #     print(left_image.shape)
    #     print(right_image.shape)
    #     print(left_left_image.shape)
    #     print(right_right_image.shape)
    #     print(center_image.shape)
    #     print(resolution)
    #     break


    # for idx, sample in enumerate(test_loader):
        
    #     left_image = sample['img_left']
    #     right_image = sample['img_right']
    #     disp = sample['gt_disp']
        
        
    #     print(left_image.shape)
    #     print(right_image.shape)
    #     print(disp.shape)
        
    #     break
        
        
        
        
        
    
