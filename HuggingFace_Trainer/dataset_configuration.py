import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from dataloader.kitti_loader_bmvc import KITTI_Multi_View_Dataset
from dataloader import kitti_transforms_bmvc
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
def prepare_dataset(source_datapath,
                    outside_view_datapath,
                    center_view_datapath,
                    confidence_datapath,
                    trainlist,
                    vallist,
                    batch_size,
                    datathread):
    
    train_transform_list = [kitti_transforms_bmvc.RandomCrop(320, 960),
                            kitti_transforms_bmvc.ToTensor(),
                            kitti_transforms_bmvc.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    train_transform = kitti_transforms_bmvc.Compose(train_transform_list)
    
    
    val_transform_list = [kitti_transforms_bmvc.ToTensor(),
                        kitti_transforms_bmvc.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    val_transform = kitti_transforms_bmvc.Compose(val_transform_list)
    
    


    train_dataset = KITTI_Multi_View_Dataset(source_datapath=source_datapath,
                                             outside_view_datapath=outside_view_datapath,
                                             center_view_datapath=center_view_datapath,
                                             confidence_datapath=confidence_datapath,
                                             train_datalist=trainlist,
                                             test_datalist=vallist,
                                             mode='train',
                                             transform=train_transform,
                                             visible_list=['left','right'])
    
    test_dataset = KITTI_Multi_View_Dataset(source_datapath=source_datapath,
                                             outside_view_datapath=outside_view_datapath,
                                             center_view_datapath=center_view_datapath,
                                             confidence_datapath=confidence_datapath,
                                             train_datalist=trainlist,
                                             test_datalist=vallist,
                                             mode='test',
                                             transform=val_transform)
    
    

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = 1, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)
    
    
    return train_loader,test_loader,num_batches_per_epoch





if __name__=="__main__":
    
    source_datapath = "/data1/KITTI/KITTI_Raw/"
    rendered_datapath_outside = "/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"
    rendered_datapath_center = "/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train_SmallSet/"
    rendered_dataset_confidence = "/data1/KITTI/Rendered_Results/Simple_UNet/Reference_PSNR/"
    
    
    trainlist = "/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTISmallSet/kitti_raw_train.txt"
    vallist = "/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTI_2015_train.txt"
    
    
    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(source_datapath=source_datapath,
                                                                    outside_view_datapath=rendered_datapath_outside,
                                                                    center_view_datapath=rendered_datapath_center,
                                                                    confidence_datapath=rendered_dataset_confidence,
                                                                    trainlist=trainlist,
                                                                    vallist=vallist,
                                                                    batch_size=1,
                                                                    datathread=1,
                                                                     )
    
    # test the trainloader
    for sample in tqdm(train_loader):
        
        left_image = sample['img_left']
        right_image = sample['img_right']
        # left_left_image = sample['img_left_left']
        # right_right_image = sample['img_right_right']
        # center_image = sample['img_center']
        # quality = sample['img_quality']
        
        
        
        # left_image = Image_DeNormalization(left_image)
        # right_image = Image_DeNormalization(right_image)
        # left_left_image = Image_DeNormalization(left_left_image)
        # right_right_image = Image_DeNormalization(right_right_image)
        # center_image = Image_DeNormalization(center_image)
        
        
        # left_image_vis =  Convert_IMGTensor_To_Numpy(left_image)
        # right_image_vis = Convert_IMGTensor_To_Numpy(right_image)
        # left_left_image_vis = Convert_IMGTensor_To_Numpy(left_left_image)
        # right_right_image_vis = Convert_IMGTensor_To_Numpy(right_right_image)
        # center_image_vis = Convert_IMGTensor_To_Numpy(center_image)
        
        
        # plt.figure(figsize=(15,1))
        # plt.subplot(1,5,1)
        # plt.axis('off')
        # plt.title("left-left-image")
        # plt.imshow(left_left_image_vis)
        # plt.subplot(1,5,2)
        # plt.axis("off")
        # plt.title("left-image")
        # plt.imshow(left_image_vis)
        # plt.subplot(1,5,3)
        # plt.axis("off")
        # plt.title("center-image")
        # plt.imshow(center_image_vis)
        # plt.subplot(1,5,4)
        # plt.axis("off")
        # plt.title("right-image")
        # plt.imshow(right_image_vis)
        # plt.subplot(1,5,5)
        # plt.axis("off")
        # plt.title("right-right-image")
        # plt.imshow(right_right_image_vis)
        # plt.show()
        # plt.savefig("dataset.png")
        
        # quit()
   
        
        

    # test the valloader
    # for sample in tqdm(test_loader):
    #     left_image = sample['img_left']
    #     right_image = sample['img_right']
    #     gt_disp = sample['gt_disp']

        
    
        

    print("all files is ok")
    
    