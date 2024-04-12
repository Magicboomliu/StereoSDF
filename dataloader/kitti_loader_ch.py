from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from utils import utils
from utils.kitti_io import read_img, read_disp,read_kitti_step1,read_kitti_step2,read_kitti_image_step1,read_kitti_image_step2
from skimage import io, transform
import numpy as np
from PIL import Image

def compute_left_occ_region(w, disp):
    """
    Compute occluded region on the left image border
    :param w: image width
    :param disp: left disparity
    :return: occ mask
    """
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    shifted_coord = coord - disp
    occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ
    return occ_mask




class KITTI_Multi_View_Dataset(Dataset):
    def __init__(self, 
                 source_datapath,
                 outside_view_datapath,
                 center_view_datapath,
                 confidence_datapath,
                 train_datalist,
                 test_datalist,
                 mode='train',
                 transform=None,
                 visible_list = ["left","right","left_left","right_right","center","confidence"]):
        super(KITTI_Multi_View_Dataset, self).__init__()


        self.mode = mode
        self.source_datapath = source_datapath
        self.outside_view_datapath = outside_view_datapath
        self.center_view_datapath = center_view_datapath
        self.confidence_datapath = confidence_datapath
        
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(384, 1280)
        self.scale_size =(384,1280)
        
        self.visible_list = visible_list
        

        dataset_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        self.samples = []
        data_filenames = dataset_dict[mode]
        lines = utils.read_text_lines(data_filenames)

        
        for line in lines:
            sample = dict()
            if self.mode=='train':
                left_image = line
                if 'left' in self.visible_list:
                    sample['left'] = os.path.join(self.source_datapath,left_image)
                if 'right' in self.visible_list:
                    sample['right'] = sample['left'].replace("image_02","image_03")
                
                if 'left_left' in self.visible_list:
                    rendered_left_left = os.path.join(self.outside_view_datapath,left_image)
                    rendered_left_left = rendered_left_left.replace("image_02","image_01")
                    sample['left_left'] = rendered_left_left
                    
                if "right_right" in self.visible_list:
                    rendered_right_right = rendered_left_left.replace("image_01","image_04")
                    sample['right_right'] = rendered_right_right
                
                if 'center' in self.visible_list:
                    rendered_center = os.path.join(self.center_view_datapath,left_image)
                    rendered_center = rendered_center.replace("image_02","image_02_center")
                    sample['center'] = rendered_center
                    
                if "confidence" in self.visible_list:
                    # loaded the confidnce
                    rendered_img_confidence = os.path.join(self.confidence_datapath,left_image)
                    rendered_img_confidence = rendered_img_confidence.replace(".png",".txt")
                    sample['quality'] = rendered_img_confidence
                
                # gt disparity
                
                
                self.samples.append(sample)
                 
            
            elif self.mode=='val' or self.mode=='test':
                splits = line.split()
                left_image = splits[0]
                right_image = splits[1]
                disp_path = splits[2]
            
                sample['left'] = os.path.join(self.source_datapath,left_image)
                sample['right'] = os.path.join(self.source_datapath,right_image)
                sample['disp'] = os.path.join(self.source_datapath,disp_path)
                
                self.samples.append(sample)
            else:
                raise NotImplementedError
        
    
        
                

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        


        # read the kitti raw dataset
        if self.mode=='train':
            if 'left' in self.visible_list: 
                sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
            if 'right' in self.visible_list:
               
                sample['img_right'] = read_img(sample_path['right']) # [H,W,3]
            if 'left_left' in self.visible_list:
               
                sample['img_left_left'] = read_img(sample_path['left_left']) # [H,W,3]
            if 'right_right' in self.visible_list:
                
                sample['img_right_right'] = read_img(sample_path['right_right']) # [H,W,3]
            if 'center' in self.visible_list:
                
                sample['img_center'] = read_img(sample_path['center']) # [H,W,3]
            if 'confidence' in self.visible_list:

                sample['img_quality'] = np.loadtxt(sample_path['quality'],dtype=float)[0:1]
            
            
            
            
        
        elif self.mode=='val' or self.mode=='test':
            # print(sample_path['left'])
            if sample_path['disp'] is not None:
                # Image Crop Operation
                left_im = read_kitti_image_step1(sample_path['left']) #[H,W,3]
                
                right_im = read_kitti_image_step1(sample_path['right'])        
                w, h = left_im.size
                left_image = left_im
                right_image = right_im
                sample['img_left'] = read_kitti_image_step2(left_image)
                sample['img_right'] = read_kitti_image_step2(right_image)
                w1,h1 = left_image.size
                
                
                # Disparity Crop Operation
                if sample_path['disp'] is not None:
                    gt_disp = read_kitti_step1(sample_path['disp'])
                    w, h = gt_disp.size
                    dataL = gt_disp
                    dataL = read_kitti_step2(dataL)
                    sample['gt_disp']= dataL
                
        else:
            raise NotImplementedError
        

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size
