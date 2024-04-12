import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
from utils.utils import read_text_lines
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import shutil

if __name__=="__main__":
    
    source_datapath="/data1/KITTI/KITTI_Raw/"
    outside_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"
    center_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train_SmallSet/"
    confidence_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/Reference_PSNR/"
    
    
    example_path = "example_path.txt"
    
    lines = read_text_lines(example_path)
    
    idx = 0
    
    saved_folder_initial = "visualizations_initial/example"
    for line in tqdm(lines):
        
        idx = idx + 1        
        saved_folder = saved_folder_initial+ "{}".format(idx)
        os.makedirs(saved_folder,exist_ok=True)
        
        
        left_image_path = os.path.join(source_datapath,line)
        right_image_path = left_image_path.replace("image_02","image_03")
        
        left_left_image = os.path.join(outside_view_datapath,line)
        left_left_image = left_left_image.replace("image_02","image_01")
        right_right_image = left_left_image.replace("image_01","image_04")
        
        center_image = os.path.join(center_view_datapath,line)
        center_image = center_image.replace("image_02","image_02_center")
        
        assert os.path.exists(left_image_path)
        assert os.path.exists(right_image_path)
        assert os.path.exists(left_left_image)
        assert os.path.exists(right_right_image)
        assert os.path.exists(center_image)
        
        
        saved_left_name = os.path.join(saved_folder,"1.png")
        saved_right_name = os.path.join(saved_folder,"2.png")
        
        saved_left_left = os.path.join(saved_folder,"0.png")
        saved_right_right = os.path.join(saved_folder,"3.png")
        
        # saved_center = os.path.join(saved_folder,'2.png')
        
        try:
            shutil.copyfile(left_image_path,saved_left_name)
            shutil.copyfile(right_image_path,saved_right_name)
            shutil.copyfile(left_left_image,saved_left_left)
            shutil.copyfile(right_right_image,saved_right_right)
            # shutil.copyfile(center_image,saved_center)        
        except:
            print("sss")
        

        
        
    print("Done")


    