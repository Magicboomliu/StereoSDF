import os
import numpy as np
import sys
sys.path.append("..")
from utils.utils import read_text_lines
from  tqdm import tqdm
if __name__=="__main__":

    source_datapath="/data1/KITTI/KITTI_Raw/"
    outside_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train"
    center_view_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/KITTI_Train_SmallSet/"
    confidence_datapath="/data1/KITTI/Rendered_Results/Simple_UNet/Reference_PSNR/"
    trainlist="/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/KITTISmallSet/new_kitti_raw_subset2.txt"
    
    lines = read_text_lines(trainlist)
    
    
    valid_id = []
    valid_id2 = []
    for line in tqdm(lines):
        left_image = os.path.join(source_datapath,line)
        right_image = left_image.replace("image_02","image_03")
        center_image = os.path.join(center_view_datapath,line)
        center_image = center_image.replace("image_02","image_02_center")
        
        rendered_img_confidence = os.path.join(confidence_datapath,line)
        rendered_img_confidence = rendered_img_confidence.replace(".png",".txt")
        
        
        left_left_image = os.path.join(outside_view_datapath,line)
        left_left_image  = left_left_image.replace("image_02","image_01")
        right_right_image = left_left_image.replace("image_01","image_04")
        

        assert os.path.exists(rendered_img_confidence)
        assert os.path.exists(right_image)
        assert os.path.exists(left_image)
        assert os.path.exists(left_left_image)
        assert os.path.exists(right_right_image)
        assert os.path.exists(center_image)
    

    
    print("OK")
    pass
