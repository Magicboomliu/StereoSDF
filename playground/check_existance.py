import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
from utils.utils import read_text_lines



if __name__=="__main__":
    
    data_path = "/media/zliu/data12/dataset/KITTI/KITTI_Raw/"
    rendered_path = "/media/zliu/data12/dataset/KITTI/NewViews/"
    
    kitti_all = "/home/zliu/ACMMM2024/UnsupervisedStereo/StereoSDF/filenames/kitti_raw_complete.txt"
    
    
    kitti_contents = read_text_lines(kitti_all)
    for idx, fname in enumerate(kitti_contents):
        splits = fname.split()
        left_image_name = splits[0]
        right_image_name = splits[1]
        basename = os.path.basename(left_image_name)
        
        left_image_name = os.path.join(data_path,left_image_name)
        right_image_name = os.path.join(data_path,right_image_name)
        rendered_left_left = left_image_name.replace(data_path,rendered_path)
        rendered_right_right = right_image_name.replace(data_path,rendered_path)
        
        rendered_left_left = rendered_left_left.replace(basename,"left_left_from_left_"+basename)
        rendered_right_right = rendered_right_right.replace(basename,"right_right_from_right_"+basename)
        
        
        assert os.path.exists(rendered_left_left)
        assert os.path.exists(rendered_right_right)

    print("OK")