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
import cv2
from skimage.metrics import structural_similarity as compare_ssim

from PIL import Image

def get_ssim(image1, image2):
    if image1.shape[-1]==3:
        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
        image2 = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)    
    ssim = compare_ssim(image1,image2)
    return ssim


# Read Image
def read_img(filename):
    # Convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img



if __name__=="__main__":
    
    source_datapath="/data1/KITTI/KITTI_Raw"
    target_datapath="/data1/KITTI/KITTI_Train"
    
    example_path = "/home/zliu/Desktop/UnsupervisedStereo/StereoSDF/filenames/kitti_raw_all.txt"
    lines = read_text_lines(example_path)
    
    confidence_list = []
    
    for fname in tqdm(lines):
        left_image_path = os.path.join(source_datapath,fname)
        right_image_path = left_image_path.replace("image_02","image_03")
        
        left_left_image_path = left_image_path.replace("image_02","image_01")
        left_left_image_path = left_left_image_path.replace("KITTI_Raw","KITTI_Train")
        
        right_right_image_path = left_image_path.replace("image_02","image_04")
        right_right_image_path = right_right_image_path.replace("KITTI_Raw","KITTI_Train")
        
        center_image_path =left_image_path.replace("image_02","image_025")
        center_image_path = center_image_path.replace("KITTI_Raw","KITTI_Train")
        
        
        

        
        
    print("Done!, All Files is Exist!")
    



    