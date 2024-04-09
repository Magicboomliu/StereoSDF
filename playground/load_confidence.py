import os
import numpy as np

if __name__=="__main__":
    example_txt = "/data1/KITTI/Rendered_Results/Simple_UNet/Reference_PSNR/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.txt"
    
    context = np.loadtxt(example_txt,dtype=float)
    psnr = context[0:1]
    ssim = context[1:]
    
    
    print(psnr)
    print(ssim)
    
    
    pass