import os
import numpy as np
import torch
import sys
sys.path.append("..")

from models.PAMStereo.PASMnet import PASMnet


if __name__=="__main__":
    
    
    stereonet = PASMnet()
    saved_path = "../outputs/simple_pamstereo/ckpt_step_1.pt"

    
    ckpts = torch.load(saved_path)
    
    model_ckpt = ckpts['model_state']
    
    stereonet.load_state_dict(model_ckpt)
    
    stereonet.cuda()
