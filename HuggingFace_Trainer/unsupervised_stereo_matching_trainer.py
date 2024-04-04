import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from models.PAMStereo.PASMnet import PASMnet
from dataloader.kitti_loader import StereoDataset
import json
from dataloader import kitti_transform

from losses.pam_loss import PAMStereoLoss
from tqdm import tqdm
import argparse

# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Stereo Matching Training")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data1/KITTI/KITTI_Raw",
        required=True,
        help="Specify the dataset name used for training/validation.",
    )



if __name__=="__main__":
    pass


