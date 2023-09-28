import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np


import time
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torchvision.ops as ops
import torch.nn.functional as F

from core.modules.ViTPose_pytorch.models.model import ViTPose
from core.modules.ViTPose_pytorch.utils.dist_util import get_dist_info, init_dist
from core.modules.ViTPose_pytorch.utils.top_down_eval import keypoints_from_heatmaps
from typing import Dict, Tuple

# from ultralytics import YOLO

from core.modules.ViTPose_pytorch.configs.ViTPose_large_coco_256x192 import model as model_cfg
from core.modules.ViTPose_pytorch.configs.ViTPose_large_coco_256x192 import data_cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HeatmapInfer(nn.Module):
    def __init__(self):
        super(HeatmapInfer, self).__init__()
        self.CKPT_PATH = "ViTPose_pytorch/checkpoints/vitpose-l-multi-coco.pth" #Might need adjustment to your global path
        self.vit_pose = ViTPose(model_cfg)
        self.vit_pose.load_state_dict(torch.load(self.CKPT_PATH)['state_dict'])
        self.vit_pose.to(device)
        for name, module in self.vit_pose.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                frozen_module = ops.FrozenBatchNorm2d(module.num_features)
                frozen_module.running_mean = module.running_mean
                frozen_module.running_var = module.running_var
                frozen_module.weight = nn.Parameter(module.weight.data.clone().detach())
                frozen_module.bias = nn.Parameter(module.bias.data.clone().detach())
                parent_name, sub_name = name.rsplit(".", 1)
                parent_module = self.vit_pose
                for n in parent_name.split("."):
                    parent_module = parent_module._modules[n]
                setattr(parent_module, sub_name, frozen_module)
        for p_ in self.vit_pose.parameters():
            p_.requires_grad = False

        self.img_size = data_cfg['image_size']
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, img, gt):
        predict_imgs = F.interpolate(img, size=(self.img_size[1], self.img_size[0]), mode='bilinear', align_corners=False)
        assert(predict_imgs.shape[-3] == 3)
        predict_imgs = predict_imgs.flip(-3)
        gt_imgs = F.interpolate(gt, size=(self.img_size[1], self.img_size[0]), mode='bilinear', align_corners=False)
        assert(gt_imgs.shape[-3] == 3)
        gt_imgs = gt_imgs.flip(-3)
        cur_heatmaps = self.vit_pose(predict_imgs)
        gt_heatmaps = self.vit_pose(gt_imgs)
        return cur_heatmaps, gt_heatmaps
            


                


