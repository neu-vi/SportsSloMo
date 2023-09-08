import argparse
import os
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np


import time
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

from ViTPose_pytorch.models.model import ViTPose
from ViTPose_pytorch.utils.visualization import draw_points_and_skeleton, joints_dict
from ViTPose_pytorch.utils.dist_util import get_dist_info, init_dist
from ViTPose_pytorch.utils.top_down_eval import keypoints_from_heatmaps
from typing import Dict, Tuple

from ultralytics import YOLO
yolo_model = YOLO("yolov8n.yaml")  # build a new model from scratch
yolo_model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

__all__ = ['inference']

@torch.no_grad()
def inference(img_path, img_size, pose_model, device):
    yolo_results = yolo_model(img_path)
    yolo_result = yolo_results[0]
    boxes = yolo_result.boxes
    cls = boxes.cls
    xyxy = boxes.xyxy
    xyxys = xyxy[cls==0]
    img = Image.open(img_path)
    num_person = xyxys.shape[0]
    points_all = []
    for i in range(num_person):
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        xyxy = xyxys[i].int().tolist()
        img_tensor = img_tensor[:,:,xyxy[1]:xyxy[3] ,xyxy[0]:xyxy[2]]
        _, _, org_h, org_w = img_tensor.shape
        img_tensor = transforms.Resize((img_size[1], img_size[0]))(img_tensor)
        heatmaps = pose_model(img_tensor).detach().cpu().numpy()
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                            unbiased=True, use_udp=True)
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)
        points[:,:,0] += xyxy[1]
        points[:,:,1] += xyxy[0]
        points_all.append(points)
    if len(points_all) > 0:
        points_all = np.concatenate(points_all, 0)
    else:
        points_all = None

    return points_all

if __name__ == "__main__":
    from configs.ViTPose_large_coco_256x192 import model as model_cfg
    from configs.ViTPose_large_coco_256x192 import data_cfg
    
    # Prepare model
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    CKPT_PATH = "ViTPose_pytorch/checkpoints/vitpose-l-multi-coco.pth"
    vit_pose = ViTPose(model_cfg)
    vit_pose.load_state_dict(torch.load(CKPT_PATH)['state_dict'])
    vit_pose.to(device)

    img_size = data_cfg['image_size']

    img_root = 'YOUR_PATH_TO_SportsSloMo_frames'
    dst_pose_root = 'YOUR_PATH_TO_SportsSloMo_kpts'
    if not os.path.exists(dst_pose_root):
        os.mkdir(dst_pose_root)
    for idx, name in enumerate(sorted(os.listdir(img_root))):
        img_dir = os.path.join(img_root, name)
        kpt_dir = os.path.join(dst_pose_root,name)
        if not os.path.exists(kpt_dir):
            os.mkdir(kpt_dir)
        
        all_img_paths = sorted([os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith('.png')])

        for img_path in all_img_paths:
            kpt = inference(img_path=img_path, img_size=img_size, pose_model=vit_pose, device=device)
            
            base_name = os.path.basename(img_path)
            name_without_extension = os.path.splitext(base_name)[0]
            kpt_dst_path = os.path.join(kpt_dir, name_without_extension + '.npy')
            np.save(kpt_dst_path, kpt)