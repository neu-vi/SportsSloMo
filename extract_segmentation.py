import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
import time
import os
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from Mask2Former.mask2former import add_maskformer2_config
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'Mask2Former/checkpoints/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)
img_root = 'YOUR_PATH_TO_SportsSloMo_frames'
dst_seg_root = 'YOUR_PATH_TO_SportsSloMo_segmentation'

if not os.path.exists(dst_seg_root):
    os.mkdir(dst_seg_root)
for idx, name in enumerate(sorted(os.listdir(img_root))):
    img_dir = os.path.join(img_root, name)
    seg_dir = os.path.join(dst_seg_root,name)
    if not os.path.exists(seg_dir):
        os.mkdir(seg_dir)
    
    all_img_paths = sorted([os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith('.png')])
    for img_path in all_img_paths:
        img = read_image(img_path, format="BGR")

        outputs = predictor(img)

        seg = outputs["panoptic_seg"][0].to("cpu").numpy()
        info = np.array(outputs["panoptic_seg"][1])

        base_name = os.path.basename(img_path)
        name_without_extension = os.path.splitext(base_name)[0]
        seg_dst_path = os.path.join(seg_dir, name_without_extension + '_seg.npy')
        info_dst_path = os.path.join(seg_dir, name_without_extension + '_info.npy')

        np.save(seg_dst_path, seg)
        np.save(info_dst_path, info)