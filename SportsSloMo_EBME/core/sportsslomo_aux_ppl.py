import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from importlib import import_module
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from .loss import EPE, Ternary, LapLoss
from core.modules.bi_flownet import BiFlowNet
from core.modules.fusionnet import FusionNet

from core.heatmap_loss import HeatmapInfer

#####MASK2FORMER
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")
import cv2
from detectron2 import model_zoo
from core.modules.detectron2_modify.detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
from core.modules.detectron2_modify.detectron2.utils.visualizer import _PanopticPrediction
from core.modules.mask2former import add_maskformer2_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class AuxPipeline:
    def __init__(self,
            module_cfg_dict, # To check the data format, refer to the config files in `./configs`.
            optimizer_cfg_dict=None, # To check the data formart, refer to the config files in `./configs`.
            local_rank=-1, # `local_rank=-1` means test mode. We init the pipeline with test mode by default.
            training=False, # Init the pipeline for testing by default.
            resume=False # If set `True`, we will restart the experiment from previous checkpoint.
            ):
        self.module_cfg_dict = module_cfg_dict
        self.optimizer_cfg_dict = optimizer_cfg_dict
        self.epe = EPE()
        self.ter = Ternary()
        self.laploss = LapLoss()
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_maskformer2_config(self.cfg)
        self.cfg.merge_from_file("Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
        self.cfg.MODEL.WEIGHTS = 'Mask2Former/checkpoints/model_final_f07440.pkl'
        self.cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        self.predictor = DefaultPredictor(self.cfg)

        self.HeatmapInfer = HeatmapInfer()
        self.heatmaploss = JointsMSELoss()

        self.init_ppl_modules()
        self.device()
        self.training = training

       

        # We note that in practical, the `lr` of AdamW is reset from the outside,
        # using cosine annealing during the while training process.
        if training:
            self.optimG = AdamW(itertools.chain(
                filter(lambda p: p.requires_grad, self.bi_flownet.parameters()),
                filter(lambda p: p.requires_grad, self.fusionnet.parameters())),
                lr=optimizer_cfg_dict["init_lr"],
                weight_decay=optimizer_cfg_dict["weight_decay"])

        # `local_rank == -1` is used for testing, which does not need DDP
        if local_rank != -1:
            if not vars(module_cfg_dict["bi_flownet"]).get("fix_pretrain", False):
                self.bi_flownet = DDP(self.bi_flownet, device_ids=[local_rank],
                        output_device=local_rank, find_unused_parameters=False)
            if not vars(module_cfg_dict["fusionnet"]).get("fix_pretrain", False):
                self.fusionnet = DDP(self.fusionnet, device_ids=[local_rank],
                        output_device=local_rank, find_unused_parameters=False)

        # Restart the experiment from last saved model, by loading the state of the optimizer
        if resume:
            assert training, "Finetuning EBME with auxiliary loss."
            print("Restart optimizer state to fintune with auxiliary loss.")


    def train(self):
        self.bi_flownet.train()
        self.fusionnet.train()


    def eval(self):
        self.bi_flownet.eval()
        self.fusionnet.eval()


    def device(self):
        self.bi_flownet.to(DEVICE)
        self.fusionnet.to(DEVICE)

    @staticmethod
    def convert_state_dict(rand_state_dict, pretrained_state_dict):
        param =  {
            k.replace("module.", ""): v
            for k, v in pretrained_state_dict.items()
            }
        param = {k: v
                for k, v in param.items()
                if ((k in rand_state_dict) and (rand_state_dict[k].shape == param[k].shape))
                }
        rand_state_dict.update(param)
        return rand_state_dict


    def init_ppl_modules(self, load_pretrain=True):

        def load_pretrained_state_dict(module, module_name, module_args):
            load_pretrain = module_args.load_pretrain \
                    if "load_pretrain" in module_args else True
            if not load_pretrain:
                print("Train %s from random initialization." % module_name)
                return False

            model_file = module_args.model_file \
                    if "model_file" in module_args else ""
            if (model_file == "") or (not os.path.exists(model_file)):
                raise ValueError("Please set the correct path for pretrained %s!" % module_name)

            print("Load pretrained model for %s from %s." % (module_name, model_file))
            rand_state_dict = module.state_dict()
            pretrained_state_dict = torch.load(model_file)

            return AuxPipeline.convert_state_dict(rand_state_dict, pretrained_state_dict)

        # init instance for bi_flownet and fusionnet
        bi_flownet_args = self.module_cfg_dict["bi_flownet"]
        self.bi_flownet = BiFlowNet(bi_flownet_args)
        fusionnet_args = self.module_cfg_dict["fusionnet"]
        self.fusionnet = FusionNet(fusionnet_args)

        # load pretrained model by default
        state_dict = load_pretrained_state_dict(self.bi_flownet, "bi_flownet",
                bi_flownet_args)
        if state_dict:
            self.bi_flownet.load_state_dict(state_dict)
        state_dict = load_pretrained_state_dict(self.fusionnet, "fusionnet",
                fusionnet_args)
        if state_dict:
            self.fusionnet.load_state_dict(state_dict)


    def save_optimizer_state(self, path, rank, step):
        if rank == 0:
            optimizer_ckpt = {
                     "optimizer": self.optimG.state_dict(),
                     "step": step
                     }
            torch.save(optimizer_ckpt, "{}/optimizer-ckpt.pth".format(path))


    def save_model(self, path, rank, save_step=None):
        if (rank == 0) and (save_step is None):
            torch.save(self.bi_flownet.state_dict(), '{}/bi-flownet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(), '{}/fusionnet.pkl'.format(path))
        if (rank == 0) and (save_step is not None):
            torch.save(self.bi_flownet.state_dict(), '{}/bi-flownet-{}.pkl'.format(path, save_step))
            torch.save(self.fusionnet.state_dict(), '{}/fusionnet-{}.pkl'.format(path, save_step))


    def inference(self, img0, img1, time_period=0.5):
        bi_flow = self.bi_flownet(img0, img1)
        interp_img = self.fusionnet(img0, img1, bi_flow, time_period=time_period)
        return interp_img, bi_flow

    def dice_loss(self, pred, target, smooth=1.):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        loss = 1 - dice
        return loss

    def sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum()

    def panoptic_seg_loss(self,pred,target):
        semantic_loss_dice = 0
        semantic_loss_focal = 0
        num_stuff_segments = 0
        target_masks = {}
        for mask, sinfo in target.semantic_masks():
            category_idx = sinfo["category_id"]
            target_masks[category_idx] = mask
            num_stuff_segments += 1
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            is_stuff = not sinfo["isthing"]
            if is_stuff:
                if category_idx in target_masks:
                    target_mask = target_masks[category_idx]
                    semantic_loss_dice += self.dice_loss(mask, target_mask)
                    semantic_loss_focal += self.sigmoid_focal_loss(mask, target_mask)
                else:
                    # If target does not have this category, set target mask to all zeros
                    semantic_loss_dice += 0
                    semantic_loss_focal += 0
        if num_stuff_segments > 0:
            semantic_loss_dice /= num_stuff_segments
            semantic_loss_focal /= num_stuff_segments
        
        instance_loss_dice = 0
        instance_loss_focal = 0
        num_thing_segments = 0
        target_instances = {}
        for mask, sinfo in target.instance_masks():
            category_idx = sinfo["category_id"]
            target_instances[sinfo["id"]] = mask
            num_thing_segments += 1
        for mask, sinfo in pred.instance_masks():
            category_idx = sinfo["category_id"]
            instance_id = sinfo["id"]
            is_thing = sinfo["isthing"]
            if is_thing:
                if instance_id in target_instances:
                    target_mask = target_instances[instance_id]
                    instance_loss_dice += self.dice_loss(mask, target_mask)
                    instance_loss_focal += self.sigmoid_focal_loss(mask, target_mask)
                else:
                    instance_loss_dice += 0
                    instance_loss_focal += 0

        if num_thing_segments > 0:
            instance_loss_dice /= num_thing_segments
            instance_loss_focal /= num_thing_segments
        total_focal_loss = instance_loss_focal + semantic_loss_focal
        total_dice_loss = semantic_loss_dice + instance_loss_dice
        loss = 20.0 * total_focal_loss + 1.0 * total_dice_loss
        return loss
    
    def train_one_iter(self, img0, img1, gt, seg, info, learning_rate=0,
            bi_flow_gt=None, time_period=0.5, loss_type="l2+0.1census+aux"):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()
        bi_flow = self.bi_flownet(img0, img1)
        interp_img = self.fusionnet(img0, img1, bi_flow, time_period=time_period)

        if loss_type == "l2":
            loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_G = loss_l2
        elif loss_type == "l2+census":
            loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_ter = self.ter(interp_img, gt).mean()
            loss_G = loss_l2 + loss_ter
        elif loss_type == "l2+0.1census":
            loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_ter = 0.1 * self.ter(interp_img, gt).mean()
            loss_G = loss_l2 + loss_ter
        elif loss_type == "l1":
            loss_l1 = (interp_img - gt).abs().mean()
            loss_G = loss_l1
            with torch.no_grad():
                loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
        elif loss_type == "lap":
            loss_interp_lapl1 = (self.laploss(interp_img, gt)).mean()
            loss_G = loss_interp_lapl1
            with torch.no_grad():
                loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
        elif loss_type == "l2+lap":
            loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_interp_lapl1 = (self.laploss(interp_img, gt)).mean()
            loss_G = loss_l2 + loss_interp_lapl1
        elif loss_type == "l2+0.1census+aux": 
            loss_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_ter = 0.1 * self.ter(interp_img, gt).mean()
            predicted_kpt, gt_kpt = self.HeatmapInfer(interp_img, gt)
            loss_kpt = 0.1 * self.heatmaploss(predicted_kpt, gt_kpt)

            bs = interp_img.shape[0]
            loss_panoptic = 0
            interp_img_return = interp_img.clone()
            interp_img = interp_img * 255
            assert(interp_img.shape[-3] == 3)
            interp_img = interp_img.flip(-3)
            interp_img = F.interpolate(interp_img, size=(750,1333), mode='bicubic', align_corners=False)
            panoptic_outputs = self.predictor(interp_img)
            for b in range(bs):
                cur_predict = panoptic_outputs[b]
                pred_seg = cur_predict["panoptic_seg"][0]
                pred_info = cur_predict["panoptic_seg"][1]
                cur_target_seg = seg[b, :]
                cur_target_info = info[b]
                pred = _PanopticPrediction(pred_seg, pred_info, metadata=coco_metadata)
                target = _PanopticPrediction(cur_target_seg, cur_target_info, metadata=coco_metadata)
                loss_panoptic += self.panoptic_seg_loss(pred, target)
            loss_panoptic = 0.001 * (loss_panoptic / bs)

            loss_G = loss_l2 + loss_ter + loss_panoptic + loss_kpt

        else:
            raise ValueError("Unsupported loss type!")

        self.optimG.zero_grad()
        loss_G.backward()
        self.optimG.step()


        return interp_img_return, bi_flow, loss_l2, loss_panoptic, loss_kpt



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass