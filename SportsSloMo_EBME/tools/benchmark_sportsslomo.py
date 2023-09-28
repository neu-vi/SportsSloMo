import cv2
import math
import numpy as np
import argparse
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.unified_ppl import Pipeline
from core.dataset import SportsSloMoDataset
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def evaluate(ppl, data_root, batch_size, nr_data_worker=4, test_aug=False):
    dataset = SportsSloMoDataset(dataset_name='validation', data_root=data_root)
    val_data = DataLoader(dataset, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)

    psnr_list = []
    ssim_list = []
    ie_list = []
    nr_val = val_data.__len__()
    for i, data in enumerate(val_data):
        data_gpu, t_interp = data
        data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.

        img0 = data_gpu[:, :3]
        img1 = data_gpu[:, 3:6]
        gt = data_gpu[:, 6:9]
        t_interp_gpu = t_interp.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            if test_aug:
                pred1, _ = ppl.inference(img0, img1,time_period=t_interp_gpu)
                img0 = torch.flip(img0, dims=[2 ,3])
                img1 = torch.flip(img1, dims=[2 ,3])
                pred2, _ = ppl.inference(img0, img1)
                pred2 = torch.flip(pred2, dims=[2 ,3])
                pred = 0.5 * pred1 + 0.5 * pred2
            else:
                pred, _ = ppl.inference(img0, img1,time_period=t_interp_gpu)

        batch_psnr = []
        batch_ssim = []
        batch_ie = []
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            ssim = ssim_matlab(
                    this_pred.unsqueeze(0),
                    this_gt.unsqueeze(0)).cpu().numpy()
            ssim = float(ssim)
            ssim_list.append(ssim)
            batch_ssim.append(ssim)
            psnr = -10 * math.log10(
                    torch.mean(
                        (this_gt - this_pred) * (this_gt - this_pred)
                        ).cpu().data)
            psnr_list.append(psnr)
            batch_psnr.append(psnr)
            this_gt_ie = np.round((this_gt * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
            this_pred_ie = np.round((this_pred * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
            IE = this_gt_ie.astype(float) - this_pred_ie.astype(float)
            ie = np.mean(np.absolute(IE))
            batch_ie.append(ie)
            ie_list.append(ie)
        print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}; ie: {:.4f}'.format(i, nr_val,
            np.mean(batch_psnr), np.mean(batch_ssim), np.mean(batch_ie)))

    psnr = np.array(psnr_list).mean()
    print('average psnr: {:.4f}'.format(psnr))
    ssim = np.array(ssim_list).mean()
    print('average ssim: {:.4f}'.format(ssim))
    final_ie = np.array(ie_list).mean()
    print('average ie: {:.4f}'.format(final_ie))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on sports')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--data_root', type=str,
            help='root dir of vimeo_triplet', default="/YOURPATH_TO_SportsSloMo_frames")
    parser.add_argument('--batch_size', type=int, default=8,
            help='batch size for data loader')
    parser.add_argument('--nr_data_worker', type=int, default=4,
            help='number of the worker for data loader')

    #**********************************************************#
    # => args for optical flow model
    parser.add_argument('--flow_model_file', type=str,
            default="checkpoints/bi-flownet-1105000.pkl",
            help='weight of the bi-directional flow model')
    parser.add_argument('--pyr_level', type=int, default=3,
            help='the number of pyramid levels in testing')

    #**********************************************************#
    # => args for frame fusion (synthesis) model
    parser.add_argument('--fusion_model_file', type=str,
            default="checkpoints/fusionnet-1105000.pkl",
            help='weight of the frame fusion model')
    # set `high_synthesis` as True, only when training or loading
    # high-resolution synthesis model.
    parser.add_argument('--high_synthesis', type=bool, default=False,
            help='whether use high-resolution synthesis')

    #**********************************************************#
    # => whether use test augmentation
    parser.add_argument('--test_aug', type=bool, default=False,
            help='whether use test time augmentation')

    args = parser.parse_args()

    #**********************************************************#
    # => init the benchmarking environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    torch.backends.cudnn.benchmark = True

    #**********************************************************#
    # => init the pipeline and start to benchmark
    bi_flownet_args = argparse.Namespace()
    bi_flownet_args.pyr_level = args.pyr_level
    bi_flownet_args.load_pretrain = True
    bi_flownet_args.model_file = args.flow_model_file

    fusionnet_args = argparse.Namespace()
    fusionnet_args.high_synthesis = args.high_synthesis
    fusionnet_args.load_pretrain = True
    fusionnet_args.model_file = args.fusion_model_file

    module_cfg_dict = dict(
            bi_flownet = bi_flownet_args,
            fusionnet = fusionnet_args
            )

    ppl = Pipeline(module_cfg_dict)

    print("benchmarking on SportsSloMo...")
    evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker,
            args.test_aug)