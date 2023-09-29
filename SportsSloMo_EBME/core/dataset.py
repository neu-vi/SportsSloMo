import cv2
import os
import json
import torch
import numpy as np
import random
from glob import glob
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)


class SportsSloMoDataset(Dataset):
    def __init__(self, dataset_name, data_root, batch_size=32, has_aug=True):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.has_aug = has_aug
        self.crop_h = 640
        self.crop_w = 640
        self.image_root = self.data_root
        self.interp_factor = 8
        
        train_fn = "./splits/vfi_train.txt"
        test_fn = "./splits/vfi_test.txt"
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.load_data()


    def __len__(self):
        if self.dataset_name == 'train':
            return len(self.meta_data) // 9
        else: 
            return 7 * (len(self.meta_data) // 9)


    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist


    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1


    def getimg(self, index):
        base_idx = 9 * (index // 9)
        if self.dataset_name == 'train':
            # Randomly choose a target from 1-7 for training
            target_idx = np.random.randint(1, 8)
        else:
            # Sequentially choose a target from 1-7 for testing
            target_idx = (index % 7) + 1
        
        # Get image paths
        imgpath_0 = os.path.join(self.image_root, self.meta_data[base_idx])
        imgpath_target = os.path.join(self.image_root, self.meta_data[base_idx + target_idx])
        imgpath_8 = os.path.join(self.image_root, self.meta_data[base_idx + 8])
        # Compute the timestamp t for interpolation
        interp_idx = torch.Tensor([target_idx]).view(1, 1, 1)
        t_interp = interp_idx / self.interp_factor

        # Load images
        img0 = cv2.imread(imgpath_0)
        gt = cv2.imread(imgpath_target)
        img1 = cv2.imread(imgpath_8)

        return img0, gt, img1, t_interp

    def __getitem__(self, index):
        img0, gt, img1, t_interp = self.getimg(index)
        img0, gt, img1 = self.aug(img0, gt, img1, self.crop_h, self.crop_w)
        if self.dataset_name == 'train':
            if self.has_aug:
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[::-1]
                    img1 = img1[::-1]
                    gt = gt[::-1]
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, ::-1]
                    img1 = img1[:, ::-1]
                    gt = gt[:, ::-1]
                if random.uniform(0, 1) < 0.5:
                    rot_option = np.random.randint(1, 4)
                    img0 = np.rot90(img0, rot_option)
                    img1 = np.rot90(img1, rot_option)
                    gt = np.rot90(gt, rot_option)
            
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0), t_interp

if  __name__ == "__main__":
    pass

class SportsSloMoAuxDataset(Dataset):
    def __init__(self, dataset_name, data_root, batch_size=32, has_aug=True):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.has_aug = has_aug
        self.crop_h = 640
        self.crop_w = 640
        self.image_root = self.data_root
        self.interp_factor = 8
        
        train_fn = "./splits/vfi_train.txt"
        test_fn = "./splits/vfi_test.txt"
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        self.seg_root = 'YOURPATH_TO_SportsSloMo_segmentation'
        self.load_data()


    def __len__(self):
        if self.dataset_name == 'train':
            return len(self.meta_data) // 9
        else: 
            return 7 * (len(self.meta_data) // 9)


    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist


    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1


    def getimg(self, index):
        base_idx = 9 * (index // 9)
        if self.dataset_name == 'train':
            # Randomly choose a target from 1-7 for training
            target_idx = np.random.randint(1, 8)
        else:
            # Sequentially choose a target from 1-7 for testing
            target_idx = (index % 7) + 1
        
        # Get image paths
        imgpath_0 = os.path.join(self.image_root, self.meta_data[base_idx])
        imgpath_target = os.path.join(self.image_root, self.meta_data[base_idx + target_idx])
        imgpath_8 = os.path.join(self.image_root, self.meta_data[base_idx + 8])

        # Get Segmentation paths
        seg_root_target, _ = os.path.splitext(self.meta_data[base_idx + target_idx])
        seg_path_target =  os.path.join(self.seg_root, seg_root_target + '_seg.npy')
        info_path_target =  os.path.join(self.seg_root, seg_root_target + '_info.npy')

        # Compute the timestamp t for interpolation
        interp_idx = torch.Tensor([target_idx]).view(1, 1, 1)
        t_interp = interp_idx / self.interp_factor

        # Load images
        img0 = cv2.imread(imgpath_0)
        gt = cv2.imread(imgpath_target)
        img1 = cv2.imread(imgpath_8)

        # Load Segmentation Masks
        seg = np.load(seg_path_target[0], allow_pickle=True)
        info = info_path_target

        return img0, gt, img1, t_interp, seg, info

    def __getitem__(self, index):
        img0, gt, img1, t_interp, seg, info = self.getimg(index)
        img0, gt, img1 = self.aug(img0, gt, img1, self.crop_h, self.crop_w)
            
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        seg = torch.from_numpy(seg.copy())
        return torch.cat((img0, img1, gt), 0), t_interp, seg, info

if  __name__ == "__main__":
    pass

