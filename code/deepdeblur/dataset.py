# GoPro 数据集加载：GOPRO_Large/train/<video>/{blur_gamma,sharp}/*.png
# 数据增强：随机裁剪 256x256、随机水平/垂直翻转、随机旋转 90 度
import os
import random
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg')

class GoProDataset(Dataset):
    def __init__(self, root, split='train', crop=256, augment=True, blur_dir='blur_gamma'):
        self.pairs = []
        base = os.path.join(root, split)
        for video in sorted(os.listdir(base)):
            blur_folder = os.path.join(base, video, blur_dir)
            sharp_folder = os.path.join(base, video, 'sharp')
            if not os.path.isdir(blur_folder):
                continue
            for name in sorted(os.listdir(blur_folder)):
                if name.lower().endswith(IMG_EXTS):
                    sharp_name = name.replace('blur_gamma', 'sharp').replace('blur', 'sharp') if name.startswith('blur') else name
                    self.pairs.append((os.path.join(blur_folder, name),
                                       os.path.join(sharp_folder, sharp_name)))
        self.crop = crop
        self.augment = augment
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        img = imageio.imread(path).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    def __getitem__(self, idx):
        blur = self._load(self.pairs[idx][0])
        sharp = self._load(self.pairs[idx][1])

        if self.augment:
            h, w, _ = blur.shape
            c = self.crop
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            blur = blur[top:top + c, left:left + c]
            sharp = sharp[top:top + c, left:left + c]
            if random.random() < 0.5:
                blur, sharp = blur[:, ::-1], sharp[:, ::-1]
            if random.random() < 0.5:
                blur, sharp = blur[::-1], sharp[::-1]
            k = random.randint(0, 3)
            if k:
                blur, sharp = np.rot90(blur, k), np.rot90(sharp, k)
        else:
            h, w, _ = blur.shape
            c = min(self.crop, h, w)
            blur = blur[:c, :c]
            sharp = sharp[:c, :c]

        blur = torch.from_numpy(np.ascontiguousarray(blur.transpose(2, 0, 1)))
        sharp = torch.from_numpy(np.ascontiguousarray(sharp.transpose(2, 0, 1)))
        return blur, sharp
