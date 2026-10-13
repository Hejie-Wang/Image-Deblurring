# VRT 数据集：Su 数据 5 帧输入 (T,3,H,W) + 中心帧清晰标签
import os
import random

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg')


class SuSeqDataset(Dataset):
    def __init__(self, root, split='train', crop=256, augment=True, frames=5):
        self.samples = []  # (blur_dir, sharp_dir, names, center_index)
        half = frames // 2
        base = os.path.join(root, split)
        for video in sorted(os.listdir(base)):
            blur_dir = os.path.join(base, video, 'blur')
            sharp_dir = os.path.join(base, video, 'sharp')
            if not os.path.isdir(blur_dir):
                continue
            names = sorted(f for f in os.listdir(blur_dir) if f.lower().endswith(IMG_EXTS))
            sharp_names = set(os.listdir(sharp_dir))
            for i in range(half, len(names) - half):
                if names[i] in sharp_names:
                    self.samples.append((blur_dir, sharp_dir, names, i))
        self.crop = crop
        self.augment = augment
        self.frames = frames

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        img = imageio.imread(path).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    def __getitem__(self, idx):
        blur_dir, sharp_dir, names, i = self.samples[idx]
        half = self.frames // 2
        frames = [self._load(os.path.join(blur_dir, names[i + d])) for d in range(-half, half + 1)]
        sharp = self._load(os.path.join(sharp_dir, names[i]))
        h, w, _ = frames[0].shape

        if self.augment:
            c = self.crop
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            frames = [f[top:top + c, left:left + c] for f in frames]
            sharp = sharp[top:top + c, left:left + c]
            if random.random() < 0.5:
                frames = [f[:, ::-1] for f in frames]
                sharp = sharp[:, ::-1]
            if random.random() < 0.5:
                frames = [f[::-1] for f in frames]
                sharp = sharp[::-1]
            k = random.randint(0, 3)
            if k:
                frames = [np.rot90(f, k) for f in frames]
                sharp = np.rot90(sharp, k)
        else:
            c = min(self.crop, h, w)
            frames = [f[:c, :c] for f in frames]
            sharp = sharp[:c, :c]

        seq = torch.from_numpy(np.ascontiguousarray(np.stack(frames).transpose(0, 3, 1, 2)))  # (T,3,H,W)
        sharp = torch.from_numpy(np.ascontiguousarray(sharp.transpose(2, 0, 1)))
        return seq, sharp
