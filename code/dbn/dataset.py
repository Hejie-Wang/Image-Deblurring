# Su 数据集加载（NOALIGN 版本）：中心帧 t 的输入为 [t-2, t+2] 共 5 帧模糊帧直接堆叠
# 数据增强：5 帧同参数随机裁剪/翻转/旋转
import os
import random
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg')

class SuVideoDataset(Dataset):
    def __init__(self, root, split='train', crop=256, augment=True, frames=5):
        self.samples = []  # (blur_dir, sharp_dir, names, center_index)
        self.frames = frames
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
        stack = [self._load(os.path.join(blur_dir, names[i + d])) for d in range(-half, half + 1)]
        sharp = self._load(os.path.join(sharp_dir, names[i]))
        h, w, _ = stack[0].shape

        if self.augment:
            c = self.crop
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            stack = [s[top:top + c, left:left + c] for s in stack]
            sharp = sharp[top:top + c, left:left + c]
            if random.random() < 0.5:
                stack = [s[:, ::-1] for s in stack]
                sharp = sharp[:, ::-1]
            if random.random() < 0.5:
                stack = [s[::-1] for s in stack]
                sharp = sharp[::-1]
            k = random.randint(0, 3)
            if k:
                stack = [np.rot90(s, k) for s in stack]
                sharp = np.rot90(sharp, k)
        else:
            c = min(self.crop, h, w)
            stack = [s[:c, :c] for s in stack]
            sharp = sharp[:c, :c]

        blur = np.concatenate(stack, axis=-1)  # H W 15
        blur = torch.from_numpy(np.ascontiguousarray(blur.transpose(2, 0, 1)))
        sharp = torch.from_numpy(np.ascontiguousarray(sharp.transpose(2, 0, 1)))
        return blur, sharp
