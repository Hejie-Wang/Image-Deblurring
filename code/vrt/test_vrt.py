# VRT 测试（官方协议）：10 个留出视频 × 各 30 帧，5 帧输入全分辨率推理
import argparse
import csv
import math
import os

import imageio.v2 as imageio
import numpy as np
import torch

from vrt import VRT

TEST_VIDEOS = ['IMG_0003', 'IMG_0021', 'IMG_0030', 'IMG_0031', 'IMG_0032',
               'IMG_0033', 'IMG_0037', 'IMG_0039', 'IMG_0049', '720p_240fps_2']
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def load(path):
    img = imageio.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='vrt_results')
    parser.add_argument('--num_compare', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    model = VRT().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    frame_ids = np.floor(np.linspace(2, 97, 30)).astype(int) - 1
    rows = []
    compare_saved = 0
    with torch.no_grad():
        for vi, video in enumerate(TEST_VIDEOS):
            blur_dir = os.path.join(args.data_root, 'train', video, 'blur')
            sharp_dir = os.path.join(args.data_root, 'train', video, 'sharp')
            names = sorted(f for f in os.listdir(blur_dir) if f.lower().endswith(IMG_EXTS))
            psnrs = []
            for fid in frame_ids:
                frames = [load(os.path.join(blur_dir, names[fid + d])) for d in range(-2, 3)]
                sharp = load(os.path.join(sharp_dir, names[fid]))
                seq = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2)).unsqueeze(0).to(device)
                sharp_t = torch.from_numpy(sharp.transpose(2, 0, 1)).unsqueeze(0).to(device)
                out = model(seq).clamp(0, 1)
                psnrs.append(compute_psnr(out, sharp_t))
                if compare_saved < args.num_compare and vi < 4 and fid == frame_ids[15]:
                    def to_img(t):
                        return (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pair = np.concatenate([to_img(seq[0, 2:3]), to_img(out)], axis=0)
                    imageio.imwrite(os.path.join(args.save_dir, f'compare_{compare_saved:02d}_{video}.png'), pair)
                    imageio.imwrite(os.path.join(args.save_dir, f'gt_{compare_saved:02d}_{video}.png'), to_img(sharp_t))
                    compare_saved += 1
            rows.append((video, float(np.mean(psnrs))))
            print(f'{video:20s}  PSNR {np.mean(psnrs):.3f} dB', flush=True)

    avg = sum(p for _, p in rows) / len(rows)
    print(f'FINAL AVG PSNR: {avg:.3f} dB over {len(rows)} videos', flush=True)
    with open(os.path.join(args.save_dir, 'psnr.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'psnr'])
        for v, p in rows:
            w.writerow([v, f'{p:.4f}'])
        w.writerow(['average', f'{avg:.4f}'])


if __name__ == '__main__':
    main()
