# DeepDeblur 测试脚本：GoPro test 集上计算 PSNR，并输出模糊/恢复对比图
import argparse
import csv
import math
import os

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GoProDataset
from models import DeepDeblurNet


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='test_results')
    parser.add_argument('--max_images', type=int, default=0, help='0 表示跑全测试集')
    parser.add_argument('--num_compare', type=int, default=4, help='输出对比图的张数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    model = DeepDeblurNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = GoProDataset(args.data_root, 'test', crop=512, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    psnr_list = []
    compare_saved = 0
    with torch.no_grad():
        for i, (blur, sharp) in enumerate(loader):
            if args.max_images and i >= args.max_images:
                break
            blur, sharp = blur.to(device), sharp.to(device)
            out = model(blur)[2].clamp(0, 1)
            p = compute_psnr(out, sharp)
            psnr_list.append(p)
            if compare_saved < args.num_compare and (i % 300 == 0 or i == len(loader) - 1):
                def to_img(t):
                    return (t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                b_img, o_img, s_img = to_img(blur), to_img(out), to_img(sharp)
                # 上：模糊输入；下：网络恢复输出
                pair = np.concatenate([b_img, o_img], axis=0)
                imageio.imwrite(os.path.join(args.save_dir, f'compare_{compare_saved:02d}_idx{i:04d}.png'), pair)
                imageio.imwrite(os.path.join(args.save_dir, f'gt_{compare_saved:02d}_idx{i:04d}.png'), s_img)
                compare_saved += 1
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(loader)}  running psnr {sum(psnr_list) / len(psnr_list):.3f} dB', flush=True)

    avg = sum(psnr_list) / len(psnr_list)
    print(f'FINAL PSNR: {avg:.3f} dB on {len(psnr_list)} images', flush=True)
    with open(os.path.join(args.save_dir, 'psnr.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'psnr'])
        for i, p in enumerate(psnr_list):
            w.writerow([i, f'{p:.4f}'])
        w.writerow(['average', f'{avg:.4f}'])


if __name__ == '__main__':
    main()
