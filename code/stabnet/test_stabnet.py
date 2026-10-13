# StabNet 预测：对留出验证视频做在线稳定化，输出 PSNR + 原始/稳定/真值对比条图
import argparse
import csv
import math
import os

import imageio.v2 as imageio
import numpy as np
import torch

from stabnet import StabNet, grid_warp

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def load_video(root, kind, video, device):
    d = os.path.join(root, kind, video)
    names = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
    frames = [imageio.imread(os.path.join(d, n)).astype(np.float32) / 255.0 for n in names]
    return torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2)).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='stab_results')
    parser.add_argument('--videos', type=str, default='')
    parser.add_argument('--save_gif', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)
    model = StabNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    finest = max(model.grid_sizes)
    videos = args.videos.split(',') if args.videos else sorted(
        d for d in os.listdir(os.path.join(args.data_root, 'unstable')))

    rows = []
    with torch.no_grad():
        for v in videos:
            un = load_video(args.data_root, 'unstable', v, device)
            try:
                st = load_video(args.data_root, 'stable', v, device)
            except Exception:
                st = None
            t = min(un.shape[0], st.shape[0]) if st is not None else un.shape[0]
            prev_stab = un[0:1]
            outs, psnrs = [], []
            itf_stab, itf_raw = [], []
            for tt in range(t):
                inp = torch.cat([un[tt:tt + 1], prev_stab], dim=1)
                flows = model(inp)
                out = grid_warp(un[tt:tt + 1], flows[finest]).clamp(0, 1)
                outs.append(out[0])
                if tt > 0:
                    # ITF：相邻帧 PSNR（越高越平稳），对原始帧与稳定帧分别计算
                    mse_s = torch.mean((out - outs[-2]) ** 2).item()
                    itf_stab.append(10 * math.log10(1.0 / max(mse_s, 1e-10)))
                    mse_r = torch.mean((un[tt] - un[tt - 1]) ** 2).item()
                    itf_raw.append(10 * math.log10(1.0 / max(mse_r, 1e-10)))
                if st is not None:
                    mse = torch.mean((out - st[tt:tt + 1]) ** 2).item()
                    psnrs.append(99.0 if mse == 0 else 10 * math.log10(1.0 / mse))
                prev_stab = out
            if psnrs:
                rows.append((v, sum(psnrs) / len(psnrs)))
                print(f'{v:10s}  PSNR {rows[-1][1]:.3f} dB  ITF raw {np.mean(itf_raw):.2f} -> stab {np.mean(itf_stab):.2f}  ({t} frames)', flush=True)
            else:
                rows.append((v, np.mean(itf_stab)))
                print(f'{v:10s}  ITF raw {np.mean(itf_raw):.2f} -> stab {np.mean(itf_stab):.2f}  ({t} frames)', flush=True)
            # 对比条图：抖动输入 | 稳定输出 | GT 稳定
            idx = t // 2
            panels = [un[idx].cpu().numpy().transpose(1, 2, 0),
                      outs[idx].cpu().numpy().transpose(1, 2, 0)]
            if st is not None:
                panels.append(st[idx].cpu().numpy().transpose(1, 2, 0))
            strip = np.concatenate(panels, axis=1)
            imageio.imwrite(os.path.join(args.save_dir, f'strip_{v}.png'), (strip * 255).astype(np.uint8))
            if args.save_gif:
                gif_frames = [(torch.cat([un[i], outs[i].clamp(0, 1)], dim=2).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                              for i in range(0, t, 2)]
                imageio.mimsave(os.path.join(args.save_dir, f'compare_{v}.gif'), gif_frames, duration=0.05)

    if rows:
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
