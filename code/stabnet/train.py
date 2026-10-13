# StabNet 训练（在线递归）：逐视频串行处理，prev_stab 为网络自身上一帧输出（detach）
# 梯度每 ACC_STEPS 帧累积更新一次；数据预加载到 GPU 内存
import argparse
import csv
import os
import time

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from stabnet import StabLoss, StabNet, grid_warp

VAL_VIDEOS = ['13', '15', '17']  # 留出 3 个视频做验证，其余 20 个训练
ACC_STEPS = 1
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def list_frames(root, kind, video):
    d = os.path.join(root, kind, video)
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def load_video(root, kind, video):
    names = list_frames(root, kind, video)
    frames = [imageio.imread(os.path.join(root, kind, video, n)).astype(np.float32) / 255.0 for n in names]
    return torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2))  # (T,3,H,W) CPU，逐帧搬运以省显存


def stabilize_frame(model, cur, prev_stab):
    inp = torch.cat([cur, prev_stab], dim=1)  # (1,6,H,W)
    flows = model(inp)
    stab_frames = {g: grid_warp(cur, f) for g, f in flows.items()}
    return stab_frames, flows


def vertex_gt_flow(dxdyda, H, W, grids, device):
    # 由校正仿射 (dx,dy,da) 计算 warpAffine 的采样场 M^{-1}p - p，在网格顶点处取像素位移
    dx, dy, da = float(dxdyda[0]), float(dxdyda[1]), float(dxdyda[2])
    R = np.array([[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]], dtype=np.float64)
    Minv_R, Minv_t = R.T, -R.T @ np.array([dx, dy])
    out = {}
    for g in grids:
        xs = np.linspace(-1, 1, g + 1) * ((W - 1) / 2) + (W - 1) / 2
        ys = np.linspace(-1, 1, g + 1) * ((H - 1) / 2) + (H - 1) / 2
        gx, gy = np.meshgrid(xs, ys)
        sx = Minv_R[0, 0] * gx + Minv_R[0, 1] * gy + Minv_t[0]
        sy = Minv_R[1, 0] * gx + Minv_R[1, 1] * gy + Minv_t[1]
        out[g] = torch.from_numpy(np.stack([sx - gx, sy - gy]).astype(np.float32)).to(device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='DeepStab_frames 根目录')
    parser.add_argument('--save_dir', type=str, default='experiment')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'lr', 'train_loss', 'val_psnr', 'seconds'])

    videos = sorted(d for d in os.listdir(os.path.join(args.data_root, 'unstable')))
    train_videos = [v for v in videos if v not in VAL_VIDEOS]
    print(f'{len(train_videos)} train videos, {len(VAL_VIDEOS)} val videos', flush=True)

    # 预加载到 CPU 内存（逐帧搬运 GPU）；同时加载经典稳像的逐帧校正参数用于网格位移监督
    data, corrs = {}, {}
    for v in videos:
        un = load_video(args.data_root, 'unstable', v)
        st = load_video(args.data_root, 'stable', v)
        corr = np.load(os.path.join(args.data_root, 'corr', v + '.npy'))  # (T,3): dx,dy,da
        t = min(un.shape[0], st.shape[0], corr.shape[0])
        data[v] = (un[:t], st[:t])
        corrs[v] = corr[:t]
        print(f'loaded {v}: {t} frames', flush=True)



    model = StabNet().to(device)
    grids = model.grid_sizes
    # 与注释一致：预测头零初始化 -> 初始网格变换为恒等
    for head in model.heads.values():
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 分阶段：前 3 个 epoch 仅用网格位移监督锚定 warp 场，再启用全部损失微调
    crit_mesh = StabLoss(w_rec=0.0, w_sta=0.0, w_dis=0.0, w_mesh=1.0)
    criterion = StabLoss()
    mesh_epochs = 8

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'resumed from {args.resume} at epoch {start_epoch}', flush=True)

    finest = max(model.grid_sizes)

    for epoch in range(start_epoch, args.epochs + 1):
        # 网格锚定阶段用大学习率，进入全损失微调阶段后降为 1/10
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr if epoch <= mesh_epochs else args.lr * 0.1
        model.train()
        t0 = time.time()
        losses = []
        optimizer.zero_grad()
        step = 0
        for v in train_videos:
            un, st = data[v]
            H, W = un.shape[2], un.shape[3]
            prev_stab = un[0:1].to(device)  # 首帧以自身作为"上一稳定帧"（恒等初始化）
            past = []
            print(f'  [{v}] {un.shape[0]} frames', flush=True)
            for t in range(1, un.shape[0]):
                cur, gt = un[t:t + 1].to(device, non_blocking=True), st[t:t + 1].to(device, non_blocking=True)
                stab_frames, flows = stabilize_frame(model, cur, prev_stab)
                gt_vf = vertex_gt_flow(corrs[v][t], H, W, grids, device)
                crit_now = crit_mesh if epoch <= mesh_epochs else criterion
                loss = crit_now(stab_frames, flows, gt, prev_stab=past, gt_vertex_flow=gt_vf)
                (loss / ACC_STEPS).backward()
                prev_stab = stab_frames[finest].detach()
                past.append(prev_stab)
                if len(past) > 2:
                    past.pop(0)
                losses.append(loss.item())
                step += 1
                if step % ACC_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = sum(losses) / len(losses)
        dt = time.time() - t0

        val_psnr = ''
        if epoch % args.val_every == 0:
            model.eval()
            psnrs = []
            with torch.no_grad():
                for v in VAL_VIDEOS:
                    un, st = data[v]
                    H, W = un.shape[2], un.shape[3]
                    prev_stab = un[0:1].to(device)
                    for t in range(1, un.shape[0]):
                        cur, gt = un[t:t + 1].to(device), st[t:t + 1].to(device)
                        stab_frames, _ = stabilize_frame(model, cur, prev_stab)
                        out = stab_frames[finest].clamp(0, 1)
                        mse = torch.mean((out - gt) ** 2).item()
                        psnrs.append(99.0 if mse == 0 else 10 * __import__('math').log10(1.0 / mse))
                        prev_stab = stab_frames[finest]
            val_psnr = sum(psnrs) / len(psnrs)

        lr_now = optimizer.param_groups[0]['lr']
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{lr_now:g}', f'{avg_loss:.6f}', f'{val_psnr:.4f}' if val_psnr != '' else '', f'{dt:.0f}'])
        print(f'epoch {epoch:3d}/{args.epochs}  loss {avg_loss:.6f}  psnr {val_psnr}  {dt:.0f}s', flush=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                   os.path.join(args.save_dir, 'latest.pth'))
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch{epoch:03d}.pth'))
        plot_log(log_path, os.path.join(args.save_dir, 'convergence.png'))

    print('training done.', flush=True)


def plot_log(log_path, out_path):
    epochs, losses, psnrs = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            losses.append(float(row['train_loss']))
            psnrs.append(float(row['val_psnr']) if row['val_psnr'] else None)
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(epochs, losses, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='b')
    ax2 = ax1.twinx()
    ax2.plot([e for e, p in zip(epochs, psnrs) if p], [p for p in psnrs if p], 'r.-')
    ax2.set_ylabel('val PSNR (dB)', color='r')
    ax1.set_title('StabNet training convergence')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
