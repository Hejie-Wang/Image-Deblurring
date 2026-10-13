# DBN 训练脚本（网络为用户设计的 FlowNet+warp+decoder，见 models.py）
# Charbonnier + flow 平滑正则（DBNLoss），Adam lr=1e-4，100 epoch，3 卡 DDP
import argparse
import csv
import math
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import SuVideoDataset
from models import DBNLoss, DeBlurNet


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='SuDBN 数据集根目录')
    parser.add_argument('--save_dir', type=str, default='experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=24, help='总 batch size（3 卡 DDP 下每卡 8）')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--flow_w', type=float, default=0.01)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    ddp = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank, world = 0, 1

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        log_path = os.path.join(args.save_dir, 'log.csv')
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'lr', 'train_loss', 'val_psnr', 'seconds'])

    dataset = SuVideoDataset(args.data_root, 'train', crop=args.crop, augment=True)
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = None
    else:
        sampler, shuffle = None, True
    loader = DataLoader(dataset, batch_size=args.batch_size // max(world, 1), shuffle=shuffle,
                        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    val_dataset = SuVideoDataset(args.data_root, 'train', crop=256, augment=False) if rank == 0 else None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4) if rank == 0 else None
    val_indices = set(range(0, len(val_dataset), 200)) if rank == 0 else set()

    model = DeBlurNet().to(device)
    model.init_weights()
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 用 hook 捕获 flownet 的 (B,8,H,W) 原始流输出，供 DBNLoss 平滑正则使用
    net = model.module if hasattr(model, 'module') else model
    flow_capture = {}
    net.flownet.register_forward_hook(lambda m, i, o: flow_capture.__setitem__('flow', o))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DBNLoss(flow_w=args.flow_w)
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if rank == 0:
            print(f'resumed from {args.resume} at epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        losses = []
        for blur, sharp in loader:
            blur = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)
            out = model(blur)
            loss = criterion(out, sharp, flow_capture['flow'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        dt = time.time() - t0

        if rank == 0:
            val_psnr = ''
            if epoch % args.val_every == 0:
                model.eval()
                psnr_sum, n = 0.0, 0
                with torch.no_grad():
                    for i, (blur, sharp) in enumerate(val_loader):
                        if i not in val_indices:
                            continue
                        out = net(blur.to(device)).clamp(0, 1)
                        psnr_sum += compute_psnr(out, sharp.to(device))
                        n += 1
                val_psnr = psnr_sum / max(n, 1)
                model.train()
            lr_now = optimizer.param_groups[0]['lr']
            with open(os.path.join(args.save_dir, 'log.csv'), 'a', newline='') as f:
                csv.writer(f).writerow([epoch, f'{lr_now:g}', f'{avg_loss:.6f}', f'{val_psnr:.4f}' if val_psnr != '' else '', f'{dt:.0f}'])
            print(f'epoch {epoch:3d}/{args.epochs}  loss {avg_loss:.6f}  psnr {val_psnr}  {dt:.0f}s', flush=True)
            torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(args.save_dir, 'latest.pth'))
            if epoch % 10 == 0 or epoch == args.epochs:
                torch.save(net.state_dict(), os.path.join(args.save_dir, f'epoch{epoch:03d}.pth'))
            plot_log(os.path.join(args.save_dir, 'log.csv'), os.path.join(args.save_dir, 'convergence.png'))

    if rank == 0:
        print('training done.')


def plot_log(log_path, out_path):
    epochs, losses, psnrs = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            losses.append(float(row['train_loss']))
            psnrs.append(float(row['val_psnr']) if row['val_psnr'] else None)
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(epochs, losses, 'b-', label='train loss (Charbonnier+flow)')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='b')
    ax2 = ax1.twinx()
    ax2.plot([e for e, p in zip(epochs, psnrs) if p], [p for p in psnrs if p], 'r.-', label='val PSNR (dB)')
    ax2.set_ylabel('PSNR (dB)', color='r')
    ax1.set_title('DBN training convergence')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
