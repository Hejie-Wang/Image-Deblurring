# VRT 训练脚本：Charbonnier 损失，Adam lr=1e-4，100 epoch，3 卡 DDP
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

from dataset import SuSeqDataset
from vrt import CharbonnierLoss, VRT


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='SuDBN 根目录')
    parser.add_argument('--save_dir', type=str, default='experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=12, help='总 batch（3 卡 DDP 每卡 4）')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
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

    dataset = SuSeqDataset(args.data_root, 'train', crop=args.crop, augment=True)
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = None
    else:
        sampler, shuffle = None, True
    loader = DataLoader(dataset, batch_size=args.batch_size // max(world, 1), shuffle=shuffle,
                        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    # 验证：官方 10 个测试视频上抽 30 帧（协议同 DBN），rank0 执行
    val_dataset = SuSeqDataset(args.data_root, 'train', crop=256, augment=False) if rank == 0 else None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4) if rank == 0 else None
    val_indices = set(range(0, len(val_dataset), 300)) if rank == 0 else set()

    model = VRT().to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CharbonnierLoss()
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if hasattr(model, 'module') else model).load_state_dict(ckpt['model'])
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
        for seq, sharp in loader:
            seq = seq.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)
            out = model(seq)
            loss = criterion(out, sharp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        dt = time.time() - t0

        if rank == 0:
            val_psnr = ''
            model.eval()
            psnr_sum, n = 0.0, 0
            with torch.no_grad():
                for i, (seq, sharp) in enumerate(val_loader):
                    if i not in val_indices:
                        continue
                    out = (model.module if hasattr(model, 'module') else model)(seq.to(device)).clamp(0, 1)
                    psnr_sum += compute_psnr(out, sharp.to(device))
                    n += 1
            val_psnr = psnr_sum / max(n, 1)
            lr_now = optimizer.param_groups[0]['lr']
            with open(os.path.join(args.save_dir, 'log.csv'), 'a', newline='') as f:
                csv.writer(f).writerow([epoch, f'{lr_now:g}', f'{avg_loss:.6f}', f'{val_psnr:.4f}', f'{dt:.0f}'])
            print(f'epoch {epoch:3d}/{args.epochs}  loss {avg_loss:.6f}  psnr {val_psnr:.4f}  {dt:.0f}s', flush=True)
            torch.save({'model': (model.module if hasattr(model, 'module') else model).state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       os.path.join(args.save_dir, 'latest.pth'))
            if epoch % 10 == 0 or epoch == args.epochs:
                torch.save((model.module if hasattr(model, 'module') else model).state_dict(),
                           os.path.join(args.save_dir, f'epoch{epoch:03d}.pth'))
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
    ax1.plot(epochs, losses, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='b')
    ax2 = ax1.twinx()
    ax2.plot([e for e, p in zip(epochs, psnrs) if p], [p for p in psnrs if p], 'r.-')
    ax2.set_ylabel('val PSNR (dB)', color='r')
    ax1.set_title('VRT training convergence')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
