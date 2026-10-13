# 逐视频搜索 unstable/stable 时间偏移（归一化互相关，多帧投票）
import os

import imageio.v2 as imageio
import numpy as np

ROOT = os.path.expanduser("~/datasets/DeepStab_frames")


def gray_small(path):
    a = imageio.imread(path).astype(np.float32).mean(axis=2)
    return a[::8, ::8]


def best_offset_for_video(v, probes=(0.2, 0.4, 0.5, 0.6, 0.8)):
    un = sorted(os.listdir(f"{ROOT}/unstable/{v}"))
    st = sorted(os.listdir(f"{ROOT}/stable/{v}"))
    votes = []
    for p in probes:
        i = int(len(un) * p)
        a = gray_small(f"{ROOT}/unstable/{v}/{un[i]}")
        scores = []
        for off in range(-60, 61):
            j = i + off
            if j < 0 or j >= len(st):
                continue
            b = gray_small(f"{ROOT}/stable/{v}/{st[j]}")
            scores.append((np.corrcoef(a.ravel(), b.ravel())[0, 1], off))
        scores.sort(reverse=True)
        votes.append(scores[0][1])
    vals, counts = np.unique(votes, return_counts=True)
    off = int(vals[np.argmax(counts)])
    return off, len(un), len(st), votes


videos = sorted(d for d in os.listdir(f"{ROOT}/unstable"))
out = {}
for v in videos:
    off, nu, ns, votes = best_offset_for_video(v)
    out[v] = off
    print(f"video {v:>4s}: offset {off:+3d}  un {nu}  st {ns}  votes {votes}", flush=True)

with open(os.path.expanduser("~/projects/stabnet/offsets.txt"), "w") as f:
    for v, off in out.items():
        f.write(f"{v} {off}\n")
print("saved offsets.txt", flush=True)
