# 经典稳像生成伪 GT：ORB 特征 + RANSAC 估计相邻帧全局仿射变换 -> 轨迹平滑 -> warp 出稳定帧
# 输出: ~/datasets/DeepStab_pseudo/{unstable,stable}/<视频名>/%05d.jpg （unstable 为原帧，stable 为伪稳定帧）
import os

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SRC = os.path.expanduser("~/datasets/DeepStab_frames/unstable")
DST = os.path.expanduser("~/datasets/DeepStab_pseudo")
SMOOTH_RADIUS = 30

def stabilize_video(name):
    out_dir = os.path.join(DST, "stable", name)
    src_dir = os.path.join(SRC, name)
    corr_file = os.path.join(DST, "corr", name + ".npy")
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0 and os.path.isfile(corr_file):
        return f"{name}: done"
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
    imgs = [cv2.imread(os.path.join(src_dir, f)) for f in frames]

    # 1) 相邻帧全局运动（2x3 仿射，ORB + RANSAC）
    transforms = [np.eye(2, 3, dtype=np.float32)]
    orb = cv2.ORB_create(2000)
    for i in range(1, len(imgs)):
        g1 = cv2.cvtColor(imgs[i - 1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
            transforms.append(transforms[-1].copy())
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(d1, d2)
        matches = sorted(matches, key=lambda m: m.distance)[:1000]
        src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        transforms.append(M if M is not None else transforms[-1].copy())

    # 2) 轨迹分解 + 滑动平均平滑
    path = np.zeros((len(imgs), 3), dtype=np.float64)  # dx, dy, da 累积轨迹
    for i, M in enumerate(transforms):
        path[i, 0] = (path[i - 1, 0] + M[0, 2]) if i else M[0, 2]
        path[i, 1] = (path[i - 1, 1] + M[1, 2]) if i else M[1, 2]
        path[i, 2] = (path[i - 1, 2] + np.arctan2(M[1, 0], M[0, 0])) if i else np.arctan2(M[1, 0], M[0, 0])
    smooth = path.copy()
    for i in range(len(path)):
        lo, hi = max(0, i - SMOOTH_RADIUS), min(len(path), i + SMOOTH_RADIUS + 1)
        smooth[i] = path[lo:hi].mean(axis=0)

    # 3) 应用平滑校正 warp 出稳定帧，并保存逐帧校正参数（供网格位移监督）
    h, w = imgs[0].shape[:2]
    corr = np.zeros((len(imgs), 3), dtype=np.float32)
    for i, img in enumerate(imgs):
        d = path[i] - smooth[i]
        corr[i] = d
        M = np.array([[np.cos(d[2]), -np.sin(d[2]), d[0]],
                      [np.sin(d[2]),  np.cos(d[2]), d[1]]], dtype=np.float32)
        out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.save(os.path.join(DST, "corr", name + ".npy"), corr)
    return f"{name}: {len(imgs)} frames stabilized"

videos = sorted(d for d in os.listdir(SRC))
os.makedirs(os.path.join(DST, "unstable"), exist_ok=True)
os.makedirs(os.path.join(DST, "corr"), exist_ok=True)
for v in videos:
    lnk = os.path.join(DST, "unstable", v)
    if not os.path.isdir(lnk):
        os.symlink(os.path.join(SRC, v), lnk)

with ThreadPoolExecutor(max_workers=6) as ex:
    for msg in ex.map(stabilize_video, videos):
        print(msg, flush=True)
print("PSEUDO GT DONE", flush=True)
