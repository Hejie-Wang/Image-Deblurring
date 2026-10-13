# Su et al. CVPR2017 数据集整理：input/ 已是模糊帧 jpg，与 GT/ 清晰帧 jpg 同名配对
# 输出: ~/datasets/SuDBN/train/<video>/{blur,sharp}/ 原名硬链接
import os
from concurrent.futures import ThreadPoolExecutor

SRC = os.path.expanduser("~/datasets/DeepVideoDeblurring/DeepVideoDeblurring_Dataset/quantitative_datasets")
DST = os.path.expanduser("~/datasets/SuDBN/train")

videos = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d, "input")))
print(f"{len(videos)} videos found", flush=True)

def process(video):
    vdir = os.path.join(SRC, video)
    out_blur = os.path.join(DST, video, "blur")
    out_sharp = os.path.join(DST, video, "sharp")
    if os.path.isdir(out_sharp) and len(os.listdir(out_sharp)) > 0:
        return f"{video}: already done"
    os.makedirs(out_blur, exist_ok=True)
    os.makedirs(out_sharp, exist_ok=True)
    blurs = sorted(f for f in os.listdir(os.path.join(vdir, "input")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    gts = sorted(f for f in os.listdir(os.path.join(vdir, "GT")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    n = 0
    for b, g in zip(blurs, gts):
        os.link(os.path.join(vdir, "input", b), os.path.join(out_blur, b))
        os.link(os.path.join(vdir, "GT", g), os.path.join(out_sharp, g))
        n += 1
    return f"{video}: {n} pairs (blur {len(blurs)}, gt {len(gts)})"

with ThreadPoolExecutor(max_workers=8) as ex:
    for msg in ex.map(process, videos):
        print(msg, flush=True)
print("ALL PREPROCESSING DONE", flush=True)
