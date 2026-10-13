# DeepStab avi 抽帧：unstable/stable 各 23 个视频 -> jpg 帧序列
# 输出: ~/datasets/DeepStab_frames/{unstable,stable}/<视频名>/%05d.jpg
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

SRC = os.path.expanduser("~/datasets/DeepStab/data/datas/Regular")
DST = os.path.expanduser("~/datasets/DeepStab_frames")

jobs = []
for kind in ("unstable", "stable"):
    for name in sorted(os.listdir(os.path.join(SRC, kind))):
        if not name.endswith(".avi"):
            continue
        vid = name[:-4]
        out_dir = os.path.join(DST, kind, vid)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            continue
        os.makedirs(out_dir, exist_ok=True)
        jobs.append((os.path.join(SRC, kind, name), out_dir))

def extract(job):
    src, out = job
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src,
                    "-start_number", "0", "-q:v", "2", os.path.join(out, "%05d.jpg")], check=True)
    return f"{out}: {len(os.listdir(out))} frames"

with ThreadPoolExecutor(max_workers=8) as ex:
    for msg in ex.map(extract, jobs):
        print(msg, flush=True)
print("EXTRACTION DONE", flush=True)
