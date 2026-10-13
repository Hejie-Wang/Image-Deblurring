# Image-Deblurring

图像去模糊与视频稳像的深度学习实验仓库（机器视觉与人工智能课程作业）。

围绕相机抖动/运动模糊问题，实现并训练了四个网络，覆盖单帧去模糊、多帧视频去模糊、时空 Transformer 去模糊、视频稳像四条技术路线。

## 结果总览

| 网络 | 类型 | 数据集 | 测试协议 | 结果 |
|---|---|---|---|---|
| **DeepDeblur** | 多尺度由粗到细 CNN（Nah et al., CVPR 2017 思路） | GoPro | 测试集全量 1111 张 | **24.85 dB** |
| **DBN** | FlowNet 光流对齐 + 多帧融合（自设计） | Su Quantitative (CVPR 2017) | 官方 10 个留出视频 × 各 30 帧 | **27.48 dB** |
| **VRT** | 光流对齐 + 时空窗口注意力 Transformer（自设计） | Su Quantitative | 同上 | **27.53 dB** |
| **StabNet** | 多网格变换回归（Wang et al., TIP 2019 思路） | DeepStab（伪 GT） | 23 个视频 ITF | **23/23 全部提升，平均 +1.5 dB** |

DBN 与 VRT 的对比：两种架构在同一数据集、同一测试协议下打平（27.48 vs 27.53 dB）。小数据量（约 6.7k 训练样本）+ 轻量配置下，Transformer 的长程时序建模优势尚未体现，光流对齐 + 卷积融合已足以覆盖帧间运动。

## 目录结构

```
code/                  PyTorch 实现（模型定义 / 数据集加载 / 训练与测试脚本）
├── deepdeblur/        多尺度 CNN 去模糊
├── dbn/               FlowNet 对齐 + 融合的视频去模糊
├── vrt/               光流对齐 + 窗口注意力的视频去模糊
└── stabnet/           多网格变换学习的视频稳像（含伪 GT 生成脚本）

results/               可视化结果与量化指标
├── deepdeblur_gopro/  compare_*.png（上：模糊输入 / 下：网络输出）、gt_*.png、psnr.csv
├── dbn_official/      同上（官方严格 train/test 划分，报告采用）
├── vrt_su/            同上
└── stabnet/           strip_*.png（抖动输入 | 稳定输出 | GT 三联条图）、psnr.csv（含 ITF）

curves/                各网络训练收敛曲线 convergence.png 与训练日志 log.csv
├── deepdeblur/  dbn/  vrt/  stabnet/

weights/               最终模型权重（final.pth）
├── deepdeblur/  dbn/  vrt/  stabnet/
```

## 复现

### 环境

- PyTorch 2.5.1 + CUDA 12.1，单机 3×A100-80GB（DDP）
- 依赖：`pip install torch numpy imageio scikit-image matplotlib tqdm`

### 数据集

| 数据集 | 来源 | 说明 |
|---|---|---|
| GoPro | [官方发布页](https://seungjunnah.github.io/Datasets/gopro) | 2103 训练对 / 1111 测试对 |
| Su Quantitative | [项目页](https://github.com/shuochsu/DeepVideoDeblurring) | 71 个手持视频；官方划分 61 训练 / 10 测试 |
| DeepStab | [TIP 2019 项目数据](https://cg.cs.tsinghua.edu.cn/people/~miao/stabnet/data.zip) | 23 对抖动/稳定 AVI |

### 训练

```bash
# DeepDeblur（3 卡 DDP，batch 24，100 epoch）
torchrun --nproc_per_node=3 code/deepdeblur/train.py --data_root <GOPRO> --batch_size 24

# DBN / VRT（Su 数据集，需先按视频整理为 <video>/{blur,sharp} 结构）
torchrun --nproc_per_node=3 code/dbn/train.py  --data_root <SuDBN_official>
torchrun --nproc_per_node=3 code/vrt/train.py  --data_root <SuDBN_official>

# StabNet：先运行 code/stabnet/make_pseudo_gt.py 生成帧级对齐伪稳定 GT
python code/stabnet/make_pseudo_gt.py
python code/stabnet/train.py --data_root <DeepStab_pseudo>
```

### 测试

```bash
python code/deepdeblur/test.py     --data_root <GOPRO> --ckpt weights/deepdeblur/final.pth
python code/dbn/test_dbn.py        --data_root <SuDBN> --ckpt weights/dbn/final.pth
python code/vrt/test_vrt.py        --data_root <SuDBN> --ckpt weights/vrt/final.pth
python code/stabnet/test_stabnet.py --data_root <DeepStab_pseudo> --ckpt weights/stabnet/final.pth
```

## 实现说明

- **测试协议**：DBN/VRT 采用 Su et al. 官方协议——10 个留出视频，每个取 `floor(linspace(2, 97, 30))` 共 30 帧，全分辨率推理计算 PSNR。
- **DBN 输入**：相邻 5 帧（NOALIGN 直接堆叠，15 通道），FlowNet 预测 4 组邻帧到中心帧的残差光流，warp 后与原始帧拼接（27 通道）送入解码器；损失为 Charbonnier + 光流一阶平滑正则。
- **VRT**：三级金字塔光流预对齐，TMSA（时空窗口注意力）与 Swin 风格窗口注意力交替，三尺度编解码 + skip 融合，输出中间帧。
- **StabNet**：编码器-解码器 + 全局池化回归 4×4/8×8 两级网格顶点位移，`grid_sample` 实现网格 warp；训练分两阶段——前 8 epoch 仅用网格位移监督（经典稳像 ORB+RANSAC 轨迹平滑得到的校正场）锚定 warp 头，之后启用全损失（重建 + 加速度时间稳定 + 形变平滑 + 网格监督）微调。
- **DeepStab 伪 GT**：官方数据中 stable 视频与 unstable 帧级时间不对齐（重渲染、非恒定偏移），故用经典稳像算法生成帧级对齐的伪稳定 GT 用于训练；ITF（帧间保真度）指标不依赖 GT，直接在原始抖动视频上计算。
