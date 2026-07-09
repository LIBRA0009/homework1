## DG-SegFormer for SUIM Underwater Semantic Segmentation

本仓库基于 MMSegmentation 1.x，面向 SUIM 水下语义分割数据集实现了
SegFormer-B2 RGB baseline、DepthConcat、Geometry-only 和 DGMF
（Depth-Geometric Modulation Fusion）等实验。核心目标是验证单目相对深度、
局部几何变化和深度边缘先验对水下语义分割的作用。

> 原始 MMSegmentation README 请参考官方仓库。本节是本课程项目的说明。

### 1. 环境依赖

推荐服务器环境如下：

```text
Python          3.10
PyTorch         2.1.0 + CUDA 12.1
Torchvision     0.16.0
MMCV            2.1.0
MMEngine        0.10.7
MMSegmentation  1.2.2
NumPy           1.26.4
GPU             4 × NVIDIA GeForce RTX 4090 D
```

基础依赖安装：

```bash
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
pip install -r requirements.txt
pip install -v -e .
```

如果需要使用 CLIP/open-clip 相关模块，额外安装：

```bash
pip install -r requirements/multimodal.txt
```

Depth Anything V2 不包含在本仓库中，需要单独下载官方代码和权重。本项目实验使用：

```text
Depth-Anything-V2 root: /data/yjm/2026/waterhomework/Depth-Anything-V2-main
checkpoint: checkpoints/depth_anything_v2_vits.pth
```

### 2. 数据准备

原始 SUIM 数据目录示例：

```text
datasets/
├── train_val/
│   ├── images/
│   └── masks/
└── TEST/
    ├── images/
    └── masks/
```

SUIM 原始数据存在两个主要问题：

1. 部分 RGB 图像与 mask 尺寸不一致；
2. 彩色 mask 中存在接近标准色但非标准色的污染像素。

因此需要先进行清洗和格式转换：

```bash
python tools/dataset_converters/suim.py \
  /data/yjm/2026/waterhomework/mmsegmentation/datasets \
  -o /data/yjm/2026/waterhomework/mmsegmentation/datasets_suim_clean
```

再整理为 MMSegmentation 训练格式：

```bash
python tools/dataset_converters/suim_prepare.py \
  /data/yjm/2026/waterhomework/mmsegmentation/datasets_suim_clean \
  -o data/SUIM \
  --seed 42 \
  --val-ratio 0.2
```

整理后的数据结构：

```text
data/SUIM/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
├── depth/
│   ├── train/
│   └── val/
└── splits/
    ├── train.txt
    └── val.txt
```

当前实验使用清洗后的 `train_val` 数据划分：

```text
train: 1190 images
val:   298 images
test:  0 images
```

### 3. 离线生成相对深度

使用 Depth Anything V2-Small 生成相对深度图：

```bash
python tools/generate_suim_depth.py \
  --data-root data/SUIM \
  --checkpoint checkpoints/depth_anything_v2_vits.pth \
  --encoder vits \
  --depth-anything-root /data/yjm/2026/waterhomework/Depth-Anything-V2-main \
  --device cuda:0
```

输出为：

```text
data/SUIM/depth/train/*.npy
data/SUIM/depth/val/*.npy
data/SUIM/depth_vis/*.jpg
```

深度图采用逐图 P2/P98 百分位归一化，保存为 `[0, 1]` 范围内的
`float32 .npy`。

### 4. 主要模型与配置

| 方法 | 配置文件 | 说明 |
|---|---|---|
| RGB baseline | `configs/segformer/segformer_mit-b2_8xb8-40k_suim-512x512.py` | 原始 SegFormer-B2 RGB 输入 |
| DepthConcat | `configs/segformer/segformer_mit-b2_8xb8-17k_suim-depthconcat-512x512.py` | RGB 与 depth 拼成 4 通道输入 |
| Geometry-only | `configs/segformer/segformer_mit-b2_8xb8-17k_suim-geometry-512x512.py` | 仅使用 RGB 高层局部几何增强 |
| DGMF λ=0.1 | `configs/segformer/segformer_mit-b2_8xb8-17k_suim-dgmf-512x512.py` | 深度-几何残差融合，λ=0.1 |
| DGMF λ=0.05 | `configs/segformer/segformer_mit-b2_8xb8-17k_suim-dgmf-r005-512x512.py` | 深度-几何残差融合，λ=0.05 |

DGMF 的输入数据管线输出 RGB 与深度组成的四通道张量：

```text
I_4 = Concat(I_rgb, D),  I_4 ∈ R^{B×4×H×W}
```

模型内部将四通道输入拆分为 RGB 图像和单通道深度图：

```text
I_rgb = I_4[:, 0:3, :, :]
D     = I_4[:, 3:4, :, :]
```

其中 RGB 图像输入 MiT-B2 backbone，深度图只送入 DGMF 模块，用于构造
depth feature、depth edge 和 depth-geometric residual。

### 5. 训练命令

RGB baseline：

```bash
python tools/train.py \
  configs/segformer/segformer_mit-b2_8xb8-40k_suim-512x512.py
```

DepthConcat：

```bash
python tools/train.py \
  configs/segformer/segformer_mit-b2_8xb8-17k_suim-depthconcat-512x512.py
```

Geometry-only：

```bash
python tools/train.py \
  configs/segformer/segformer_mit-b2_8xb8-17k_suim-geometry-512x512.py
```

DGMF λ=0.1：

```bash
python tools/train.py \
  configs/segformer/segformer_mit-b2_8xb8-17k_suim-dgmf-512x512.py
```

DGMF λ=0.05：

```bash
python tools/train.py \
  configs/segformer/segformer_mit-b2_8xb8-17k_suim-dgmf-r005-512x512.py
```

### 6. 实验结果

所有结果均选择验证集 mIoU 最高的 checkpoint。

| 方法 | best iter | mIoU | mDice | mAcc | aAcc |
|---|---:|---:|---:|---:|---:|
| RGB baseline | 13000 | 71.28 | 81.56 | 80.58 | 86.00 |
| DepthConcat | 14000 | 71.91 | 82.14 | 81.00 | 86.20 |
| Geometry-only | 16000 | 71.82 | 81.98 | 81.16 | 86.34 |
| DGMF λ=0.1 | 14000 | 71.64 | 81.83 | 80.75 | 85.98 |
| DGMF λ=0.05 | 14000 | 71.93 | 82.03 | 80.94 | 86.19 |

逐类别 IoU：

| 方法 | BW | HD | PF | WR | RO | RI | FV | SR | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RGB baseline | 86.16 | 78.57 | 29.49 | 75.13 | 90.55 | 79.36 | 77.35 | 53.66 | 71.28 |
| DepthConcat | 87.43 | 78.49 | 32.50 | 76.89 | 89.71 | 78.36 | 78.90 | 52.95 | 71.91 |
| Geometry-only | 87.00 | 80.48 | 30.56 | 76.38 | 88.92 | 79.20 | 78.50 | 53.54 | 71.82 |
| DGMF λ=0.1 | 87.25 | 79.68 | 30.35 | 76.75 | 90.17 | 78.04 | 77.93 | 52.94 | 71.64 |
| DGMF λ=0.05 | 87.31 | 80.13 | 30.11 | 77.00 | 90.59 | 78.29 | 77.48 | 54.55 | 71.93 |

当前最好结果为 `DGMF λ=0.05`，mIoU 为 71.93。需要注意的是，
其相对 DepthConcat 的提升仅为 0.02 个百分点，因此报告中不应夸大复杂融合
相对简单深度拼接的优势。

### 7. 结果汇总与可视化

汇总实验结果：

```bash
python tools/summarize_suim_experiments.py \
  --work-dirs work_dirs \
  --out-dir main_exp/experiment_results
```

输出包括：

```text
summary_metrics.csv
leaderboard_best_miou.csv
per_class_iou.csv
delta_vs_baseline.csv
experiment_report_tables.md
fig_best_miou_zoom.svg
fig_per_class_iou.svg
```

生成定性可视化样本：

```bash
python tools/visualize_suim_predictions.py \
  configs/segformer/segformer_mit-b2_8xb8-17k_suim-dgmf-r005-512x512.py \
  work_dirs/segformer_mit-b2_8xb8-17k_suim-dgmf-r005-512x512/best_mIoU_iter_14000.pth \
  --split val \
  --out-dir work_dirs/vis_dgmf_r005 \
  --num-samples 24 \
  --show-depth \
  --device cuda:0
```

输出拼图格式：

```text
RGB image | relative depth | ground truth | prediction | error map
```

### 8. 项目新增代码

| 文件 | 作用 |
|---|---|
| `tools/dataset_converters/suim.py` | SUIM 彩色 mask 清洗与标签转换 |
| `tools/dataset_converters/suim_prepare.py` | 清洗后数据划分和 MMSeg 格式整理 |
| `tools/generate_suim_depth.py` | Depth Anything V2 离线深度生成 |
| `tools/model_converters/segformer_pretrain_3ch_to_4ch.py` | SegFormer 3 通道预训练权重扩展为 4 通道 |
| `tools/visualize_suim_predictions.py` | 生成定性分割可视化样本 |
| `tools/summarize_suim_experiments.py` | 汇总实验表格和图 |
| `mmseg/datasets/suim.py` | SUIMDataset 注册 |
| `mmseg/datasets/transforms/loading.py` | `LoadSUIMDepthAsChannel` 深度通道加载 |
| `mmseg/models/decode_heads/geometry_segformer_head.py` | Geometry-only head |
| `mmseg/models/segmentors/dgmf_encoder_decoder.py` | DGMF 输入拆分和前向逻辑 |
| `mmseg/models/decode_heads/dgmf_segformer_head.py` | DGMF 融合模块 |

### 9. 仍需补充或注意

1. 当前结果主要来自单次训练，尚未进行多随机种子均值和标准差实验。
2. 官方 TEST 语义标签尚未完整纳入当前清洗和评价流程。
3. Depth Anything V2 输出为相对深度，不代表真实物理距离。

---