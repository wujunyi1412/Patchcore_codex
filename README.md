# Minimal PatchCore

这是一个精简版 PatchCore 实现，主要覆盖三件事：

- 从正常样本图片中训练 memory bank
- 对单张图片或文件夹做推理
- 将模型侧的特征抽取部分导出为 ONNX

项目不再依赖 MVTec 固定目录结构，也不再依赖之前较重的脚本流程。

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 训练

训练输入只需要一个正常图片文件夹。

```bash
python train.py \
  --train-dir dataset/train/good \
  --output-dir artifacts/wr50_bank \
  --backbone wide_resnet50_2 \
  --image-size 320
```

常用参数：

- `--layers layer2 layer3`
- `--sampler approx_greedy_coreset --sample-ratio 0.1`
- `--preprocess gaussian_blur`
- `--cpu`

使用 DINOv2 backbone：

```bash
python train.py \
  --train-dir dataset/train/good \
  --output-dir artifacts/dinov2_bank \
  --backbone dinov2_vits14 \
  --image-size 320
```

当前支持的 DINOv2 backbone：

- `dinov2_vits14`
- `dinov2_vitb14`
- `dinov2_vitl14`
- `dinov2_vitg14`

这些模型通过 `torch.hub.load("facebookresearch/dinov2", ...)` 加载，第一次运行时需要能访问
PyTorch Hub 和模型权重下载地址。

采样器选项：

- `identity`：保留全部 patch embeddings
- `random`：随机降采样
- `greedy_coreset`：质量通常更好，但训练阶段最慢
- `approx_greedy_coreset`：大 memory bank 场景下推荐，速度和质量比较均衡

## 推理

推理输入可以是一张图片，也可以是一个图片文件夹。

```bash
python infer.py \
  --model-dir artifacts/wr50_bank \
  --input dataset/test \
  --output-dir infer_out \
  --threshold 1.0
```

输出：

- `infer_out/results.csv`
- `infer_out/above_threshold/`
- `infer_out/below_threshold/`

## 导出 ONNX

这里只导出模型侧，也就是预处理后的图片到 patch embedding 的部分。
FAISS index 和距离搜索仍然在 ONNX 外部执行。

```bash
python convert_to_onnx.py \
  --model-dir artifacts/wr50_bank \
  --output-dir onnx_out
```

输出：

- `onnx_out/patchcore_embedder.onnx`
- `onnx_out/onnx_metadata.json`

## 代码结构

核心代码在 `src/patchcore/`：

- `preprocess.py`：图片读取、resize、padding、normalize
- `backbones.py`：backbone 注册和加载
- `embedding.py`：patchify 和 embedding 聚合
- `common.py`：FAISS index 和 mask 上采样
- `sampler.py`：memory bank 采样策略
- `patchcore.py`：训练和推理主流程
- `onnx.py`：ONNX 导出包装

## 特征抽取和得分流程

这一节按当前代码实现来解释，主要对应：

- `src/patchcore/patchcore.py`
- `src/patchcore/embedding.py`
- `src/patchcore/common.py`

符号说明：

- `B`：batch size。本实现通常一次处理一张图，所以 `B=1`。
- `S`：输入正方形尺寸，来自 `--image-size`。
- `L`：选择的 backbone 特征层数量，默认通常是 `layer2 layer3`，所以 `L=2`。
- `C_l, H_l, W_l`：第 `l` 个特征层的通道数、高、宽。
- `P`：patch size，来自 `--patch-size`，默认 `3`。
- `D_pre`：单层 patch pooling 后的维度，来自 `--pretrain-embed-dim`，默认 `1024`。
- `D`：最终 patch embedding 维度，来自 `--target-embed-dim`，默认 `1024`。
- `Q`：单张图片产生的 query patch embedding 数量。
- `M`：采样后的 memory bank embedding 数量。
- `K`：最近邻数量，来自 `--num-neighbors`，默认 `1`。

### 1. 图片预处理

`ImagePreprocessor` 会读取图片，转成 RGB，可选做简单滤波，然后保持长宽比 resize，
再 padding 成正方形，最后转 tensor 并做 ImageNet normalize。

维度变化：

```text
原始图片                         [H0, W0, 3]
resize + padding 成正方形          [S, S, 3]
ToTensor + Normalize + batch       [B, 3, S, S]
```

例如 `--image-size 320` 时，模型输入是：

```text
[1, 3, 320, 320]
```

### 2. Backbone 抽取特征图

`FeatureExtractor` 会在配置的层上注册 forward hook，运行 torchvision backbone，
然后按层顺序返回捕获到的 feature maps。

默认 `wide_resnet50_2` 使用 `layer2 layer3`。这两层典型空间 stride 分别是输入图的
`1/8` 和 `1/16`：

```text
输入图片 tensor       [B, 3, S, S]
layer2 feature map    [B, C2, S/8,  S/8]
layer3 feature map    [B, C3, S/16, S/16]
```

当 `S=320` 时，通常是：

```text
layer2    [1, 512, 40, 40]
layer3    [1, 1024, 20, 20]
```

具体通道数取决于选择的 backbone 和 layer 名称。

### 3. 对每个 feature map 做 patchify

`PatchMaker.patchify()` 使用 `torch.nn.Unfold` 对每个 feature map 滑窗取 patch。
默认 `patch_size=3`、`patch_stride=1`，并使用对称 padding，所以 patch 网格的高宽
和原 feature map 的高宽一致。

对单层 feature map：

```text
feature map                         [B, C_l, H_l, W_l]
Unfold 之后                         [B, C_l * P * P, H_l * W_l]
reshape + permute 之后              [B, H_l * W_l, C_l, P, P]
patch_shape                         (H_l, W_l)
```

例如 `S=320`、`layer2`、`P=3`：

```text
[1, 512, 40, 40] -> [1, 1600, 512, 3, 3]
patch_shape = (40, 40)
```

### 4. 对齐不同层的 patch 网格

PatchCore 会融合多个特征层的 patch。由于更深的层空间分辨率更小，
`_align_patch_grid()` 会把所有层的 patch 网格上采样到第一层的 patch 网格大小。

默认层顺序是 `layer2 layer3`，所以参考网格是 `layer2` 的 `(40, 40)`：

```text
layer2 patches    [B, 40*40, 512, 3, 3]     已经是 (40, 40)
layer3 patches    [B, 20*20, 1024, 3, 3]    需要对齐到 (40, 40)
aligned layer3    [B, 40*40, 1024, 3, 3]
```

这一步之后，每个特征层都有相同数量的 patch 位置：

```text
Q = B * H_ref * W_ref
```

对于一张 `320 x 320` 图片，并以 `layer2` 为参考网格：

```text
Q = 1 * 40 * 40 = 1600
```

### 5. 每层 patch pooling 到 `D_pre`

每个对齐后的 patch 会把通道和局部 patch 维度展开，然后用
`adaptive_avg_pool1d` 压到 `D_pre`。

对每个特征层：

```text
aligned patches                  [B, H_ref*W_ref, C_l, P, P]
合并 batch 和 patch 位置          [Q, C_l, P, P]
展开 patch 内容                   [Q, 1, C_l * P * P]
adaptive_avg_pool1d 到 D_pre      [Q, D_pre]
```

默认情况下，每个选中的特征层都会贡献一个 `[Q, 1024]` 矩阵，
不管它原本的通道数是多少。

### 6. 融合多层，得到最终 patch embeddings

`PatchcoreEmbedder.forward()` 会把每层 pooled vectors stack 起来，
再展平 layer 维度，最后再次使用 `adaptive_avg_pool1d` 压到最终维度 `D`。

```text
每层 pooled 结果列表        L 个 [Q, D_pre]
torch.stack(dim=1)          [Q, L, D_pre]
reshape                     [Q, 1, L * D_pre]
adaptive_avg_pool1d 到 D    [Q, D]
```

默认 `L=2`、`D_pre=1024`、`D=1024`：

```text
layer2 pooled               [1600, 1024]
layer3 pooled               [1600, 1024]
stack                       [1600, 2, 1024]
flatten                     [1600, 1, 2048]
final embeddings            [1600, 1024]
patch_shape                 (40, 40)
```

训练时存进 memory bank、推理时拿去查最近邻的，就是这些最终 patch embeddings。

### 7. 训练：构建 memory bank

`PatchCoreModel.fit()` 会对所有正常训练图片执行上面的 embedding 流程，
把所有图片的 patch embeddings 拼接起来，然后可选采样，最后构建 FAISS
`IndexFlatL2`。

对于 `N` 张训练图：

```text
单张图片 embeddings               [Q, D]
全部训练 embeddings               [N * Q, D]
采样之后                          [M, D]
FAISS memory bank                 M 个 D 维向量
```

如果 `N=100`、`S=320`、默认层配置，并且不采样：

```text
Q = 1600
全部训练 embeddings = [160000, 1024]
```

如果使用 `--sampler approx_greedy_coreset --sample-ratio 0.1`，memory bank 大约是：

```text
M = 160000 * 0.1 = 16000
memory bank = [16000, 1024]
```

保存的文件：

```text
model_config.json     配置和 memory bank 统计信息
memory_bank.faiss     FAISS L2 搜索 index
```

### 8. 推理：最近邻距离得到 patch 分数

推理时，测试图片会先被转成 query patch embeddings，然后拿这些 query 去 FAISS
memory bank 里查最近邻。

```text
测试图片 tensor              [1, 3, S, S]
query embeddings             [Q, D]
memory bank                  [M, D]
FAISS 返回 distances         [Q, K]
FAISS 返回 indices           [Q, K]
```

`IndexFlatL2` 返回的是平方 L2 距离。当前代码会对每个 query patch 的 `K` 个距离取平均：

```text
patch_scores = mean(distances, axis=1)
patch_scores shape = [Q]
```

当默认 `K=1` 时，这就是每个 patch 到 memory bank 最近邻的距离。

### 9. 图片分数和热力图

图片级异常分数取所有 patch 分数中的最大值：

```text
image_score = max(patch_scores)
```

然后把 `patch_scores` reshape 回参考 feature grid，再上采样到输入图大小：

```text
patch_scores                  [Q]
patch_map                     [1, H_ref, W_ref]
bilinear upsample             [1, S, S]
gaussian smoothing            [S, S]
```

对于 `S=320`、默认层配置：

```text
patch_scores                  [1600]
patch_map                     [1, 40, 40]
mask                          [320, 320]
```

`infer.py` 会把这个 mask 覆盖到处理后的图片上，并把标量 `image_score` 写入
`results.csv`。阈值判断是在 PatchCore 外部做的：

```text
image_score >= threshold -> ANOMALY
image_score < threshold  -> OK
```

### 默认配置的完整维度例子

一张 `320 x 320` 图片，使用 `wide_resnet50_2`、`layer2 layer3`、`patch_size=3`、
`D_pre=1024`、`D=1024`：

```text
image                  [1, 3, 320, 320]
layer2                 [1, 512, 40, 40]
layer3                 [1, 1024, 20, 20]
layer2 patchify        [1, 1600, 512, 3, 3]
layer3 patchify        [1, 400, 1024, 3, 3]
layer3 aligned         [1, 1600, 1024, 3, 3]
layer2 pooled          [1600, 1024]
layer3 pooled          [1600, 1024]
merged embeddings      [1600, 1024]
FAISS distances        [1600, K]
patch_scores           [1600]
patch_map              [1, 40, 40]
mask                   [320, 320]
image_score            对 1600 个 patch_scores 取最大值后的标量
```

## 更换 Backbone

Backbone 切换集中在 config 和 `src/patchcore/backbones.py`。

ResNet 类 backbone 直接通过 forward hook 抽取 `layer2/layer3` 这类 CNN feature map。
DINOv2 是 ViT，不会天然输出 `[B, C, H, W]` 的 CNN feature map，所以代码里加了
`DinoV2FeatureMapBackbone` 包装：

- 默认层名使用 transformer block，例如 `blocks.5 blocks.11`。
- wrapper 调用 DINOv2 的 `get_intermediate_layers(..., reshape=True)`。
- DINOv2 patch tokens 会被还原成 `[B, C, H_token, W_token]`。
- 后续 `patchify -> embedding -> FAISS` 流程保持不变。

DINOv2 默认 patch size 是 `14`。如果 `--image-size 320`，token 网格通常是：

```text
H_token = floor(320 / 14) = 22
W_token = floor(320 / 14) = 22
```

因此一张图默认产生的 patch embedding 数量约为：

```text
Q = 1 * 22 * 22 = 484
```

这和 ResNet `layer2` 在 `320 x 320` 下的 `40 x 40 = 1600` 个 patch 不同。
