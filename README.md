# Minimal PatchCore

This repository is now a small PatchCore implementation focused on three tasks:

- train a memory bank from normal images
- run inference on a single image or a folder
- export the model part to ONNX

The project no longer depends on MVTec-specific directory layouts or the previous script-heavy workflow.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Train

Training input is just a folder of normal images.

```bash
python train.py \
  --train-dir dataset/train/good \
  --output-dir artifacts/wr50_bank \
  --backbone wide_resnet50_2 \
  --image-size 320
```

Useful options:

- `--layers layer2 layer3`
- `--sampler approx_greedy_coreset --sample-ratio 0.1`
- `--preprocess gaussian_blur`
- `--cpu`

Sampler options:

- `identity`: keep all patch embeddings
- `random`: random downsampling
- `greedy_coreset`: best quality, usually slowest at training time
- `approx_greedy_coreset`: recommended balance for large memory banks

## Infer

Inference input can be one image or a folder.

```bash
python infer.py \
  --model-dir artifacts/wr50_bank \
  --input dataset/test \
  --output-dir infer_out \
  --threshold 1.0
```

Output:

- `infer_out/results.csv`
- `infer_out/above_threshold/`
- `infer_out/below_threshold/`

## Export ONNX

Only the model side is exported. The FAISS index and distance search stay outside ONNX.

```bash
python convert_to_onnx.py \
  --model-dir artifacts/wr50_bank \
  --output-dir onnx_out
```

Output:

- `onnx_out/patchcore_embedder.onnx`
- `onnx_out/onnx_metadata.json`

## Structure

Core code lives in `src/patchcore/`:

- `preprocess.py`: image loading and resize/pad/normalize
- `backbones.py`: backbone registry
- `embedding.py`: patchify and embedding aggregation
- `common.py`: FAISS index and mask upsampling
- `sampler.py`: memory bank sampling strategies
- `patchcore.py`: train/infer orchestration
- `onnx.py`: ONNX export wrapper

## Feature extraction and scoring flow

This section follows the current code path in `src/patchcore/patchcore.py`,
`src/patchcore/embedding.py`, and `src/patchcore/common.py`.

Notation:

- `B`: batch size. This implementation usually processes one image at a time, so `B=1`.
- `S`: configured square image size, from `--image-size`.
- `L`: number of selected backbone layers, normally `2` for `layer2 layer3`.
- `C_l, H_l, W_l`: channel, height, and width of feature layer `l`.
- `P`: patch size, from `--patch-size`, default `3`.
- `D_pre`: per-layer pooled patch dimension, from `--pretrain-embed-dim`, default `1024`.
- `D`: final embedding dimension, from `--target-embed-dim`, default `1024`.
- `Q`: number of query patch embeddings from one image.
- `M`: number of memory bank embeddings after optional sampling.
- `K`: nearest neighbors, from `--num-neighbors`, default `1`.

### 1. Image preprocessing

`ImagePreprocessor` loads an image, converts it to RGB, optionally applies a simple
filter, resizes it with aspect ratio preserved, pads it to a square, converts it to a
tensor, and applies ImageNet normalization.

Shape change:

```text
original image                 [H0, W0, 3]
resize + pad to square          [S, S, 3]
ToTensor + Normalize + batch    [B, 3, S, S]
```

For the common command `--image-size 320`, the model input is:

```text
[1, 3, 320, 320]
```

### 2. Backbone feature maps

`FeatureExtractor` registers forward hooks on the configured layers, runs the
torchvision backbone, and returns the captured feature maps in layer order.

For the default `wide_resnet50_2` with `layer2 layer3`, the typical spatial strides
are `8` and `16` relative to the input image:

```text
input image tensor    [B, 3, S, S]
layer2 feature map    [B, C2, S/8,  S/8]
layer3 feature map    [B, C3, S/16, S/16]
```

With `S=320`, this is typically:

```text
layer2    [1, 512, 40, 40]
layer3    [1, 1024, 20, 20]
```

The exact channel counts depend on the selected backbone and layer names.

### 3. Patchify each feature map

`PatchMaker.patchify()` applies `torch.nn.Unfold` to every selected feature map.
With the default `patch_size=3`, `patch_stride=1`, and symmetric padding, the patch
grid keeps the same height and width as the feature map.

For one feature map:

```text
feature map                         [B, C_l, H_l, W_l]
after Unfold                        [B, C_l * P * P, H_l * W_l]
reshape + permute                   [B, H_l * W_l, C_l, P, P]
patch_shape                         (H_l, W_l)
```

Example for `S=320`, `layer2`, `P=3`:

```text
[1, 512, 40, 40] -> [1, 1600, 512, 3, 3]
patch_shape = (40, 40)
```

### 4. Align patch grids across layers

PatchCore combines patches from multiple feature layers. Since deeper layers have
smaller spatial grids, `_align_patch_grid()` upsamples every layer's patch grid to
the first selected layer's grid.

With default layer order `layer2 layer3`, the reference grid is `layer2`:

```text
layer2 patches    [B, 40*40, 512, 3, 3]     already on (40, 40)
layer3 patches    [B, 20*20, 1024, 3, 3]    aligned to (40, 40)
aligned layer3    [B, 40*40, 1024, 3, 3]
```

After this step, each layer has the same number of patch locations:

```text
Q = B * H_ref * W_ref
```

For one `320 x 320` image with `layer2` as reference:

```text
Q = 1 * 40 * 40 = 1600
```

### 5. Pool each layer patch to `D_pre`

Each aligned patch tensor is flattened across channel and local patch dimensions,
then adaptively average-pooled to `D_pre`.

For each layer:

```text
aligned patches                  [B, H_ref*W_ref, C_l, P, P]
flatten batch/locations          [Q, C_l, P, P]
flatten patch content            [Q, 1, C_l * P * P]
adaptive_avg_pool1d to D_pre     [Q, D_pre]
```

Using defaults, every selected layer contributes one `[Q, 1024]` matrix, regardless
of its original channel count.

### 6. Merge layers into final patch embeddings

`PatchcoreEmbedder.forward()` stacks the per-layer pooled vectors, flattens the
layer dimension, and adaptively average-pools again to `D`.

```text
pooled per layer list         L tensors of [Q, D_pre]
torch.stack(dim=1)            [Q, L, D_pre]
reshape                       [Q, 1, L * D_pre]
adaptive_avg_pool1d to D      [Q, D]
```

With defaults `L=2`, `D_pre=1024`, `D=1024`:

```text
layer2 pooled                 [1600, 1024]
layer3 pooled                 [1600, 1024]
stack                         [1600, 2, 1024]
flatten                       [1600, 1, 2048]
final embeddings              [1600, 1024]
patch_shape                   (40, 40)
```

These final patch embeddings are what training stores and inference searches.

### 7. Training: build the memory bank

`PatchCoreModel.fit()` embeds every normal training image with the flow above,
concatenates all patch embeddings, optionally samples them, and builds a FAISS
`IndexFlatL2`.

For `N` training images:

```text
one image embeddings                 [Q, D]
all training embeddings              [N * Q, D]
after sampler                        [M, D]
FAISS memory bank                    M vectors, each D-dimensional
```

If `N=100`, `S=320`, default layers, and no sampling:

```text
Q = 1600
all training embeddings = [160000, 1024]
```

With `--sampler approx_greedy_coreset --sample-ratio 0.1`, the memory bank is about:

```text
M = 160000 * 0.1 = 16000
memory bank = [16000, 1024]
```

The saved files are:

```text
model_config.json     config + memory bank stats
memory_bank.faiss     FAISS L2 search index
```

### 8. Inference: nearest-neighbor patch scores

Inference embeds a test image into query patch embeddings and searches the FAISS
memory bank.

```text
test image tensor              [1, 3, S, S]
query embeddings               [Q, D]
memory bank                    [M, D]
FAISS search result distances  [Q, K]
FAISS search result indices    [Q, K]
```

`IndexFlatL2` returns squared L2 distances. The current scoring code averages the
`K` distances for each query patch:

```text
patch_scores = mean(distances, axis=1)
patch_scores shape = [Q]
```

With default `K=1`, this is just the nearest memory-bank distance per patch.

### 9. Image score and heatmap

The image-level anomaly score is the maximum patch score:

```text
image_score = max(patch_scores)
```

Then patch scores are reshaped back to the reference feature grid and upsampled to
image size:

```text
patch_scores                  [Q]
patch_map                     [1, H_ref, W_ref]
bilinear upsample             [1, S, S]
gaussian smoothing            [S, S]
```

For `S=320`, default layers:

```text
patch_scores                  [1600]
patch_map                     [1, 40, 40]
mask                          [320, 320]
```

`infer.py` overlays this mask on the processed image and writes the scalar
`image_score` to `results.csv`. The threshold comparison is external to PatchCore
itself:

```text
image_score >= threshold -> ANOMALY
image_score < threshold  -> OK
```

### End-to-end default example

For one `320 x 320` image, `wide_resnet50_2`, `layer2 layer3`, `patch_size=3`,
`D_pre=1024`, and `D=1024`:

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
image_score            scalar max over 1600 patch scores
```

## Backbone changes

Backbone switching is now localized to config and `src/patchcore/backbones.py`.
If you want to add DINOv2 later, the right extension point is the backbone/extractor layer, not the CLI scripts.
