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

## Backbone changes

Backbone switching is now localized to config and `src/patchcore/backbones.py`.
If you want to add DINOv2 later, the right extension point is the backbone/extractor layer, not the CLI scripts.
