#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TOKENIZERS_PARALLELISM=false \
python main.py \
  --supervision growsp_pseudo \
  --growsp_data_dir /home/magic/magic/cm/codex/GrowSP/data/S3DIS/input \
  --growsp_pseudo_label_dir /home/magic/magic/cm/codex/GrowSP/ckpt/S3DIS/pseudo_t1260 \
  --growsp_backbone_ckpt /home/magic/magic/cm/codex/GrowSP/ckpt/S3DIS/ckpts/model_1270_checkpoint.pth
