# Unsupervised GrowSP Interactive4D

This mode trains Interactive4D without S3DIS ground-truth labels in the training loop.

Training supervision comes from frozen GrowSP outputs:

- prepared S3DIS point clouds: `/home/magic/magic/cm/codex/GrowSP/data/S3DIS/input`
- GrowSP pseudo labels: `/home/magic/magic/cm/codex/GrowSP/ckpt/S3DIS/pseudo_t1260`
- optional GrowSP backbone initialization: `/home/magic/magic/cm/codex/GrowSP/ckpt/S3DIS/ckpts/model_1270_checkpoint.pth`

The dataset rebuilds GrowSP's 0.05m quantized point set, aligns each `.npy` pseudo label file to that point set, relabels valid pseudo clusters as interactive objects, and uses those pseudo objects for click/line/box/text interaction training.

S3DIS ground truth remains available only for validation metrics.

Run:

```bash
cd /home/magic/magic/cm/codex/Interactive4D
conda activate cm_inter4d
CUDA_VISIBLE_DEVICES=0 bash scripts/train_unsupervised_growsp_s3dis.sh
```

Debug smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python main.py \
  --debug \
  --supervision growsp_pseudo \
  --growsp_backbone_ckpt /home/magic/magic/cm/codex/GrowSP/ckpt/S3DIS/ckpts/model_1270_checkpoint.pth
```
