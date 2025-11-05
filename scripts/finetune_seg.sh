#!/usr/bin/env bash
set -e
CKPT_PHY="outputs/runs/ssl/byol_physaug/checkpoints/best.pt"
CKPT_VAN="outputs/runs/ssl/byol_vanilla/checkpoints/best.pt"

for SRC in "$CKPT_PHY" "$CKPT_VAN"
do
  python -m src.downstream.seg_train \
    --cfg configs/finetune_seg.yaml \
    --paths configs/paths.yaml \
    --pretrained "$SRC"
done
