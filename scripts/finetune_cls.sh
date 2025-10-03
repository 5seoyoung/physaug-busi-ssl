#!/usr/bin/env bash
set -e
CKPT_PHY="outputs/runs/ssl/byol_physaug/checkpoints/best.pt"
CKPT_VAN="outputs/runs/ssl/byol_vanilla/checkpoints/best.pt"

for SRC in "$CKPT_PHY" "$CKPT_VAN"
do
  python -m src.downstream.cls_train \
    --cfg configs/finetune_cls.yaml \
    --paths configs/paths.yaml \
    --pretrained "$SRC"
done
