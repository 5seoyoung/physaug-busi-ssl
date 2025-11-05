#!/usr/bin/env bash
set -euo pipefail

PRE_VAN="outputs/runs/ssl/byol_vanilla/checkpoints/best.pt"
PRE_PHY="outputs/runs/ssl/byol_physaug/checkpoints/best.pt"
OUTROOT="outputs/runs/cv"
SEEDS=(0 1 2)   # fold id와 동일하게 씀
RATIOS=(0.1 1.0)
MODELS=("vanilla" "physaug")

mkdir -p "$OUTROOT"

run_fold () {
  local model=$1 ratio=$2 fold=$3 pre=$4 tag=$5
  local out="$OUTROOT/${tag}_r${ratio}_f${fold}"
  mkdir -p "$out"
  PYTHONUNBUFFERED=1 python -u src/downstream/cls_train.py \
    --paths configs/paths.yaml \
    --pretrained "$pre" \
    --label_ratio "$ratio" \
    --epochs 8 \
    --workers 0 \
    --freeze_backbone \
    --save_pred_csv \
    --cv_json configs/folds/cv3.json \
    --cv_fold "$fold" \
    --out_dir "$out" \
    2>&1 | tee "$out/train_full.log"
}

for model in "${MODELS[@]}"; do
  if [[ "$model" == "vanilla" ]]; then pre="$PRE_VAN"; else pre="$PRE_PHY"; fi
  for ratio in "${RATIOS[@]}"; do
    for fold in "${SEEDS[@]}"; do
      tag="${model}"
      run_fold "$model" "$ratio" "$fold" "$pre" "$tag"
    done
  done
done
