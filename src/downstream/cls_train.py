# -*- coding: utf-8 -*-
"""
Supervised classifier fine-tuning on BUSI with BYOL-pretrained encoder.

- Prints:
  [Run summary] train=... | val=... | batch=... | img=...
  [Run summary] classes=[...] | device=...
  [INFO] load pretrained (BYOL) from: ...
  [INFO] loaded=..., missing=..., unexpected=...
  [Epoch i/E] loss=... | val_acc=...
  [Metrics] epoch=i | macro_f1=...
  [OK] Finished. Best cls ckpt at: ... | best_acc=... | X.Ys

- Options:
  --freeze_backbone           # linear probe
  --use_ce_weight             # class-weighted CE
  --use_weighted_sampler      # WeightedRandomSampler (shuffle 대체)
"""

print('[BOOT] cls_train imported')

import os, time, json, math, random
from pathlib import Path
from collections import Counter

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

try:
    import timm
except Exception as e:
    timm = None

# =============== METRICS UTILS ====================
def save_metrics(y_true, y_pred, class_names, out_dir, tag):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump({"macro_f1": macro_f1, "report": rep, "cm": cm.tolist()}, f, indent=2, ensure_ascii=False)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.title(f"Confusion Matrix ({tag})")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out / f"cm_{tag}.png")
    plt.close(fig)

    return macro_f1
# ===================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_paths_yaml(paths_yaml: str) -> Path:
    """
    기대 구조 예시(유연 파싱):
    busi_raw: data/raw/BUSI
    혹은:
    datasets:
      busi_raw: data/raw/BUSI
    """
    with open(paths_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict):
        if "busi_raw" in cfg:
            return Path(cfg["busi_raw"])
        if "datasets" in cfg and isinstance(cfg["datasets"], dict) and "busi_raw" in cfg["datasets"]:
            return Path(cfg["datasets"]["busi_raw"])
    # fallback
    return Path("data/raw/BUSI")


def _is_busi_valid_file(p: str) -> bool:
    """BUSI 폴더에서 *_mask 파일 제외."""
    name = Path(p).name.lower()
    if "mask" in name:
        return False
    # torchvision이 확장자 필터를 하므로 여기선 mask만 걸러도 충분
    return True


def build_datasets(paths_yaml: str, img_size: int, label_ratio: float):
    root_raw = _read_paths_yaml(paths_yaml)
    busi_root = root_raw / "Dataset_BUSI_with_GT"  # benign/malignant/normal 하위에 원본/마스크 혼재
    if not busi_root.exists():
        # 혹시 상위에 바로 3개 클래스 폴더가 있는 경우 fallback
        if (root_raw / "benign").exists():
            busi_root = root_raw
        else:
            raise FileNotFoundError(f"BUSI root not found: {busi_root}")

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # *** 핵심: *_mask 파일 제외 ***
    full_trainable = datasets.ImageFolder(
        busi_root, transform=train_tf, is_valid_file=_is_busi_valid_file
    )
    classes = full_trainable.classes  # ['benign', 'malignant', 'normal'] (가정)

    n_total = len(full_trainable)
    val_size = int(round(n_total * 0.2))
    train_size = n_total - val_size

    # 재현성 split
    g = torch.Generator().manual_seed(123)
    train_ds_idx, val_ds_idx = random_split(range(n_total), [train_size, val_size], generator=g)

    # 인덱스 서브셋 생성
    train_ds = Subset(full_trainable, list(train_ds_idx.indices))
    # val은 같은 필터 + val transform
    full_valable = datasets.ImageFolder(
        busi_root, transform=val_tf, is_valid_file=_is_busi_valid_file
    )
    val_ds = Subset(full_valable, list(val_ds_idx.indices))

    # label_ratio 적용 (train만)
    if label_ratio < 1.0:
        # 원본 라벨 벡터를 이용해 클래스별 비율 샘플링
        base = full_trainable
        y_train = [base.targets[i] for i in train_ds.indices]
        per_class = Counter(y_train)
        want = {c: max(1, int(math.floor(per_class[c] * label_ratio))) for c in per_class}
        taken = {c: 0 for c in per_class}
        sub_idx = []
        for idx in train_ds.indices:
            c = base.targets[idx]
            if taken[c] < want[c]:
                sub_idx.append(idx)
                taken[c] += 1
        train_ds = Subset(full_trainable, sub_idx)

    return train_ds, val_ds, classes


def build_model(n_classes: int):
    if timm is None:
        raise RuntimeError("timm is required. Install via `pip install timm`.")
    # BYOL에서 사용한 encoder와 동일 가정: mobilenetv3_small_100
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=n_classes)
    return model


def _maybe_strip_prefix(sd, prefix):
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _find_state_dict(ckpt):
    """
    다양한 저장 포맷 처리:
    - {'model': state_dict}
    - {'state_dict': state_dict}
    - 바로 state_dict
    - {'online_encoder': sd}, {'encoder': sd}, {'backbone': sd} 등
    """
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "online_encoder", "encoder", "backbone"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        for v in ckpt.values():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                return v
    return ckpt


def load_pretrained_into_classifer_model(model, ckpt_path: str):
    if not ckpt_path or not Path(ckpt_path).exists():
        return (0, 0, 0)

    print(f"[INFO] load pretrained (BYOL) from: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _find_state_dict(raw)

    for p in ["module.", "encoder.", "backbone.", "online_encoder."]:
        if any(k.startswith(p) for k in sd.keys()):
            sd = _maybe_strip_prefix(sd, p)

    head_like = tuple(n for n, _ in model.named_parameters() if any(k in n for k in ("classifier", "fc", "head")))
    head_roots = set(n.split(".")[0] for n in head_like)

    pruned = {}
    for k, v in sd.items():
        if any(k.startswith(hr) for hr in head_roots) and (("weight" in k) or ("bias" in k)):
            continue
        pruned[k] = v

    missing, unexpected = model.load_state_dict(pruned, strict=False)
    loaded = len(pruned) - len(unexpected)
    print(f"[INFO] loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}")
    return (loaded, len(missing), len(unexpected))


def evaluate(model, dl, device, classes, out_dir, epoch_tag=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    all_true, all_pred = [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, targets in dl:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = ce(logits, targets)
            loss_sum += loss.item()
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            all_true.extend(targets.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    val_acc = correct / max(1, total)
    if epoch_tag is not None:
        macro_f1 = save_metrics(all_true, all_pred, classes, out_dir, f"epoch{epoch_tag:03d}")
        print(f"[Metrics] epoch={epoch_tag} | macro_f1={macro_f1:.4f}")
    return loss_sum / max(1, total), val_acc


def main(args):
    set_seed(42)
    device = get_device()

    # datasets
    train_ds, val_ds, classes = build_datasets(args.paths, args.img_size, args.label_ratio)

    # class imbalance (optional)
    ce_weight_tensor = None
    sampler = None
    if args.use_ce_weight:
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        if isinstance(train_ds, Subset):
            full_targets = [base_ds.targets[i] for i in train_ds.indices]
        else:
            full_targets = base_ds.targets
        counts = Counter(full_targets)
        class_counts = [counts.get(i, 0) for i in range(len(classes))]
        class_weights = [1.0 / (c + 1e-6) for c in class_counts]
        ce_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"[INFO] class_counts={class_counts} | ce_weight={class_weights}")

    if args.use_weighted_sampler:
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        if isinstance(train_ds, Subset):
            labels = [base_ds.targets[i] for i in train_ds.indices]
        else:
            labels = base_ds.targets

        if ce_weight_tensor is None:
            counts = Counter(labels)
            class_counts = [counts.get(i, 0) for i in range(len(classes))]
            class_weights = [1.0 / (c + 1e-6) for c in class_counts]
            ce_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            print(f"[INFO] (sampler) class_counts={class_counts} | ce_weight={class_weights}")
        sample_weights = [float(ce_weight_tensor.cpu().numpy()[y]) for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # loaders
    pin_mem = (device.type == "cuda")  # MPS에서는 경고 나와서 CUDA만 pin_memory
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=pin_mem
    )
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          pin_memory=pin_mem)

    print(f"[Run summary] train={len(train_ds)} | val={len(val_ds)} | batch={args.batch_size} | img={args.img_size}")
    print(f"[Run summary] classes={classes} | device={device.type}")

    # model
    n_classes = len(classes)
    if n_classes < 2:
        raise RuntimeError(f"클래스 수가 이상합니다: {n_classes}")
    model = build_model(n_classes).to(device)

    # linear probe (optional)
    if args.freeze_backbone:
        head_keys = ("classifier", "fc", "head")
        for n, p in model.named_parameters():
            if not any(k in n for k in head_keys):
                p.requires_grad = False
        print("[INFO] backbone frozen (linear probe)")

    # load pretrained BYOL encoder (strict=False)
    load_pretrained_into_classifer_model(model, args.pretrained)

    # optim/sched
    criterion = nn.CrossEntropyLoss(weight=ce_weight_tensor)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = out_dir / "best_cls.pt"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_run = 0.0
        seen = 0
        for images, targets in train_dl:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            loss_run += loss.item() * bs
            seen += bs

        scheduler.step()

        # validation
        val_loss, val_acc = evaluate(model, val_dl, device, classes, out_dir, epoch_tag=epoch)
        train_loss = loss_run / max(1, seen)
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        # keep best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    sec = time.time() - t0
    print(f"[OK] Finished. Best cls ckpt at: {best_path} | best_acc={best_acc:.4f} | {sec:.1f}s")


if __name__ == "__main__":
    import argparse
    print("[BOOT] __main__ entry")

    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--label_ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=128)

    # optional flags
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone params (linear probe).")
    ap.add_argument("--use_ce_weight", action="store_true",
                    help="Use class-weighted cross-entropy.")
    ap.add_argument("--use_weighted_sampler", action="store_true",
                    help="Use WeightedRandomSampler for training loader (replaces shuffle).")

    args = ap.parse_args()
    main(args)
