# src/downstream/cls_train.py
# ------------------------------------------------------------
# BYOL 사전학습 가중치로 BUSI 3-class 분류 파인튜닝/리니어프로브 스크립트
# - 안전한 evaluate() (eval/no_grad/argmax)
# - 환자그룹 중복/클래스 카운트 sanity 로그
# - 가중치 로딩(유연한 prefix 제거, strict=False) + 로딩 통계
# - 옵션: 가중치 샘플러, CE class weight, 백본 고정, 예측 CSV 저장,
#         퍼뮤테이션 테스트(--permute_train_labels),
#         샘플러 타입만 확인하고 종료(--debug_samplers_only)
# - 외부에서 쓸 수 있게 build_dataloaders() 함수 export
# ------------------------------------------------------------
print("[BOOT] cls_train imported")

import os
import re
import time
import math
import csv
import yaml
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, SequentialSampler, RandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# -------------------------
# 유틸
# -------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_paths_yaml(paths_yaml: str) -> Dict:
    with open(paths_yaml, "r") as f:
        d = yaml.safe_load(f)
    return d


def resolve_busi_root(paths_dict: Dict) -> str:
    """
    paths.yaml 안의 키 몇 개를 순서대로 시도해서 BUSI 루트를 찾는다.
    최종적으로 '.../Dataset_BUSI_with_GT' 폴더를 반환.
    """
    candidates = [
        paths_dict.get("busi_raw"),
        paths_dict.get("BUSI"),
        paths_dict.get("busi"),
        paths_dict.get("data_root"),
    ]
    candidates = [c for c in candidates if c]
    if not candidates:
        raise FileNotFoundError("paths.yaml 에서 BUSI 경로를 찾지 못했습니다 (busi_raw/BUSI/busi/data_root).")

    base = candidates[0]
    # 일반적으로 data/raw/BUSI/Dataset_BUSI_with_GT 구조
    root = os.path.join(base, "Dataset_BUSI_with_GT")
    if not os.path.isdir(root):
        # 혹시 이미 class 폴더 바로 아래일 수도 있음
        if any(os.path.isdir(os.path.join(base, d)) for d in ("benign", "malignant", "normal")):
            root = base
        else:
            raise FileNotFoundError(f"BUSI root not found: {root}")
    return root


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # 검증용 변환은 약하게(Resize + CenterCrop + Normalize)
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    # 학습용: 너무 과하지 않게
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def extract_group_id_from_name(path: str) -> Optional[str]:
    """
    BUSI 파일명에서 환자 ID를 추출. 예) 'benign (123)_123.png' -> '123'
    괄호번호가 없으면 None
    """
    name = os.path.basename(path)
    m = re.search(r"\((\d+)\)", name)
    return m.group(1) if m else None


def stratified_group_split(paths: List[str], labels: List[int], val_ratio: float = 0.2, seed: int = 0):
    """
    환자 그룹 단위로 train/val을 분리.
    클래스 불균형은 크게 건드리지 않고 그룹단위로 80/20 근사.
    """
    rng = random.Random(seed)
    # 그룹 -> (paths, labels) 리스트
    group_to_items = defaultdict(list)
    for p, y in zip(paths, labels):
        gid = extract_group_id_from_name(p)
        group_to_items[gid].append((p, y))

    groups = list(group_to_items.keys())
    rng.shuffle(groups)

    n_total = len(groups)
    n_val = max(1, int(round(n_total * val_ratio)))
    val_groups = set(groups[:n_val])
    tr_paths, tr_labels, va_paths, va_labels = [], [], [], []

    for gid, items in group_to_items.items():
        if gid in val_groups:
            for p, y in items: va_paths.append(p); va_labels.append(y)
        else:
            for p, y in items: tr_paths.append(p); tr_labels.append(y)

    return tr_paths, tr_labels, va_paths, va_labels


def subset_by_label_ratio(paths: List[str], labels: List[int], ratio: float, seed: int = 0):
    """
    라벨 비율(ratio)로 train subset을 샘플링(각 클래스별로 비율 유지하려고 노력).
    """
    if ratio >= 0.999:
        return paths, labels
    rng = random.Random(seed)
    per_class = defaultdict(list)
    for i, (p, y) in enumerate(zip(paths, labels)):
        per_class[y].append((p, y))
    sub_paths, sub_labels = [], []
    for y, items in per_class.items():
        n = max(1, int(round(len(items) * ratio)))
        rng.shuffle(items)
        items = items[:n]
        for p, y in items:
            sub_paths.append(p); sub_labels.append(y)
    return sub_paths, sub_labels


class ListDataset(torch.utils.data.Dataset):
    """
    (경로, 라벨) 리스트를 받아 이미지 로드하는 간단 Dataset.
    torchvision ImageFolder와 동일한 __getitem__ 시그니처(img, label) 유지.
    """
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform
        # samples/paths 호환 필드(로그/CSV용)
        self.samples = [(p, y) for p, y in items]
        self.paths = [p for p, _ in items]
        self.targets = [y for _, y in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, y


def build_image_lists(busi_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    BUSI 루트에서 (경로 리스트, 라벨 리스트, 클래스 이름 리스트) 생성
    """
    cls_names = []
    for d in sorted(os.listdir(busi_root)):
        full = os.path.join(busi_root, d)
        if os.path.isdir(full):
            cls_names.append(d)
    cls_to_idx = {c: i for i, c in enumerate(sorted(cls_names))}

    all_paths, all_labels = [], []
    for c in cls_names:
        cdir = os.path.join(busi_root, c)
        for root, _, files in os.walk(cdir):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    all_paths.append(os.path.join(root, fn))
                    all_labels.append(cls_to_idx[c])

    return all_paths, all_labels, sorted(cls_names)


def make_sampler_and_weights(labels: List[int]) -> Tuple[Optional[WeightedRandomSampler], Optional[torch.Tensor], List[int]]:
    """
    WeightedRandomSampler 및 CrossEntropy 가중치(1/n_c) 계산
    """
    counts = Counter(labels)  # class -> count
    class_counts = [counts.get(c, 0) for c in sorted(set(labels))]
    # CE weight: 1/count
    ce_w = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    # sample weight per example: weight[label]
    per_class_weight = {c: (1.0 / cnt if cnt > 0 else 0.0) for c, cnt in counts.items()}
    sample_weights = [per_class_weight[y] for y in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, torch.tensor(ce_w, dtype=torch.float32), class_counts


# -------------------------
# 모델/로딩
# -------------------------
def build_model(n_classes: int) -> nn.Module:
    # torchvision mobilenet_v3_small
    m = models.mobilenet_v3_small(weights=None)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, n_classes)
    return m


def flexible_load_state_dict(model: nn.Module, ckpt_path: str) -> Tuple[int, int, int]:
    """
    다양한 BYOL 체크포인트 키(prefix)를 대비하여 최대한 맞춰서 로드.
    반환: (loaded, missing, unexpected)
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # prefix 후보들 제거 시도
    prefixes = [
        "encoder.", "backbone.",
        "online_encoder.", "online_network.encoder.",
        "module.encoder.", "model.encoder.",
        "online_backbone.",
    ]

    model_sd = model.state_dict()
    new_sd = {}
    unexpected = 0
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        # classifier 차원 mismatch는 자동으로 skip되도 됨
        if kk in model_sd and model_sd[kk].shape == v.shape:
            new_sd[kk] = v
        else:
            # 혹시 features. 하위만 맞는 경우 일부 매칭 시도 (mobilenet features.*)
            if kk.startswith("features.") and kk in model_sd and model_sd[kk].shape == v.shape:
                new_sd[kk] = v
            else:
                unexpected += 1

    missing = sum(1 for k in model_sd.keys() if k not in new_sd)
    loaded = len(new_sd)
    model.load_state_dict(new_sd, strict=False)
    return loaded, missing, unexpected


# -------------------------
# 평가/로그 저장
# -------------------------
def evaluate(model: nn.Module,
             val_dl: DataLoader,
             device: torch.device,
             class_names: List[str],
             out_dir: str,
             epoch_tag="final",
             save_csv: bool = False) -> Tuple[float, float]:
    was_training = model.training
    model.eval()

    all_true, all_pred, all_paths = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                images, labels = batch.to(device), None

            logits = model(images)
            if logits.ndim != 2 or logits.size(1) < 2:
                raise RuntimeError(f"Evaluate: unexpected logits shape {tuple(logits.shape)}")
            pred = torch.argmax(logits, dim=1)

            all_pred.extend(pred.cpu().tolist())
            if labels is not None:
                all_true.extend(labels.cpu().tolist())

            # 경로 (가능하면)
            paths = getattr(val_dl.dataset, "paths", None)
            if paths is not None:
                # DataLoader가 순차 샘플링이라면 배치 인덱스 매핑 필요하지만,
                # 여기서는 CSV 디버깅 목적이라 빈 값으로 채웁니다.
                all_paths.extend([""] * len(images))
            else:
                all_paths.extend([""] * len(images))

    os.makedirs(out_dir, exist_ok=True)

    if all_true:
        acc = accuracy_score(all_true, all_pred)
        macro_f1 = f1_score(all_true, all_pred, average="macro")
        print(f"[Metrics] epoch={epoch_tag} | macro_f1={macro_f1:.4f}")

        # 혼동행렬/리포트 저장(라벨 명시)
        try:
            rep = classification_report(
                all_true, all_pred,
                target_names=class_names,
                labels=list(range(len(class_names))),
                output_dict=True
            )
        except Exception:
            rep = classification_report(all_true, all_pred, output_dict=True)

        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(class_names))))

        torch.save(
            {"y_true": all_true, "y_pred": all_pred, "acc": acc, "macro_f1": macro_f1, "cm": cm, "report": rep},
            os.path.join(out_dir, f"val_metrics_{epoch_tag}.pt")
        )

        if save_csv:
            csv_path = os.path.join(out_dir, f"val_pred_{epoch_tag}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "y_true", "y_pred"])
                for p, t, pr in zip(all_paths, all_true, all_pred):
                    w.writerow([p, t, pr])

        if was_training:
            model.train()
        return acc, macro_f1

    else:
        if was_training:
            model.train()
        return 0.0, 0.0


# -------------------------
# DataLoader 빌드 (외부 import 용)
# -------------------------
def build_dataloaders(paths_yaml: str,
                      img_size: int = 128,
                      label_ratio: float = 1.0,
                      batch_size: int = 64,
                      use_weighted_sampler: bool = False,
                      workers: int = 0,
                      seed: int = 0):
    set_seed(seed)
    paths_dict = load_paths_yaml(paths_yaml)
    busi_root = resolve_busi_root(paths_dict)

    all_paths, all_labels, class_names = build_image_lists(busi_root)

    # 그룹 분리 (80/20)
    tr_p, tr_y, va_p, va_y = stratified_group_split(all_paths, all_labels, val_ratio=0.2, seed=seed)

    # label_ratio 적용
    tr_p, tr_y = subset_by_label_ratio(tr_p, tr_y, ratio=label_ratio, seed=seed)

    train_tf, val_tf = build_transforms(img_size)
    train_ds = ListDataset(list(zip(tr_p, tr_y)), transform=train_tf)
    val_ds = ListDataset(list(zip(va_p, va_y)), transform=val_tf)

    # 샘플러
    if use_weighted_sampler:
        sampler, ce_w, class_counts = make_sampler_and_weights(train_ds.targets)
        train_sampler = sampler
        train_shuffle = False
    else:
        train_sampler = None
        ce_w = None
        class_counts = [Counter(train_ds.targets).get(i, 0) for i in sorted(set(train_ds.targets))]
        train_shuffle = True

    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=train_shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=workers, pin_memory=True, drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, sampler=SequentialSampler(val_ds),
        num_workers=workers, pin_memory=True, drop_last=False,
    )
    return train_dl, val_dl, class_names, ce_w, class_counts


# -------------------------
# 학습 루프
# -------------------------
def train_one_epoch(model, dl, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n = 0
    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs
    return running_loss / max(1, n)


# -------------------------
# 메인
# -------------------------
def main(args):
    set_seed(0)

    # Data
    train_dl, val_dl, class_names, ce_weight_from_sampler, class_counts = build_dataloaders(
        args.paths, img_size=args.img_size, label_ratio=args.label_ratio,
        batch_size=args.batch_size, use_weighted_sampler=args.use_weighted_sampler,
        workers=args.workers, seed=0
    )

    # Sanity: 클래스 카운트 & 그룹 중복 체크
    def _count_by_class(ds):
        return Counter(int(y) for y in ds.targets)

    tr_counts = _count_by_class(train_dl.dataset)
    va_counts = _count_by_class(val_dl.dataset)
    print("[COUNT] train:", dict(tr_counts))
    print("[COUNT] val  :", dict(va_counts))

    def _group_overlap(ds_a, ds_b):
        def _ids(ds):
            ids = []
            for p in getattr(ds, "paths", []):
                gid = extract_group_id_from_name(p)
                if gid: ids.append(gid)
            return set(ids)
        sa, sb = _ids(train_dl.dataset), _ids(val_dl.dataset)
        if sa and sb:
            return len(sa & sb)
        return -1

    go = _group_overlap(train_dl.dataset, val_dl.dataset)
    if go >= 0:
        print(f"[GROUP] unique_train={len(train_dl.dataset)} | unique_val={len(val_dl.dataset)} | group_overlap={go}")
    else:
        print("[GROUP] skip (no group id available)")

    print(f"[Run summary] train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | batch={args.batch_size} | img={args.img_size}")

    # Sampler 정보 출력(디버그 플래그)
    tr_sampler_name = type(train_dl.sampler).__name__
    va_sampler_name = type(val_dl.sampler).__name__
    va_shuffle = getattr(val_dl, 'shuffle', False)
    if args.debug_samplers_only:
        print(f"[Sampler] train={tr_sampler_name} | val={va_sampler_name} | val_shuffle={va_shuffle}")
        return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Run summary] classes={class_names} | device={device.type}")

    # Model
    n_classes = len(class_names)
    model = build_model(n_classes)
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad = False
        print("[INFO] backbone frozen (linear probe)")

    # Load pretraineds
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"[INFO] load pretrained (BYOL) from: {args.pretrained}")
        loaded, missing, unexpected = flexible_load_state_dict(model, args.pretrained)
        print(f"[INFO] loaded={loaded}, missing={missing}, unexpected={unexpected}")
    else:
        print("[WARN] pretrained checkpoint not found or not given; training from scratch.")

    model = model.to(device)

    # Loss
    ce_weight = None
    # sampler를 쓰면 ce_weight_from_sampler가 계산되어 옴
    # --use_ce_weight 옵션이 켜져 있으면 CE에도 적용
    if args.use_ce_weight:
        if ce_weight_from_sampler is None:
            # 샘플러를 안 쓰는 경우에도 계산
            counts = Counter(train_dl.dataset.targets)
            class_counts2 = [counts.get(i, 0) for i in sorted(set(train_dl.dataset.targets))]
            ce_weight = torch.tensor(
                [1.0 / c if c > 0 else 0.0 for c in class_counts2],
                dtype=torch.float32, device=device
            )
            print(f"[INFO] class_counts={class_counts2} | ce_weight={ce_weight.tolist()}")
        else:
            ce_weight = ce_weight_from_sampler.to(device)
            print(f"[INFO] class_counts={class_counts} | ce_weight={ce_weight.tolist()}")

    # Sampler 로그(학습 모드에서도 한 번 출력)
    print(f"[Sampler] train={tr_sampler_name} | val={va_sampler_name} | val_shuffle={va_shuffle}")

    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    # Optimizer: backbone freeze 여부에 따라 파라미터 선택
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    os.makedirs(args.out_dir, exist_ok=True)

    best_acc = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # 퍼뮤테이션 테스트: 학습 라벨 섞기 (train_ds.targets를 덮어씌움)
        if args.permute_train_labels and epoch == 1:
            rng = random.Random(0)
            tgt = train_dl.dataset.targets
            idxs = list(range(len(tgt)))
            rng.shuffle(idxs)
            shuffled = [tgt[i] for i in idxs]
            train_dl.dataset.targets = shuffled
            # items도 갱신
            for i in range(len(train_dl.dataset.items)):
                p, _ = train_dl.dataset.items[i]
                train_dl.dataset.items[i] = (p, shuffled[i])
            print("[PERMUTE] Shuffled training labels for permutation test")

        train_loss = train_one_epoch(model, train_dl, device, criterion, optimizer)
        val_acc, macro_f1 = evaluate(
            model, val_dl, device, class_names,
            out_dir=args.out_dir, epoch_tag=epoch, save_csv=args.save_pred_csv
        )
        print(f"[Epoch {epoch}/{args.epochs}] loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": class_names},
                       os.path.join(args.out_dir, "best_cls.pt"))

    dt = time.time() - t0
    print(f"[OK] Finished. Best cls ckpt at: {os.path.join(args.out_dir, 'best_cls.pt')} | best_acc={best_acc:.4f} | {dt:.1f}s")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    print("[BOOT] __main__ entry")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="configs/paths.yaml")
    ap.add_argument("--pretrained", required=True, help="BYOL checkpoint path")
    ap.add_argument("--label_ratio", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=128)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--use_ce_weight", action="store_true")
    ap.add_argument("--use_weighted_sampler", action="store_true")

    # 디버깅/검증용
    ap.add_argument("--save_pred_csv", action="store_true",
                    help="Save per-sample predictions on val to CSV each epoch")
    ap.add_argument("--permute_train_labels", action="store_true",
                    help="Permutation test: shuffle training labels to detect leakage")
    ap.add_argument("--debug_samplers_only", action="store_true",
                    help="Just print sampler types and exit")

    args = ap.parse_args()
    main(args)
