import os, random, json, time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

try:
    import timm
except Exception as e:
    raise RuntimeError("timm이 필요합니다. 가상환경에서 `pip install timm` 해주세요.") from e

print("[BOOT] cls_train imported")

def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class BUSIDataset(ImageFolder):
    def __init__(self, root, transform=None):
        # 마스크 파일 제외
        super().__init__(root, transform=transform, is_valid_file=lambda p: "mask" not in os.path.basename(p).lower())

def _load_paths(paths_yaml):
    with open(paths_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["busi_raw"])
    if not root.exists():
        raise FileNotFoundError(f"BUSI 경로가 없습니다: {root.resolve()}")
    return root

def _split_indices(dataset, train_ratio=0.8, seed=42):
    random.seed(seed)
    cls_to_idxs = {}
    for idx, (_, y) in enumerate(dataset.samples):
        cls_to_idxs.setdefault(y, []).append(idx)
    train_idx, val_idx = [], []
    for _, idxs in cls_to_idxs.items():
        random.shuffle(idxs)
        n_tr = int(len(idxs) * train_ratio)
        train_idx += idxs[:n_tr]
        val_idx   += idxs[n_tr:]
    return train_idx, val_idx

def _subsample_per_class(dataset, indices, frac=1.0, seed=42):
    if frac >= 0.999:
        return indices
    random.seed(seed)
    cls_to_idxs = {}
    for idx in indices:
        y = dataset.samples[idx][1]
        cls_to_idxs.setdefault(y, []).append(idx)
    kept = []
    for _, idxs in cls_to_idxs.items():
        random.shuffle(idxs)
        n_keep = max(1, int(len(idxs) * frac))
        kept += idxs[:n_keep]
    return kept

def _build_transforms(img_size):
    train_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    val_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    return train_tf, val_tf

def _build_model(num_classes=3):
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=num_classes, in_chans=3)
    return model

def _try_load_ssl_backbone(model, ckpt_path):
    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"[WARN] pretrained ckpt 미존재: {ckpt_path}")
        return
    print(f"[INFO] load pretrained (BYOL) from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))

    def strip_prefix(k):
        for p in ["online.encoder.model.", "online.encoder.", "online.", "target.encoder.", "target.", "encoder.", "backbone.", "model."]:
            if k.startswith(p):
                k = k[len(p):]
        return k

    model_sd = model.state_dict()
    new_sd = {}
    loaded = 0
    for k, v in sd.items():
        kk = strip_prefix(k)
        if kk in model_sd and not kk.startswith("classifier"):
            if v.shape == model_sd[kk].shape:
                new_sd[kk] = v
                loaded += 1
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[INFO] loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}")
    if loaded == 0:
        print("[WARN] 일치하는 파라미터가 없어 랜덤 초기화로 진행합니다.")

def _train_one_epoch(model, dl, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in dl:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(dl.dataset)

@torch.no_grad()
def _eval(model, dl, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def main(args):
    device = _device()
    busi_root = _load_paths(args.paths)

    train_tf, val_tf = _build_transforms(args.img_size)

    # 전체 데이터셋 로드(마스크 제외)
    full = BUSIDataset(busi_root, transform=None)
    n_classes = len(full.classes)
    if n_classes < 2:
        raise RuntimeError(f"클래스 수가 이상합니다: {n_classes}")

    # split
    tr_idx, va_idx = _split_indices(full, train_ratio=0.8, seed=42)
    tr_idx = _subsample_per_class(full, tr_idx, frac=float(args.label_ratio), seed=42)

    train_ds = BUSIDataset(busi_root, transform=train_tf); train_ds.samples = [full.samples[i] for i in tr_idx]
    val_ds   = BUSIDataset(busi_root, transform=val_tf);   val_ds.samples   = [full.samples[i] for i in va_idx]

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    print(f"[Run summary] train={len(train_ds)} | val={len(val_ds)} | batch={args.batch_size} | img={args.img_size}")
    print(f"[Run summary] classes={full.classes} | device={device}")

    # 모델
    model = _build_model(num_classes=n_classes)
    _try_load_ssl_backbone(model, args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_path = -1.0, out_dir / "best_cls.pt"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss = _train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_acc = _eval(model, val_dl, device)
        print(f"[Epoch {ep}/{args.epochs}] loss={tr_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best_acc, "epoch": ep}, best_path)
    # 기록
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_val_acc": best_acc, "epochs": args.epochs}, f, indent=2)
    print(f"[OK] Finished. Best cls ckpt at: {best_path} | best_acc={best_acc:.4f} | {time.time()-t0:.1f}s")

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
    args = ap.parse_args()
    main(args)
