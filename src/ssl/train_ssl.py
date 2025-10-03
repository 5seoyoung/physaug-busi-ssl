# -*- coding: utf-8 -*-
import argparse, yaml, math, os, time, torch, random, numpy as np, platform
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..physaug.physaug import PhysAug
from ..data.busi import BUSIUnlabeledSSL
from .byol_lite import OnlineNet, TargetNet, update_momentum, byol_loss

def set_seed(s):
    import torch, random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def device_auto():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ---- 새로 추가: picklable top-level transform ----
class ViewTransform:
    def __init__(self, use_physaug: bool, phys_cfg: dict, hflip: bool = True):
        self.hflip = hflip
        self.aug = PhysAug(phys_cfg) if use_physaug else None

    def __call__(self, im01, size: int):
        import cv2, numpy as np, random
        if self.aug is not None:
            im = self.aug(im01.copy(), resize=size)
        else:
            im = cv2.resize(im01, (size, size), interpolation=cv2.INTER_AREA)
            if random.random() < 0.5 and self.hflip:
                im = im[:, ::-1].copy()
            if random.random() < 0.3:
                g = random.uniform(0.9, 1.1)
                im = np.clip(im * g, 0, 1)
        return im
# --------------------------------------------------

def main(args):
    cfg = yaml.safe_load(open(args.cfg, "r"))
    paths = yaml.safe_load(open(args.paths, "r"))
    phys  = yaml.safe_load(open(args.physaug, "r")) if args.physaug != "none" else {}

    out_dir = Path(cfg.get("out_dir", "outputs/runs/ssl/byol_physaug"))
    out_ckpt = out_dir / "checkpoints"
    out_ckpt.mkdir(parents=True, exist_ok=True)

    seeds = cfg.get("seeds", [0])
    dev = device_auto()

    # macOS 기본 0, 그 외 2 (인자로 덮어쓰기 가능)
    default_workers = 0 if platform.system() == "Darwin" else 2
    workers = args.workers if args.workers is not None else default_workers

    for si, seed in enumerate(seeds):
        set_seed(int(seed))
        use_phys = bool(cfg.get("augment", {}).get("use_physaug", True)) and (args.physaug != "none")
        view_fn = ViewTransform(use_physaug=use_phys, phys_cfg=phys, hflip=True)

        # data
        data_root = paths["busi_raw"]
        ds = BUSIUnlabeledSSL(data_root, view_fn=view_fn, size=int(cfg.get("img_size", 128)))
        dl = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 128)),
            shuffle=True,
            num_workers=int(workers),
            pin_memory=(dev.type == "cuda"),
            drop_last=True,
            persistent_workers=False if int(workers) == 0 else True,
        )

        # model
        enc_name = cfg.get("encoder", "mobilenetv3_small_100")
        proj_dim = cfg.get("byol", {}).get("proj_dim", 256)
        proj_hid = cfg.get("byol", {}).get("proj_hidden", 4096)
        online = OnlineNet(encoder_name=enc_name, proj_dim=proj_dim, proj_hidden=proj_hid, pretrained=True).to(dev)
        target = TargetNet(encoder_name=enc_name, proj_dim=proj_dim, proj_hidden=proj_hid, pretrained=True).to(dev)
        update_momentum(online, target, m=0.0)  # hard copy

        # optim
        opt = AdamW(
            online.parameters(),
            lr=cfg.get("optimizer", {}).get("lr", 1e-3),
            weight_decay=cfg.get("optimizer", {}).get("weight_decay", 1e-4),
        )
        epochs = int(cfg.get("epochs", 200))
        scheduler = CosineAnnealingLR(opt, T_max=epochs)

        ema_start = float(cfg.get("byol", {}).get("ema_start", 0.99))
        ema_end   = float(cfg.get("byol", {}).get("ema_end", 0.996))

        best = 1e9
        for ep in range(1, epochs + 1):
            online.train(); target.eval()
            t0 = time.time(); run_loss, nstep = 0.0, 0
            # momentum schedule (cosine from start->end)
            m = ema_end - (ema_end - ema_start) * (math.cos(math.pi * (ep - 1) / max(1, (epochs - 1))) + 1) / 2.0

            for (v1, v2) in dl:
                v1 = v1.to(dev, non_blocking=True)
                v2 = v2.to(dev, non_blocking=True)
                # forward
                p1, q1 = online(v1)
                p2, q2 = online(v2)
                with torch.no_grad():
                    z1 = target(v1)
                    z2 = target(v2)
                loss = byol_loss(q1, z2.detach()) + byol_loss(q2, z1.detach())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                update_momentum(online, target, m)

                run_loss += loss.item(); nstep += 1
                if args.smoke and nstep >= 20:
                    break

            scheduler.step()
            ep_loss = run_loss / max(1, nstep)
            dt = time.time() - t0
            print(f"[seed {seed}] Epoch {ep}/{epochs} | m={m:.5f} | loss={ep_loss:.4f} | {dt:.1f}s")

            if ep_loss < best:
                best = ep_loss
                save = {
                    "state_dict": online.state_dict(),
                    "epoch": ep, "loss": ep_loss, "seed": seed,
                    "cfg": cfg,
                }
                torch.save(save, out_ckpt / "best.pt")

    print("[OK] Finished. Best ckpt at:", out_ckpt / "best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--physaug", default="configs/physaug.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=None)  # mac 기본 0, 리눅스는 2
    args = ap.parse_args()
    main(args)
