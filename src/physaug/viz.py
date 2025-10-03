# -*- coding: utf-8 -*-
import argparse, yaml, numpy as np, cv2, sys
from pathlib import Path
import numpy as np

# 로컬 robust imread (calib와 동일 아이디어)
def robust_imread(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img

# 간단한 PhysAug 호출 (src/physaug/physaug.py)
from .physaug import PhysAug

def main(args):
    paths_cfg = yaml.safe_load(open(args.paths, "r"))
    root = Path(paths_cfg["busi_raw"])

    # 모든 주요 확장자 + 대소문자 대응
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in exts and "mask" not in p.name.lower()]
    imgs = sorted(imgs)

    print(f"[INFO] Found {len(imgs)} images (excluding *mask*).")
    if len(imgs) == 0:
        print("[WARN] No images found. Check configs/paths.yaml:busi_raw path and dataset structure.")
        sys.exit(0)

    cfg = yaml.safe_load(open(args.physaug, "r"))
    aug = PhysAug(cfg)

    outdir = Path("outputs/figures/preview")
    outdir.mkdir(parents=True, exist_ok=True)

    # 처음 8장만 미리보기
    for p in imgs[:8]:
        im = robust_imread(p)
        if im is None:
            print(f"[SKIP] Cannot read {p}")
            continue
        im = im.astype(np.float32) / 255.0
        im_aug = aug(im, resize=224)
        pair = np.hstack([cv2.resize(im, (224,224)), im_aug])
        save = outdir / (p.stem + "_pair.png")
        cv2.imwrite(str(save), (pair * 255).astype(np.uint8))
        print("[OK] saved", save)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--physaug", default="configs/physaug.yaml")
    main(ap.parse_args())
