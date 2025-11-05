# -*- coding: utf-8 -*-
import argparse, yaml, numpy as np, cv2
from pathlib import Path
from .physaug import PhysAug  # 아래 ➌에 정의

def main(args):
    cfg = yaml.safe_load(open(args.physaug))
    root = Path(yaml.safe_load(open(args.paths))["busi_raw"])
    # 샘플 6장
    imgs = [p for p in root.rglob("*.png")] + [p for p in root.rglob("*.jpg")]
    imgs = [p for p in imgs if "mask" not in p.name.lower()][:6]
    aug = PhysAug(cfg)

    outdir = Path("outputs/figures/preview"); outdir.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        im_aug = aug(im, resize=224)
        pair = np.hstack([im, im_aug])
        save = outdir/(p.stem+"_pair.png")
        cv2.imwrite(str(save), (pair*255).astype(np.uint8))
        print("[OK] saved", save)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--physaug", default="configs/physaug.yaml")
    main(ap.parse_args())
