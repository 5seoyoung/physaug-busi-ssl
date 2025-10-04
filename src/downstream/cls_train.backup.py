print('[BOOT] cls_train imported')


if __name__ == "__main__":
    import argparse, yaml
    print("[BOOT] __main__ entry")
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=str, required=True)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--label_ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="outputs/runs/cls/physaug_smoke")
    # 선택 인자(있으면 쓰고, 없어도 무시되게)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=128)
    args = ap.parse_args()
    main(args)
