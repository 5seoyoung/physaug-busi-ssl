import kagglehub, shutil, os, pathlib, json

# Kaggle dataset slug
DATASET = "sabahesaraki/breast-ultrasound-images-dataset"

def main():
    path = kagglehub.dataset_download(DATASET)
    print("[OK] Downloaded to:", path)
    # 표준 경로로 복사/정리
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    dst = repo_root / "data" / "raw" / "BUSI"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copytree(path, dst)
    print("[OK] Copied to:", dst)

    # configs/paths.yaml 자동 작성(없으면)
    cfg = repo_root / "configs" / "paths.yaml"
    if not cfg.exists():
        cfg.write_text(
            "data_root: data\n"
            "busi_raw: data/raw/BUSI\n"
            "processed_root: data/processed\n"
            "outputs_root: outputs\n"
        )
        print("[OK] Wrote configs/paths.yaml")

if __name__ == "__main__":
    main()
