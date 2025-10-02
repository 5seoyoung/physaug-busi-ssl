# -*- coding: utf-8 -*-
"""
BUSI 초음파용 PhysAug 파라미터 자동 추정:
- Speckle Contrast(SC) 분포
- Depth attenuation 계수 k 분포
- PSF 이방성(축방/측방 sigma) 근사
결과를 YAML로 저장하여 physaug 변환의 U(·) 범위로 사용.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def load_paths(paths_yaml: str):
    cfg = yaml.safe_load(open(paths_yaml, "r"))
    raw = Path(cfg["busi_raw"])
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    imgs = [p for p in raw.rglob("*") if p.suffix.lower() in exts and "mask" not in p.name.lower()]
    return sorted(imgs)


def _to_gray01(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def estimate_speckle_contrast(img: np.ndarray) -> float:
    # 약한 가우시안 → 그래디언트 → 낮은 그래디언트 영역을 균질 ROI로 사용
    g = cv2.GaussianBlur(img, (3, 3), 0.7)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 40.0)
    roi = img[mag <= thr]
    if roi.size < 100:  # 실패 시 전체
        roi = img
    mean = float(np.mean(roi) + 1e-8)
    std = float(np.std(roi))
    sc = std / mean
    return sc


def estimate_depth_attenuation(img: np.ndarray) -> float:
    # 행별 평균 강도의 로그를 깊이에 대해 선형 적합 → 기울기의 음수 절댓값을 k로 근사
    h = img.shape[0]
    prof = np.mean(img, axis=1) + 1e-6
    y = np.arange(h, dtype=np.float32)
    s, e = int(0.05 * h), int(0.95 * h)
    yy, pp = y[s:e], np.log(prof[s:e])
    A = np.vstack([yy, np.ones_like(yy)]).T
    m, b = np.linalg.lstsq(A, pp, rcond=None)[0]
    k = float(max(0.0, -m)) * 0.01  # 스케일 보정(경험적)
    return k


def estimate_psf_anisotropy(img: np.ndarray):
    # 구조텐서로 방향성 측정 → 측방(가로) 흐림이 더 크면 sigma_lat ↑
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    ex = float(np.mean(Jxx) + 1e-8)
    ey = float(np.mean(Jyy) + 1e-8)
    r = ex / ey  # 가로 에지가 세로보다 크면 r>1
    sigma_ax = 0.7                                   # 기본값(축방)
    sigma_lat = float(np.clip(1.2 * np.sqrt(max(1.0, r)), 1.0, 2.5))  # 측방
    return sigma_ax, sigma_lat


def robust_imread(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 경로/인코딩 이슈 대비
        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def main(args):
    imgs = load_paths(args.paths)
    if len(imgs) == 0:
        raise SystemExit("BUSI 이미지가 보이지 않습니다. configs/paths.yaml의 busi_raw 경로를 확인하세요.")

    sc_list, k_list, sax_list, slat_list = [], [], [], []
    sample_n = min(len(imgs), 500)  # 500장만 샘플
    idx = np.linspace(0, len(imgs) - 1, sample_n, dtype=int)

    for i in tqdm(idx, desc="Calibrating from BUSI"):
        img = robust_imread(imgs[i])
        img = _to_gray01(img)
        if img is None:
            continue
        sc = estimate_speckle_contrast(img)
        k = estimate_depth_attenuation(img)
        sax, slat = estimate_psf_anisotropy(img)
        sc_list.append(sc)
        k_list.append(k)
        sax_list.append(sax)
        slat_list.append(slat)

    def q(x, lo, hi):
        x = np.asarray(x, dtype=np.float32)
        return float(np.quantile(x, lo)), float(np.quantile(x, hi))

    sc_lo, sc_hi = q(sc_list, 0.2, 0.8)
    k_lo, k_hi = q(k_list, 0.2, 0.8)
    ax_lo, ax_hi = q(sax_list, 0.2, 0.8)
    lat_lo, lat_hi = q(slat_list, 0.2, 0.8)

    cfg = {
        "speckle": {
            "sc_min": round(sc_lo, 3),
            "sc_max": round(sc_hi, 3),
            "corr_px_min": 1.0,
            "corr_px_max": 2.0,
        },
        "attenuation": {
            "k_min": round(max(0.001, k_lo), 4),
            "k_max": round(min(0.02, k_hi), 4),
            "tgc_jitter": 0.1,
        },
        "psf": {
            "sigma_ax_min": round(ax_lo, 2),
            "sigma_ax_max": round(ax_hi, 2),
            "sigma_lat_min": round(lat_lo, 2),
            "sigma_lat_max": round(lat_hi, 2),
        },
        "shadow": {"beta_min": 0.5, "beta_max": 1.5, "gamma_min": 0.1, "gamma_max": 0.4},
        "logcomp": {"gain_min": 0.8, "gain_max": 1.2},
    }

    out = Path(args.out)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[OK] Wrote {out} with ranges:\n{json.dumps(cfg, indent=2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=str, default="configs/paths.yaml")
    ap.add_argument("--out", type=str, default="configs/physaug.yaml")
    args = ap.parse_args()
    main(args)
