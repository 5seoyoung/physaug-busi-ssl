# -*- coding: utf-8 -*-
"""
BUSI 초음파용 PhysAug 파라미터 자동 추정:
- Speckle Contrast(SC) 분포
- Depth Attenuation 계수 k 분포
- PSF 이방성(축방/측방 sigma) 근사
결과를 YAML로 저장하여 physaug 변환의 U(·) 범위로 사용.
"""
import argparse, yaml, numpy as np, cv2, json
from pathlib import Path
from tqdm import tqdm

def load_paths(paths_yaml):
    d = yaml.safe_load(open(paths_yaml))
    raw = Path(d["busi_raw"])
    imgs = sorted(list(raw.rglob("*.png")) + list(raw.rglob("*.jpg")) + list(raw.rglob("*.jpeg")))
    # BUSI는 폴더 구조가 클래스별로 나뉘어 있음(Benign/Malignant/Normal, 마스크는 _mask)
    imgs = [p for p in imgs if "mask" not in p.name.lower()]
    return imgs

def _to_gray01(img):
    if img.ndim==3: img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max()>1.0: img/=255.0
    return img

def estimate_speckle_contrast(img):
    # 균질 배경 ROI를 찾기 위해 약한 가우시안 → 소벨 → 낮은 그래디언트 픽셀
    g = cv2.GaussianBlur(img, (3,3), 0.7)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    thr = np.percentile(mag, 40)  # 하위 40%를 균질영역 근사
    roi = img[mag<=thr]
    if roi.size<100: roi = img  # 실패 시 전체 사용
    mean = float(np.mean(roi)+1e-8)
    std  = float(np.std(roi))
    sc = std/mean
    return sc

def estimate_depth_attenuation(img):
    # 행별 평균 강도 -> 로그 선형 적합하여 기울기 음수의 절댓값을 k로 근사
    h = img.shape[0]
    prof = np.mean(img, axis=1) + 1e-6
    y = np.arange(h, dtype=np.float32)
    # 상하 5% 영역 제외(크롭) 후 선형회귀
    s, e = int(0.05*h), int(0.95*h)
    yy, pp = y[s:e], np.log(prof[s:e])
    A = np.vstack([yy, np.ones_like(yy)]).T
    m, b = np.linalg.lstsq(A, pp, rcond=None)[0]
    k = float(max(0.0, -m)) * 0.01  # 스케일링(경험적)
    return k

def estimate_psf_anisotropy(img):
    # 구조텐서로 방향성 측정 → 측방(가로) 흐림이 더 크다고 가정해 sigma_lat/sigma_ax 추정
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx*gx, (0,0), 2.0)
    Jyy = cv2.GaussianBlur(gy*gy, (0,0), 2.0)
    # 축방(세로) 성분이 더 크면 측방 해상도가 낮다는 뜻 → sigma_lat가 큼
    ex = float(np.mean(Jxx)+1e-8); ey = float(np.mean(Jyy)+1e-8)
    # 간단 비율 맵핑(경험적): sigma_lat ~ f(ex/ey)
    r = ex/ey
    sigma_ax = 0.7   # 기본
    sigma_lat = float(np.clip(1.2*np.sqrt(max(1.0, r)), 1.0, 2.5))
    return sigma_ax, sigma_lat

def main(args):
    imgs = load_paths(args.paths)
    if len(imgs)==0:
        raise SystemExit("BUSI 이미지가 보이지 않습니다. configs/paths.yaml의 busi_raw 경로를 확인하세요.")

    sc_list, k_list, sax_list, slat_list = [], [], [], []
    sample_n = min(len(imgs), 500)  # 500장 샘플만 통계
    idx = np.linspace(0, len(imgs)-1, sample_n, dtype=int)

    for i in tqdm(idx, desc="Calibrating from BUSI"):
        img = cv2.imdecode(np.fromfile(str(imgs[i]), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = _to_gray01(img)
        sc = estimate_speckle_contrast(img)
        k  = estimate_depth_attenuation(img)
        sax, slat = estimate_psf_anisotropy(img)
        sc_list.append(sc); k_list.append(k); sax_list.append(sax); slat_list.append(slat)

    def q(x, lo, hi): 
        return float(np.quantile(x, lo)), float(np.quantile(x, hi))

    sc_lo, sc_hi   = q(sc_list, 0.2, 0.8)   # 과격치 배제
    k_lo, k_hi     = q(k_list, 0.2, 0.8)
    ax_lo, ax_hi   = q(sax_list, 0.2, 0.8)
    lat_lo, lat_hi = q(slat_list, 0.2, 0.8)

    cfg = {
      "speckle": {"sc_min": round(sc_lo,3), "sc_max": round(sc_hi,3), "corr_px_min":1.0, "corr_px_max":2.0},
      "attenuation": {"k_min": round(max(0.001, k_lo),4), "k_max": round(min(0.02, k_hi),4),
                      "tgc_jitter": 0.1},
      "psf": {"sigma_ax_min": round(ax_lo,2), "sigma_ax_max": round(ax_hi,2),
              "sigma_lat_min": round(lat_lo,2), "sigma_lat_max": round(lat_hi,2)},
      "shadow": {"beta_min":0.5, "beta_max":1.5, "gamma_min":0.1, "gamma_max":0.4},
      "logcomp": {"gain_min":0.8, "gain_max":1.2}
    }

    out = Path(args.out)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[OK] Wrote {out} with ranges:")
    print(json.dumps(cfg, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=str, default="configs/paths.yaml")
    ap.add_argument("--out", type=str, default="configs/physaug.yaml")
    args = ap.parse_args()
    main(args)
