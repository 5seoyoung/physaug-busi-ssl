# -*- coding: utf-8 -*-
import numpy as np, cv2, random

class PhysAug:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, img, resize=128):
        if img.ndim==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.max()>1.0: img = img.astype(np.float32)/255.0
        h,w = img.shape
        out = cv2.resize(img, (resize,resize), interpolation=cv2.INTER_AREA)

        if random.random()<0.6: out = self._psf(out)
        if random.random()<0.7: out = self._speckle(out)
        if random.random()<0.5: out = self._atten(out)
        if random.random()<0.3: out = self._shadow_or_enh(out)
        if random.random()<0.2: out = self._scanline(out)
        if random.random()<0.7: out = self._logcomp(out)
        return np.clip(out, 0, 1)

    def _u(self, a,b): return random.uniform(a,b)

    def _speckle(self, x):
        c = self.cfg["speckle"]
        sc = self._u(c["sc_min"], c["sc_max"])
        # Rayleigh sigma from SC≈sqrt(4/π -1)*σ / mean ≈ k*σ / mean → 간단하게 분산 비례 샘플
        sigma = 0.5*sc
        s = np.random.rayleigh(scale=sigma, size=x.shape).astype(np.float32)
        # 약간의 상관: 작은 가우시안 블러
        corr = self._u(c["corr_px_min"], c["corr_px_max"])
        s = cv2.GaussianBlur(s, (0,0), corr)
        return x * (1.0 + s)

    def _atten(self, x):
        a = self.cfg["attenuation"]
        k = self._u(a["k_min"], a["k_max"])
        h,w = x.shape
        y = np.arange(h, dtype=np.float32).reshape(-1,1)
        A = np.exp(-k * (y / h) * 100.0)  # 정규화 깊이
        # 간단한 TGC 지터(구간별 선형)
        tgc = 1.0 + np.interp((y/h).ravel(),
                              [0.0,0.33,0.66,1.0],
                              [self._u(-a["tgc_jitter"],a["tgc_jitter"]) for _ in range(4)]
                              ).reshape(h,1)
        return x * A * tgc

    def _psf(self, x):
        p = self.cfg["psf"]
        sax = self._u(p["sigma_ax_min"], p["sigma_ax_max"])
        slat= self._u(p["sigma_lat_min"], p["sigma_lat_max"])
        # 축방(세로) -> (ksize_y), 측방(가로) -> (ksize_x)
        kx = int(max(1, round(slat*3))*2+1)
        ky = int(max(1, round(sax*3))*2+1)
        return cv2.GaussianBlur(x, (kx,ky), 0)

    def _shadow_or_enh(self, x):
        h,w = x.shape
        m = np.zeros_like(x)
        cx, cy = int(self._u(0.3*w, 0.7*w)), int(self._u(0.3*h, 0.7*h))
        rx, ry = int(self._u(0.08*w,0.18*w)), int(self._u(0.12*h,0.22*h))
        cv2.ellipse(m, (cx,cy), (rx,ry), 0, 0, 360, 1, -1)
        # 선적분 근사: 누적합으로 번짐
        line = cv2.GaussianBlur(m, (0,0), 3.0)
        if random.random()<0.5:
            beta = self._u(self.cfg["shadow"]["beta_min"], self.cfg["shadow"]["beta_max"])
            return x * np.exp(-beta * line)
        else:
            gamma = self._u(self.cfg["shadow"]["gamma_min"], self.cfg["shadow"]["gamma_max"])
            return x * (1.0 + gamma * line)

    def _scanline(self, x):
        h,w = x.shape
        n = random.randint(0,2)
        for _ in range(n):
            col = random.randint(0,w-1)
            wv = random.randint(1,2)
            x[:, col:col+wv] *= self._u(0.2,0.6)
        return x

    def _logcomp(self, x):
        g = self.cfg["logcomp"]
        gain = self._u(g["gain_min"], g["gain_max"])
        y = np.log1p(gain * x)
        y /= y.max()+1e-8
        return y
