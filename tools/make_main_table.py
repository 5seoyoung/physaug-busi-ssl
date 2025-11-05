# tools/make_main_table.py
import os, re, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss

# === 사용할 실험 폴더(리뷰 혼선 유발 가능한 *_lp_*는 제외) ===
EXPS = [
    ("vanilla_0.1_seed0",  "outputs/runs/cls/vanilla_0.1_seed0"),
    ("physaug_0.1_seed0",  "outputs/runs/cls/physaug_0.1_seed0"),
    ("vanilla_1.0_seed0",  "outputs/runs/cls/vanilla_1.0_seed0_eval"),
    ("physaug_1.0_seed0",  "outputs/runs/cls/physaug_1.0_seed0"),
]

OUT_DIR = "outputs/tables"
OUT_CSV = os.path.join(OUT_DIR, "main_table_seed0.csv")
OUT_TSV = os.path.join(OUT_DIR, "main_table_seed0.tsv")
OUT_MD  = os.path.join(OUT_DIR, "main_table_seed0.md")

os.makedirs(OUT_DIR, exist_ok=True)

def dbg(msg): print(f"[make_table] {msg}", flush=True)

def last_csv(d):
    cands = sorted(glob.glob(os.path.join(d, "val_pred_*.csv")))
    return cands[-1] if cands else None

def pick_prob_columns(df):
    """
    확률 컬럼 자동 탐색:
    - p0,p1,p2
    - prob_0, prob_1, prob_2
    - softmax_0, softmax_1, softmax_2
    - logits_*가 있으면 softmax로 변환해서 사용
    """
    cols = list(df.columns)

    # 우선 logits 패턴 확인
    logit_cols = [c for c in cols if re.match(r"^(logit|logits)[_\-]?(\d+)$", c)]
    if logit_cols:
        # logit_* -> softmax로 변환
        logit_cols = sorted(logit_cols, key=lambda x: int(re.findall(r"(\d+)$", x)[0]))
        logits = df[logit_cols].to_numpy(dtype=float)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        return prob

    # 확률 패턴 후보들
    patterns = [
        r"^p(\d+)$",
        r"^prob[_\-]?(\d+)$",
        r"^softmax[_\-]?(\d+)$",
        r"^probability[_\-]?(\d+)$",
    ]
    # 가장 많은 매칭을 만드는 패턴 채택
    best = None
    for pat in patterns:
        matches = [(c, int(re.findall(pat, c)[0])) for c in cols if re.match(pat, c)]
        if matches:
            matches.sort(key=lambda x: x[1])  # index 순 정렬
            arr = df[[c for c,_ in matches]].to_numpy(dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                best = arr
                break
    return best  # None 허용

def macro_auroc(y_true, prob):
    if prob is None or prob.ndim != 2 or prob.shape[1] < 2:
        return np.nan
    K = prob.shape[1]
    y_onehot = np.eye(K)[y_true]
    try:
        return roc_auc_score(y_onehot, prob, average='macro', multi_class='ovr')
    except Exception:
        return np.nan

def ece(prob, y_true, n_bins=15):
    if prob is None: return np.nan
    conf = prob.max(1)
    preds = prob.argmax(1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    s=0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum()==0: 
            continue
        s += abs(correct[m].mean() - conf[m].mean()) * (m.mean())
    return s

def load_csv_metrics(csv_path):
    df = pd.read_csv(csv_path)
    # 필수 라벨 컬럼 유연 매핑
    ytrue_col = None
    ypred_col = None
    for cand in ["y_true","label","target","gt","y"]:
        if cand in df.columns: ytrue_col = cand; break
    for cand in ["y_pred","pred","prediction","yp","yhat","y_hat"]:
        if cand in df.columns: ypred_col = cand; break
    if ytrue_col is None or ypred_col is None:
        raise ValueError(f"CSV missing label columns in {csv_path}")

    y_true = df[ytrue_col].astype(int).to_numpy()
    y_pred = df[ypred_col].astype(int).to_numpy()
    prob = pick_prob_columns(df)  # None 가능

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    auroc = macro_auroc(y_true, prob)
    e     = ece(prob, y_true) if prob is not None else np.nan
    brier = (np.mean([brier_score_loss((y_true==k).astype(int), prob[:,k]) for k in range(prob.shape[1])])
             if prob is not None else np.nan)
    nll   = (log_loss(y_true, prob, labels=list(range(prob.shape[1])))
             if prob is not None else np.nan)

    return dict(Acc=acc, MacroF1=f1, MacroAUROC=auroc, ECE=e, Brier=brier, NLL=nll)

def fallback_metrics(dir_path):
    """
    CSV가 없을 때 metrics.json 또는 metrics_epoch*.json에서 최소치(Acc, MacroF1)만 수집
    """
    mjson = os.path.join(dir_path, "metrics.json")
    if os.path.isfile(mjson):
        try:
            j = json.load(open(mjson, "r"))
            return dict(Acc=float(j.get("best_acc", np.nan)),
                        MacroF1=float(j.get("best_macro_f1", np.nan)),
                        MacroAUROC=np.nan, ECE=np.nan, Brier=np.nan, NLL=np.nan,
                        _note="metrics.json")
        except Exception:
            pass
    cands = sorted(glob.glob(os.path.join(dir_path, "metrics_epoch*.json")))
    if cands:
        last = cands[-1]
        try:
            j = json.load(open(last, "r"))
            return dict(Acc=float(j.get("acc", np.nan)),
                        MacroF1=float(j.get("macro_f1", np.nan)),
                        MacroAUROC=np.nan, ECE=np.nan, Brier=np.nan, NLL=np.nan,
                        _note=os.path.basename(last))
        except Exception:
            pass
    return None

rows=[]
for tag, d in EXPS:
    dbg(f"scan: {tag} -> {d}")
    if not os.path.isdir(d):
        dbg("  ! dir missing")
        rows.append(dict(Exp=tag, Note="MISSING_DIR", Acc=np.nan, MacroF1=np.nan,
                         MacroAUROC=np.nan, ECE=np.nan, Brier=np.nan, NLL=np.nan))
        continue

    csv_path = last_csv(d)
    if csv_path:
        dbg(f"  + csv: {os.path.basename(csv_path)}")
        try:
            m = load_csv_metrics(csv_path)
            rows.append(dict(Exp=tag, Note=os.path.basename(csv_path), **m))
        except Exception as e:
            dbg(f"  ! csv parse fail: {e}")
            fb = fallback_metrics(d)
            if fb:
                rows.append(dict(Exp=tag, Note=fb.pop("_note"), **fb))
            else:
                rows.append(dict(Exp=tag, Note="CSV_PARSE_FAIL", Acc=np.nan, MacroF1=np.nan,
                                 MacroAUROC=np.nan, ECE=np.nan, Brier=np.nan, NLL=np.nan))
    else:
        dbg("  - no csv, try fallback")
        fb = fallback_metrics(d)
        if fb:
            rows.append(dict(Exp=tag, Note=fb.pop("_note"), **fb))
        else:
            rows.append(dict(Exp=tag, Note="NO_CSV_NO_FALLBACK", Acc=np.nan, MacroF1=np.nan,
                             MacroAUROC=np.nan, ECE=np.nan, Brier=np.nan, NLL=np.nan))

df = pd.DataFrame(rows, columns=["Exp","Acc","MacroF1","MacroAUROC","ECE","Brier","NLL","Note"])
dbg("=== TABLE ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

df.to_csv(OUT_CSV, index=False)
df.to_csv(OUT_TSV, index=False, sep="\t")

# Markdown도 같이 저장 (워드 붙여넣기 편하게)
md = "| Exp | Acc | MacroF1 | MacroAUROC | ECE | Brier | NLL | Note |\n|---|---:|---:|---:|---:|---:|---:|---|\n"
for _, r in df.iterrows():
    def fmt(v): 
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else ("NaN" if isinstance(v, float) else str(v))
    md += f"| {r['Exp']} | {fmt(r['Acc'])} | {fmt(r['MacroF1'])} | {fmt(r['MacroAUROC'])} | {fmt(r['ECE'])} | {fmt(r['Brier'])} | {fmt(r['NLL'])} | {r['Note']} |\n"
open(OUT_MD, "w").write(md)
dbg(f"saved: {OUT_CSV}")
dbg(f"saved: {OUT_TSV}")
dbg(f"saved: {OUT_MD}")
