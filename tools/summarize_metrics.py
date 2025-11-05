# save as: tools/summarize_metrics.py
import os, re, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss

def ece_score(y_true, prob, n_bins=15):
    # multiclass ECE (top-1 confidence)
    conf = prob.max(1)
    preds = prob.argmax(1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() == 0: 
            continue
        ece += np.abs(correct[m].mean() - conf[m].mean()) * (m.mean())
    return ece

def auroc_macro(y_true, prob):
    # one-vs-rest macro AUROC
    K = prob.shape[1]
    y = np.eye(K)[y_true]
    try:
        return roc_auc_score(y, prob, average='macro', multi_class='ovr')
    except Exception:
        return np.nan

def find_best_epoch_csv(out_dir):
    # 우선 best epoch 유추: 파일명에 epochNNN이 들어가므로 가장 마지막(best 기준 저장 시) 혹은 로그에서 best를 찾아도 ok
    cands = sorted(glob.glob(os.path.join(out_dir, "val_pred_epoch*.csv")))
    return cands[-1] if cands else None

def summarize_out(out_dir):
    path = find_best_epoch_csv(out_dir)
    if not path:
        return None
    df = pd.read_csv(path)
    y_true = df['y_true'].values.astype(int)
    y_pred = df['y_pred'].values.astype(int)
    prob = df[[c for c in df.columns if re.match(r"^p\d+$", c)]].values

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro')
    au   = auroc_macro(y_true, prob)
    # Brier: 평균(one-vs-all)로 간단화
    brs  = np.mean([brier_score_loss((y_true==k).astype(int), prob[:,k]) for k in range(prob.shape[1])])
    # NLL (cross-entropy)
    try:
        nll = log_loss(y_true, prob, labels=list(range(prob.shape[1])))
    except Exception:
        nll = np.nan
    ece  = ece_score(y_true, prob)
    return dict(out_dir=out_dir, Acc=acc, MacroF1=f1, MacroAUROC=au, ECE=ece, Brier=brs, NLL=nll)

def main():
    roots = [
        "outputs/runs/cls/physaug_10p_e20",
        "outputs/runs/cls/physaug_100p_e20",
        "outputs/runs/cls/vanilla_10p_e20",
        "outputs/runs/cls/vanilla_100p_e20",
        # seeds로 여러 번 돌렸다면 여기에 out_dir 추가
    ]
    rows = []
    for r in roots:
        s = summarize_out(r)
        if s: rows.append(s)
    out = pd.DataFrame(rows).sort_values("out_dir")
    print(out.to_string(index=False))
    out.to_csv("outputs/metrics_summary.csv", index=False)

if __name__ == "__main__":
    main()
