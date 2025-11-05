import re, json, glob, statistics as st
from pathlib import Path
import pandas as pd

rows = []
for log in glob.glob("outputs/runs/cv/*/train_full.log"):
    p = Path(log)
    exp = p.parent.name
    # tag_r{ratio}_f{fold}
    m = re.match(r"(vanilla|physaug)_r([0-9.]+)_f(\d+)", exp)
    if not m: 
        continue
    model, ratio, fold = m.group(1), float(m.group(2)), int(m.group(3))
    txt = Path(log).read_text()
    # best_acc
    m2 = re.findall(r"best_acc=([0-9.]+)", txt)
    best_acc = float(m2[-1]) if m2 else None
    # macro_f1 (마지막 Epoch 기준)
    m3 = re.findall(r"\[Metrics\] epoch=\d+ \| macro_f1=([0-9.]+)", txt)
    macro_f1_last = float(m3[-1]) if m3 else None
    rows.append(dict(model=model, ratio=ratio, fold=fold, acc=best_acc, f1=macro_f1_last))

df = pd.DataFrame(rows).sort_values(["model","ratio","fold"])
summary = []
for (model, ratio), g in df.groupby(["model","ratio"]):
    r = dict(model=model, ratio=ratio,
             acc_mean=st.mean(g["acc"]), acc_std=st.pstdev(g["acc"]),
             f1_mean=st.mean(g["f1"]), f1_std=st.pstdev(g["f1"]))
    summary.append(r)
sdf = pd.DataFrame(summary).sort_values(["ratio","model"])
Path("outputs/runs/cv/summary.csv").write_text(sdf.to_csv(index=False))
print(sdf.to_string(index=False))
