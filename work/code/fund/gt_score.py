"""T8 score: research-bar LOSO with box + residuals from features without garbage time vs the same recipe with it."""
import numpy as np, pandas as pd
D = "/home/claude/data"; FD = "/home/claude/fund"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
R = pd.concat([pd.read_parquet(f"{FD}/gt_refit_{N}.parquet") for N in range(2014, 2025)])
B = B.merge(R, on=["game", "home_team", "away_team"])
F = [c for c in B.columns if c.startswith("f:") and "|" not in c]
y = B.margin.values; seas = B.season.values
def pred(tag, box=None, res=None):
    bt, rt = box or tag, res or tag
    return (sum(B[f"{c}|{bt}"] for c in F if c != "f:o_3eff") + B["f:o_3eff"] + B.unrated + .6 * B[f"res_o|{rt}"] + .8 * B[f"res_d|{rt}"]).to_numpy()
def err(p):
    X = np.c_[np.ones(len(y)), p]; e = np.full(len(y), np.nan)
    for s in np.unique(seas):
        tr, te = seas != s, seas == s; e[te] = (y[te] - X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]) ** 2
    return e
prod = (B[F].sum(axis=1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy()
print("production files LOSO", np.sqrt(err(prod).mean()).round(4), "| rebuilt 'all' LOSO", np.sqrt(err(pred("all")).mean()).round(4),
      "| max |all - prod| pred", np.abs(pred("all") - prod).max().round(3))
e0 = err(pred("all"))
def rep(name, p):
    e1 = err(p); d = pd.Series(e1 - e0).groupby(seas).mean(); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    h1, h2 = seas <= 2019, seas >= 2020
    print(f"{name:34s} LOSO {np.sqrt(e1.mean()):.4f}  d {np.sqrt(e1.mean())-np.sqrt(e0.mean()):+.4f}  t {t:+.2f}  better {(d<0).sum()}/{len(d)}  halves "
          f"{np.sqrt(e1[h1].mean())-np.sqrt(e0[h1].mean()):+.4f} / {np.sqrt(e1[h2].mean())-np.sqrt(e0[h2].mean()):+.4f}")
rep("no garbage (box + residual refit)", pred("ng"))
rep("no garbage, box part only", pred("all", box="ng", res="all"))
rep("no garbage, residual refit only", pred("all", box="all", res="ng"))
# one feature at a time (box part swapped, residuals from the full ng refit kept at 'all')
for c in F:
    if c == "f:o_3eff": continue
    p = pred("all") - B[f"{c}|all"].to_numpy() + B[f"{c}|ng"].to_numpy(); rep("  swap " + c, p)
