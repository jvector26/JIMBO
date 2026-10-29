import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "model"); import jimbo_sim as SIM
from scipy.stats import norm
D = pd.read_parquet("replay_rows.parquet"); sched = pickle.load(open("replay_sched.pkl", "rb"))
SIG, HCA = SIM.SIG, SIM.HCA
def tau_cur(avg): return np.interp(avg, [0, 41, 70, 82], [2.4216, 2.0, 1.8, 1.8])
def exp_wins(x, k, tau=None):
    dec = 1.0 if k == np.inf else (k / (k + x.gp) if k > 0 else 0 * x.gp)
    m = (x.model + x.gap * dec).to_numpy(); m = m - m.mean()
    idx = x.t.map(SIM.IDX).to_numpy(); mm = np.zeros(30); mm[idx] = m
    r = sched[(x.S.iloc[0], x.cp.iloc[0])]
    tau = tau_cur(x.gp.mean()) if tau is None else tau
    p = norm.cdf((mm[r[:, 0]] - mm[r[:, 1]] + HCA) / np.sqrt(SIG**2 + 2 * tau**2))
    ew = np.zeros(30); np.add.at(ew, r[:, 0], p); np.add.at(ew, r[:, 1], 1 - p)
    return x.w.to_numpy() + ew[idx]
KS = [0, 5, 10, 20, 40, 80, 160, np.inf]
res = []
for (S, cp), x in D.groupby(["S", "cp"]):
    if cp == 0: continue
    for k in KS:
        e = exp_wins(x, k) - x.final_w.to_numpy()
        res.append(dict(S=S, cp=cp, k=k, sse=(e**2).sum()))
    # win-pace baseline
    pace = x.w / x.gp.clip(lower=1) * (x.gp + x.g_rem)
    res.append(dict(S=S, cp=cp, k=-1, sse=((pace - x.final_w)**2).sum()))
R = pd.DataFrame(res)
tab = R.groupby(["cp", "k"]).sse.sum().unstack() / (30 * 12); tab = np.sqrt(tab)
print("final-wins RMSE by checkpoint (rows) x gap_k (cols; -1 = win pace, inf = never decay)"); print(tab.round(3).to_string())
tot = np.sqrt(R[R.k >= 0].groupby("k").sse.sum() / (30 * 12 * 7)); print("all cps:", tot.round(3).to_dict())
R.to_parquet("gapk_res.parquet")
