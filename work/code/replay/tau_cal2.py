"""End-to-end calibration of the rest-of-season sim's team uncertainty tau (gap_k 20).
Per season/checkpoint: true = forecast + tau*z (z common across tau), games ~ Phi((dh-da+HCA)/SIG) with common uniforms.
Scores: p10-p90 coverage of final wins, CRPS of final wins, playoff-odds (top-10 in conference) Brier."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "model"); import jimbo_sim as SIM
from scipy.stats import norm
D = pd.read_parquet("replay_rows.parquet"); sched = pickle.load(open("replay_sched.pkl", "rb"))
TAUS = np.array([0, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]); N = 3000; K = 20
rows = []
for (S, cp), x in D.groupby(["S", "cp"]):
    x = x.set_index("t").reindex(SIM.TEAMS)
    dec = K / (K + x.gp); m = (x.model + x.gap * dec).to_numpy().copy(); m = m - m.mean()
    r = sched[(S, cp)]; rng = np.random.default_rng(S * 100 + cp)
    z = rng.normal(size=(N, 30)); u = rng.random((N, len(r)))
    fw = x.final_w.to_numpy()
    for tau in TAUS:
        t = m[None] + tau * z
        p = norm.cdf((t[:, r[:, 0]] - t[:, r[:, 1]] + SIM.HCA) / SIM.SIG); hw = u < p
        W = np.tile(x.w.to_numpy(), (N, 1)).astype(float)
        for k in range(30):
            W[:, k] += hw[:, r[:, 0] == k].sum(1) + (~hw[:, r[:, 1] == k]).sum(1)
        p10, p90 = np.percentile(W, 10, 0), np.percentile(W, 90, 0)
        # CRPS for an empirical ensemble: E|X-y| - 0.5 E|X-X'|
        Ws = np.sort(W, 0); i = np.arange(1, N + 1)[:, None]
        e_xx = 2 * ((2 * i - N - 1) * Ws).sum(0) / N**2
        crps = np.abs(W - fw).mean(0) - 0.5 * e_xx
        mean = W.mean(0)
        v = rng.random(30); pit = (W < fw).mean(0) + v * (W == fw).mean(0)
        # beyond-80% band, also 50% band
        p25, p75 = np.percentile(W, 25, 0), np.percentile(W, 75, 0)
        for j, tm in enumerate(SIM.TEAMS):
            rows.append(dict(S=S, cp=cp, t=tm, tau=tau, cov80=(p10[j] <= fw[j] <= p90[j]), cov50=(p25[j] <= fw[j] <= p75[j]),
                             crps=crps[j], pit=pit[j], err=mean[j] - fw[j], gp=x.gp.iloc[j]))
    print(S, cp, flush=True, end="; ")
pd.DataFrame(rows).to_parquet("tau_res2.parquet")
