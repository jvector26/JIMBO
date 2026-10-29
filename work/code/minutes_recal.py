"""Recalibration variants of the preseason minutes model, scored LOSO on player minutes and team wins (2014-25 excl 2019-20
for player fits; team RMSE over 330 team-seasons excl 2019 as in score_preseason)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import norm
D = "/home/claude/data"
M = pd.read_parquet(f"{D}/minutes_cal.parquet")
M["act_av"] = (M.gp / M.ng).clip(upper=1); M["act82"] = M.mins * 82 / M.ng; M["act_mpg"] = M.mins / M.gp.replace(0, np.nan)
M["vet"] = M.mpg_wp.notna(); M["reg"] = (M.mpg_wp >= 25).astype(float)
FIT = ~M.season.isin([2019, 2020])
A = pd.read_parquet(f"{D}/preseason_backtest.parquet")
pace = {S: (lambda T: (T.margin / T.net).mean() * 100)(pd.read_parquet(f"{D}/pre_T_{S}.parquet")) for S in range(2014, 2026)}
def allocate(m):
    m = m.clip(upper=3000).copy(); excess = m.sum() - 19680
    if excess > 0:
        for i in m.sort_values().index:
            cut = min(excess, m[i]); m[i] -= cut; excess -= cut
            if excess <= 0: break
    else:
        for _ in range(10):
            room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
    return m
def ls(X, y, w=None):
    w = np.ones(len(y)) if w is None else w
    return np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
def variant(name):
    out = []
    for S in range(2014, 2026):
        tr = M[FIT & (M.season != S)]; te = M[M.season == S].copy()
        m = te.min_model_age.copy()
        if name == "linear":      # total minutes: a + b*model
            b = ls(np.c_[np.ones(len(tr)), tr.min_model_age], tr.act82.values); m = b[0] + b[1] * te.min_model_age
        if name in ("avail", "avail+mpg"):
            v = tr[tr.vet]; Xa = lambda d: np.c_[d.reg, d.reg * d.avail_p, 1 - d.reg, (1 - d.reg) * d.avail_p]
            ba = ls(Xa(v), v.act_av.values); av = np.clip(Xa(te.fillna({"avail_p": 0})) @ ba, 0.05, 1)
            mp = te.mpg_wp
            if name == "avail+mpg":
                v2 = v[v.gp >= 5]; bmp = ls(np.c_[np.ones(len(v2)), v2.mpg_wp], v2.act_mpg.values, v2.gp.values)
                mp = bmp[0] + bmp[1] * te.mpg_wp
            vm = (mp * 82 * av * te.amult).clip(0, 3000)
            # re-level the age multipliers on the new projection (so totals stay calibrated)
            m = vm.where(te.vet, te.min_model_age)
        te["m_new"] = m.clip(lower=0)
        te["min_new"] = te.groupby("team", group_keys=False)["m_new"].apply(allocate)
        out.append(te)
    P = pd.concat(out)
    s = P[FIT.reindex(P.index).fillna(False).values & ~P.season.isin([2019, 2020])]
    s = P[~P.season.isin([2019, 2020])]
    rm = np.sqrt(((s.act82 - s.min_new) ** 2).mean())
    s["rk"] = s.groupby("season").min_new.rank(ascending=False); t = s[s.rk <= 30]
    # team wins (players with ratings = pre_P roster; absentees have no rating -> excluded, as in the existing backtest)
    q = P[P.rating.notna()].copy()
    q["min_t"] = q.groupby(["season", "team"], group_keys=False)["m_new"].apply(allocate)
    net = (q.rating * q.min_t / 3936).groupby([q.season, q.team]).sum()
    net = net - net.groupby(level=0).transform("mean")
    T = net.rename("net").reset_index(); T["wp"] = norm.cdf(T.net * T.season.map(pace) / 100 / 12.8)
    T = T.merge(A[["season", "team", "wpct"]], on=["season", "team"]); T = T[T.season != 2019]
    tr_ = np.sqrt(((82 * (T.wp - T.wpct)) ** 2).mean())
    print(f"{name:10s} player min RMSE {rm:6.1f} | top-30 proj {t.min_new.mean():5.0f} act {t.act82.mean():5.0f} | team RMSE {tr_:.3f} (n {len(T)})")
    return P
for v in ("current", "linear", "avail", "avail+mpg"): variant(v)

def team_by_season(P):
    q = P[P.rating.notna()].copy()
    q["min_t"] = q.groupby(["season", "team"], group_keys=False)["m_new"].apply(allocate)
    net = (q.rating * q.min_t / 3936).groupby([q.season, q.team]).sum(); net = net - net.groupby(level=0).transform("mean")
    T = net.rename("net").reset_index(); T["wp"] = norm.cdf(T.net * T.season.map(pace) / 100 / 12.8)
    T = T.merge(A[["season", "team", "wpct"]], on=["season", "team"]); T = T[T.season != 2019]
    return T.assign(se=(82 * (T.wp - T.wpct)) ** 2).groupby("season").se.mean()
if __name__ == "__main__":
    import sys
    base = team_by_season(variant("current"))
    for v in ("avail", "avail+mpg"):
        P = variant(v); d = team_by_season(P) - base
        print(v, "team MSE diff by season:", d.round(2).to_dict(), "t", round(d.mean() / d.std() * np.sqrt(len(d)), 2))
        s = P[~P.season.isin([2019, 2020])].copy(); s["bin"] = pd.cut(s.min_new, [-1, 500, 1000, 1500, 2000, 2250, 2500, 3001])
        print(s.groupby("bin", observed=True).agg(n=("act82", "size"), proj=("min_new", "mean"), act=("act82", "mean")).round(0).T.to_string())
