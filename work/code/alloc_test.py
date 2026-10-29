"""Crowded-roster minutes rule test (user question 2026-09-21: Dieng/Sims zeroed at MIL).
Same framework as minutes_age_fix.py: pre_P min_model x LOSO age multipliers -> team allocation to 19,680 -> team margin
-> wins vs actual (preseason backtest 2014-25 excl 2019). Also player-minute error vs actual.
Rules when a roster wants more than 19,680:
  bottom  = current: zero the smallest projections first (trim from the bottom)
  prop    = scale everyone down proportionally
  pow g   = m_new = min(m, c*m^g), c solved so the team sums to 19,680 (g=1 prop; large g -> bottom)"""
import numpy as np, pandas as pd
from scipy.stats import norm
D = "/home/claude/data"
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
EDGES = [0, 22, 25, 28, 31, 34, 50]; LABS = ["<=22", "23-25", "26-28", "29-31", "32-34", "35+"]
rows = []
for S in range(2014, 2026):
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
    last = ps[ps.season < S].sort_values("season").groupby("player").last()
    P["age"] = (P.player.map(last.age) + (S - P.player.map(last.season))).fillna(21)
    ng = {2019: 1056 * 2 / 30, 2020: 72}.get(S, 82)
    P["act_min"] = P.player.map(ps[ps.season == S].groupby("player")["min"].sum()).fillna(0) * 82 / ng
    P["bin"] = pd.cut(P.age, EDGES, labels=LABS).astype(str)
    rows.append(P)
P = pd.concat(rows, ignore_index=True)
mults = lambda tr: (lambda t: t.act_min / t.min_model)(tr[tr.season != 2019].groupby("bin")[["act_min", "min_model"]].sum())
parts = []
for S in range(2014, 2026):
    d = P[P.season == S].copy(); d["min_adj"] = d.min_model * d.bin.map(mults(P[P.season != S])); parts.append(d)
P = pd.concat(parts, ignore_index=True)

def scale_up(m):
    for _ in range(10):
        room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
    return m
def alloc(g, rule, gam=1.0):
    m = g.min_adj.clip(upper=3000).copy()
    if m.sum() <= 19680: return scale_up(m)
    if rule == "bottom":
        ex = m.sum() - 19680
        for i in m.sort_values().index:
            c = min(ex, m[i]); m[i] -= c; ex -= c
            if ex <= 0: break
        return m
    base = m.values; lo, hi = 0.0, 1.0
    f = lambda c: np.minimum(base, c * base ** gam).sum()
    while f(hi) < 19680: hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < 19680: lo = mid
        else: hi = mid
    return pd.Series(np.minimum(base, hi * base ** gam), index=m.index)

A = pd.read_parquet(f"{D}/preseason_backtest.parquet")
A = A[A.season != 2019]
def score(rule, gam=1.0):
    out = []; perr = []
    for S in range(2014, 2026):
        d = P[P.season == S].copy()
        d["mp"] = d.groupby("team", group_keys=False).apply(lambda g: alloc(g, rule, gam))
        T = pd.read_parquet(f"{D}/pre_T_{S}.parquet").set_index("team"); pace = (T.margin / T.net).mean() * 100
        net = (d.rating * d.mp / 3936).groupby(d.team).sum(); net -= net.mean()
        out.append(pd.DataFrame({"season": S, "team": net.index, "wp": norm.cdf(net.values * pace / 100 / 12.8)}))
        if S != 2019: perr.append(d[["act_min", "mp", "team", "season"]])
    M = pd.concat(out).merge(A[["season", "team", "wpct"]], on=["season", "team"])
    e = (M.wp - M.wpct) * 82; pe = pd.concat(perr)
    crowd = pe.groupby(["season", "team"]).mp.transform("size")
    return dict(rmse=np.sqrt((e ** 2).mean()), h1=np.sqrt((e[M.season <= 2018] ** 2).mean()), h2=np.sqrt((e[M.season >= 2020] ** 2).mean()),
                pl_rmse=np.sqrt(((pe.mp - pe.act_min) ** 2).mean()), err=e.values)
res = {}
for name, r, g in [("bottom", "bottom", 1), ("prop", "pow", 1.0), ("pow1.5", "pow", 1.5), ("pow2", "pow", 2.0), ("pow3", "pow", 3.0), ("pow5", "pow", 5.0)]:
    res[name] = score(r, g)
    x = res[name]; print(f"{name:7s} team RMSE {x['rmse']:.3f} (2014-18 {x['h1']:.3f}, 2020-25 {x['h2']:.3f})  player-min RMSE {x['pl_rmse']:.0f}")
b = res["bottom"]["err"]
for k, v in res.items():
    if k == "bottom": continue
    dd = v["err"] ** 2 - b ** 2; print(f"  {k} vs bottom: d MSE {dd.mean():+.3f}, t {dd.mean() / (dd.std() / np.sqrt(len(dd))):+.2f}")
crowded = P.groupby(["season", "team"]).min_adj.sum()
print("share of team-seasons over 19,680:", round((crowded > 19680).mean(), 3), "| 2026 MIL-like excess median", round((crowded[crowded > 19680] - 19680).median()))
# who gets zeroed by the bottom rule, and how much did they actually play?
z = []
for S in range(2014, 2026):
    if S == 2019: continue
    d = P[P.season == S].copy(); d["mp"] = d.groupby("team", group_keys=False).apply(lambda g: alloc(g, "bottom"))
    d["pp"] = d.groupby("team", group_keys=False).apply(lambda g: alloc(g, "pow", 1.0)); z.append(d)
z = pd.concat(z); zz = z[(z.mp < 1) & (z.min_adj > 600)]
print(f"zeroed despite 600+ wanted: {len(zz)} player-seasons; actual min mean {zz.act_min.mean():.0f}, median {zz.act_min.median():.0f}; "
      f"prop rule gave them {zz.pp.mean():.0f}; rating mean {zz.rating.mean():+.2f} vs kept rotation {z[z.mp>600].rating.mean():+.2f}")
print("their actual minutes, share >=500:", round((zz.act_min >= 500).mean(), 2))
# hybrid: every player keeps at least f x his projection; the rest of the excess is trimmed from the bottom
def alloc_floor(g, f):
    m = g.min_adj.clip(upper=3000).copy()
    if m.sum() <= 19680: return scale_up(m)
    fl = f * m; ex = m.sum() - 19680; room = m - fl
    if room.sum() < ex: return m * 19680 / m.sum()
    for i in m.sort_values().index:
        c = min(ex, room[i]); m[i] -= c; ex -= c
        if ex <= 0: break
    return m
for f in (0.25, 0.4, 0.5):
    alloc_b = alloc
    alloc = lambda g, rule, gam=1.0, f=f: alloc_floor(g, f)
    x = score("floor"); dd = x["err"] ** 2 - b ** 2
    print(f"floor {f}: team RMSE {x['rmse']:.3f} (h {x['h1']:.3f}/{x['h2']:.3f}) player-min {x['pl_rmse']:.0f}  vs bottom t {dd.mean() / (dd.std() / np.sqrt(len(dd))):+.2f}")
    alloc = alloc_b
