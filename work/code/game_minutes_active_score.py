"""Score production-like tonight-minutes rules on game_minutes_active.parquet (see that script for the absence rule)."""
import numpy as np, pandas as pd
D = "/home/claude/data"
A = pd.read_parquet(f"{D}/game_minutes_active.parquet")
UNR = -4.313 * 1.0501; A["r"] = A.rating.fillna(UNR)
pace = {S: (lambda T: (T.margin / T.net).mean() * 100)(pd.read_parquet(f"{D}/pre_T_{S}.parquet")) for S in range(2014, 2026)}
gm = pd.concat([pd.read_parquet(f"{D}/games_{S}.parquet") for S in range(2014, 2026)])[["game", "home_team", "away_team", "home_pts", "away_pts", "season"]]
def cond(x, k=2, a=.5):
    w = x.np / (x.np + k); pre = x.pre_wp.fillna(10.0)
    return w * (a * x.l5_p + (1 - a) * x.std_p).fillna(pre) + (1 - w) * pre
A["c"] = cond(A)
A["status"] = np.where(A.played, "played", np.where((A.run >= 3) | (A.c >= 20), "out", "dnp"))
act = A[A.status != "out"].copy()
print("rows", len(A), act.status.value_counts().to_dict(), "| out", (A.status == "out").sum())
def cur(x, k=5, rw=.5):
    w = x.n_tm / (x.n_tm + k); pre = x.pre_tg.fillna(6.0)
    return w * (rw * x.l10_tg + (1 - rw) * x.std_tg).fillna(pre) + (1 - w) * pre
def trim(m, keys, total=240.0):
    d = pd.DataFrame({"m": m.values, "g": keys[0].values, "t": keys[1].values}, index=m.index).sort_values(["g", "t", "m"])
    out = d.m.copy()
    tot = d.groupby(["g", "t"]).m.transform("sum")
    # bottom-trim: remove excess from the smallest first (cumulative from the bottom)
    cum = d.groupby(["g", "t"]).m.cumsum(); ex = (tot - total).clip(lower=0)
    cut = np.clip(ex - (cum - d.m), 0, d.m)
    out = d.m - cut
    sc = np.where(tot < total, total / tot, 1.0); out = out * sc
    return out.reindex(m.index)
def evaluate(m, lab, how="prop"):
    x = act[["game", "team", "min", "r", "season", "played", "pre_wp", "c"]].copy(); x["m"] = m.clip(lower=0).values
    if how == "prop": x["m"] = x.m * 240 / x.groupby(["game", "team"]).m.transform("sum")
    elif how == "trim": x["m"] = trim(x.m, (x.game, x.team))
    e = x.m - x["min"]
    s = (x.r * x.m).groupby([x.game, x.team]).sum() / 48
    g = gm.copy()
    hs = s.reindex(pd.MultiIndex.from_arrays([g.game, g.home_team])).values; as_ = s.reindex(pd.MultiIndex.from_arrays([g.game, g.away_team])).values
    g["pred"] = (hs - as_) * g.season.map(pace) / 100 + 2.6; g["marg"] = g.home_pts - g.away_pts; g = g.dropna(subset=["pred"])
    top = x.played & (x.pre_wp >= 32)
    ge = (g.pred - g.marg) ** 2
    print(f"{lab:30s} min RMSE {np.sqrt((e**2).mean()):.2f} | stars (pre 32+) when playing: pred {x.m[top].mean():.1f} act {x['min'][top].mean():.1f}"
          f" | top-3 per team pred {x.groupby(['game','team']).m.apply(lambda v: v.nlargest(3).mean()).mean():.1f}"
          f" | game RMSE {np.sqrt(ge.mean()):.4f} ({np.sqrt(ge[g.season<=2019].mean()):.4f}/{np.sqrt(ge[g.season>=2020].mean()):.4f})")
    return ge.groupby(g.season).mean()
if __name__ == "__main__":
    o = evaluate(act["min"], "oracle actual")
    b = evaluate(cur(act), "current rule (prop)")
    # top-3 actual for reference
    print("actual top-3 per team per game:", round(act.groupby(["game", "team"])["min"].apply(lambda v: v.nlargest(3).mean()).mean(), 1))
    for lab, m, how in [("conditional, prop", act.c, "prop"), ("conditional, bottom-trim", act.c, "trim")]:
        d = evaluate(m, lab, how) - b; print(f"{'':30s} vs current: season-paired t {d.mean()/d.std()*np.sqrt(len(d)):+.2f}")
    for h, q0 in ((5, .9), (10, .8)):
        q = ((act.np + q0 * h) / (act.groupby(["season", "team", "player"]).cumcount() + h)).clip(0, 1)  # played share among active team games
        d = evaluate(act.c * q, f"conditional x P(play) h{h} q0 {q0}", "prop") - b; print(f"{'':30s} vs current: season-paired t {d.mean()/d.std()*np.sqrt(len(d)):+.2f}")
