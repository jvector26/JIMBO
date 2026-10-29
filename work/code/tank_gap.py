"""IND-type check: teammates of a star who missed season N (star returns to same team in N+1). Do their N+1 ratings beat
the window (i.e. was the star-less season misleading for them)? Also which team-seasons qualify."""
import numpy as np, pandas as pd, statsmodels.api as sm
D = "/home/claude/data"
A = pd.read_parquet(f"{D}/tank_player.parquet")
R = pd.read_parquet(f"{D}/injury_return.parquet"); R = R[R.missed.astype(bool)]
for thr in [2.0, 1.0]:
    rr = R[R.rating >= thr][["season", "team", "rating"]]
    # A.season = N (last window season); star returns in N+1 to team; teammate was on that team at end of N and stays
    key = set(zip(rr.season - 1, rr.team))
    A["gapmate"] = [float((s, t) in key and tn == t) for s, t, tn in zip(A.season, A.team_lastN, A.team_next)]
    d = A
    X = pd.concat([d[["win", "dbox", "dlin", "gapmate", "tank"]], pd.get_dummies(d.bin, drop_first=True, dtype=float),
                   pd.get_dummies(d.season, prefix="s", drop_first=True, dtype=float)], axis=1)
    for y in ["act", "act_box", "act_lin"]:
        m = sm.WLS(d[y], sm.add_constant(X), weights=d.min_next).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.clu)[0]})
        print(f"star rating>={thr}: n teammates {int(d.gapmate.sum())} ({len(key)} team-seasons)  {y}: gapmate {m.params['gapmate']:+.2f} (t {m.tvalues['gapmate']:+.1f})")
    if thr == 2.0:
        g = d[d.gapmate > 0]
        print(g.groupby(["season", "team_lastN"]).apply(lambda x: pd.Series({"n": len(x), "act-win": np.average(x.act - x.win, weights=x.min_next)}), include_groups=False).round(2))
        print("all players act-win mean:", round(np.average(d.act - d.win, weights=d.min_next), 2))
