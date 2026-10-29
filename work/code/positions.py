"""Player positions per season (all 1996-2025 player-seasons).
1) 'size score' per player-season: regression of listed position (bio.json POS, coarse G/G-F/F-G/F/F-C/C-F/C) on
   height, weight and season-centered per-36 rates (ast, oreb, dreb, blk, stl, 3PA, FTA, pf) + rim share of FGA.
2) lineup rank (Basketball-Reference style): in every stint the 5 players on each side are ranked by size score
   (1 = smallest = PG ... 5 = C); a player's position = time-weighted average rank; shares at each rank also kept.
Outputs data/positions.parquet: season, player, name, pos_listed, height_in, size_score, pos_est (1-5), pct_1..pct_5,
pos5 (PG/SG/SF/PF/C), pos3 (G/W/B)."""
import numpy as np, pandas as pd
D = "/home/claude/data"
CUTS5, CUTS3 = [0, 1.5, 2.5, 3.45, 4.4, 9], [0, 2.5, 3.9, 9]  # PG/SG/SF/PF/C and guard/wing/big cuts on the final scale
MAP = {"G": 1.5, "G-F": 2.5, "F-G": 3.0, "F": 3.75, "F-C": 4.25, "C-F": 4.5, "C": 5.0}
RATES = ["ast_36", "oreb_36", "dreb_36", "blk_36", "stl_36", "fg3a_36", "fta_36"]

ps = pd.read_parquet(f"{D}/player_seasons.parquet")
ps["pf_36"] = ps.pf / ps["min"].clip(lower=1) * 36
ps["rim_share"] = (ps.rim_fga / ps.fga.clip(lower=1))
cols = RATES + ["pf_36", "rim_share"]
X = ps[cols].copy()
for c in cols:  # season-centred, clipped (small samples are noisy)
    X[c] = (X[c] - X.groupby(ps.season)[c].transform("median")).clip(*X[c].quantile([.01, .99]))
hmean, wmean = ps.height_in.mean(), ps.weight_lb.mean()
X["h"] = ps.height_in - hmean; X["w"] = ps.weight_lb - wmean
y = ps.pos.map(MAP)
wt = ps["min"].clip(upper=2500)
def fit(cols_):
    m = y.notna() & X[cols_].notna().all(axis=1) & (ps["min"] >= 200)
    A = np.c_[np.ones(m.sum()), X.loc[m, cols_]]; W = np.sqrt(wt[m].values)
    b = np.linalg.lstsq(A * W[:, None], y[m].values * W, rcond=None)[0]
    r = y[m].values - A @ b
    print(f"{'full' if 'h' in cols_ else 'stats-only'} model: R2 {1 - (wt[m]*r**2).sum()/(wt[m]*(y[m]-np.average(y[m],weights=wt[m]))**2).sum():.3f}")
    return b
b_full = fit(cols + ["h"]); b_stat = fit(cols)  # weight dropped 2026-09-21: bio weights have errors (Ajay Mitchell 235)
full = np.c_[np.ones(len(X)), X[cols + ["h"]].fillna(0)] @ b_full
stat = np.c_[np.ones(len(X)), X[cols].fillna(0)] @ b_stat
ps["size_score"] = np.where(ps.height_in.notna(), full, stat)
# shrink tiny samples toward height-only / listed
score = ps.set_index(["season", "player"]).size_score

rows = []
for s in range(1996, 2026):
    st = pd.read_parquet(f"{D}/stints_{s}.parquet", columns=["lineupA", "lineupB", "t_start", "t_end"])
    dur = (st.t_end - st.t_start).to_numpy()
    sc = score.loc[s]
    for side in "AB":
        L = np.array([list(map(int, x.split("-"))) if x.count("-") == 4 else [0] * 5 for x in st[f"lineup{side}"]])
        ok = (L > 0).all(axis=1)
        L, d = L[ok], dur[ok]
        v = sc.reindex(L.ravel()).to_numpy().reshape(L.shape)
        v = np.where(np.isnan(v), 3.0, v)
        rank = v.argsort(axis=1).argsort(axis=1) + 1
        rows.append(pd.DataFrame({"season": s, "player": L.ravel(), "rank": rank.ravel(), "sec": np.repeat(d, 5)}))
r = pd.concat(rows)
sh = r.pivot_table(index=["season", "player"], columns="rank", values="sec", aggfunc="sum", fill_value=0)
sh = sh.div(sh.sum(axis=1), axis=0); sh.columns = [f"pct_{c}" for c in sh.columns]
sh["pos_est"] = sum(sh[f"pct_{k}"] * k for k in range(1, 6))
out = ps[["season", "player", "name", "min", "pos", "height_in", "weight_lb", "size_score"]].rename(columns={"pos": "pos_listed"}) \
    .merge(sh.reset_index(), on=["season", "player"], how="left")
# players with no stints (rare): map size score onto the 1-5 scale
nos = out.pos_est.isna(); out.loc[nos, "pos_est"] = out.loc[nos, "size_score"].clip(1, 5)
# 2026-09-21 v2: pure lineup rank was fragile (near-equal guards: SGA ranked PF in OKC's small lineups). Final position =
# mean of lineup rank, size score and listed position (when known), then a ball-handling tilt for guards/wings
# (season-centred assists per 36; fades out above 3.2 so passing bigs like Jokic stay bigs).
MAPL = {"G": 1.5, "G-F": 2.5, "F-G": 3.0, "F": 3.75, "F-C": 4.25, "C-F": 4.5, "C": 5.0}
out["pos_rank"] = out.pos_est
mix = pd.concat([out.pos_rank, out.size_score.clip(1, 5), out.pos_listed.map(MAPL)], axis=1).mean(axis=1)
ast_c = (ps.ast_36 - ps.groupby("season").ast_36.transform("median")).clip(-3, 6).fillna(0).to_numpy()
tilt = 0.12 * ast_c * np.clip((3.2 - mix) / 1.0, 0, 1)
out["pos_est"] = (mix - tilt).clip(1, 5)
out["pos5"] = pd.cut(out.pos_est, CUTS5, labels=["PG", "SG", "SF", "PF", "C"]).astype(str)
out["pos3"] = pd.cut(out.pos_est, CUTS3, labels=["G", "W", "B"]).astype(str)
out.to_parquet(f"{D}/positions.parquet", index=False)
m = out["min"] >= 500
print("pos5 minute shares:", (out[m].groupby("pos5")["min"].sum() / out[m]["min"].sum()).round(3).to_dict())
print("corr pos_est vs listed:", round(np.corrcoef(out[m & out.pos_listed.notna()].pos_est, out[m & out.pos_listed.notna()].pos_listed.map(MAP))[0, 1], 3))
