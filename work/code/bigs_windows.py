"""Bigs investigation (B4/B2), step 1: per-window player table for windows N (seasons N-2..N, decay .3/.6/1).
Current backtest weights (weights_bt x .5, dreb already halved). Saves data/bigs_win_{N}.parquet:
 player, res_o, res_d, f:<feature> (W-weighted window values), components, dposs (decayed poss on court, off+def)/2,
 thin = LAM/(dposs+LAM) (ridge shrink share), pos (window min-weighted pos_est), age_N,
 ctx = decayed avg team-lineup thinness when the player is OFF (his games) minus his 4 teammates' thinness when ON.
Usage: python3 bigs_windows.py N0 N1
"""
import sys, os, json, numpy as np, pandas as pd, impact as I
LAM = 2000.0
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
PROD = os.environ.get("PROD") == "1"
if PROD:
    W = json.load(open(f"{I.D}/weights_prod.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
pos = pd.read_parquet(f"{I.D}/positions.parquet")[["season", "player", "min", "pos_est"]]
feat = I.feats()[["season", "player", "age"]]


def stint_sides(seasons, dec):
    out = []
    for s in seasons:
        st = pd.read_parquet(f"{I.D}/stints_{s}.parquet")
        m = st.margin_start.abs()
        st = st[~(((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15)))]
        st = st[(st.lineupA.str.count("-") == 4) & (st.lineupB.str.count("-") == 4)]
        pa = st.fg2aA + st.fg3aA - st.orebA + st.tovA + .44 * st.ftaA
        pb = st.fg2aB + st.fg3aB - st.orebB + st.tovB + .44 * st.ftaB
        pt = ((pa + pb) / 2).to_numpy() * dec[s]
        for me in "AB":
            out.append(pd.DataFrame({"game": st.game.to_numpy(), "team": st[f"team{me}"].to_numpy(), "w": pt,
                                     "ids": list(I.lineup_ids(st[f"lineup{me}"]))}))
    d = pd.concat(out, ignore_index=True)
    return d[d.w > 0].reset_index(drop=True)


for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{I.D}/bigs_win_{N}{'_prod' if PROD else ''}.parquet"
    if os.path.exists(fn): continue
    seasons = [N - 2, N - 1, N]; dec = dict(zip(seasons, (.3, .6, 1.)))
    rows = I.load_rows(seasons)
    o = I.fit_window(N, Wo, Wd, lam=LAM, rows=rows)
    # sample size (decayed possessions on court, average of off/def ~ same)
    S = stint_sides(seasons, dec)
    ids = np.vstack(S.ids.values)
    long = pd.DataFrame({"sid": np.repeat(np.arange(len(S)), 5), "player": ids.ravel(),
                         "game": np.repeat(S.game.values, 5), "team": np.repeat(S.team.values, 5),
                         "w": np.repeat(S.w.values, 5)})
    dposs = long.groupby("player").w.sum()
    thin = LAM / (dposs + LAM)
    long["thin"] = long.player.map(thin).to_numpy()
    S["S"] = long.groupby("sid").thin.sum().to_numpy()
    long["S"] = np.repeat(S.S.values, 5)
    tg = S.assign(wS=S.w * S.S).groupby(["game", "team"])[["w", "wS"]].sum()
    pgm = long.assign(wS4=long.w * (long.S - long.thin)).groupby(["player", "game", "team"])[["w", "wS4"]].sum()
    tt = tg.reindex(pd.MultiIndex.from_arrays([pgm.index.get_level_values(1), pgm.index.get_level_values(2)]))
    pgm["offw"] = tt.w.to_numpy() - pgm.w.to_numpy()
    pgm["offwS"] = tt.wS.to_numpy() - (long.assign(wS=long.w * long.S).groupby(["player", "game", "team"]).wS.sum().to_numpy())
    agg = pgm.groupby(level=0)[["w", "wS4", "offw", "offwS"]].sum()
    ctx = agg.offwS / agg.offw.where(agg.offw > 0) - agg.wS4 / agg.w
    # window position (minutes x decay weighted) and age in N
    pw = pos[pos.season.isin(seasons)].copy(); pw["wt"] = pw["min"] * pw.season.map(dec)
    pw = pw.assign(x=pw.pos_est * pw.wt).groupby("player")[["x", "wt"]].sum()
    o["dposs"] = o.player.map(dposs).fillna(0)
    o["thin"] = o.player.map(thin).fillna(1.0)
    o["ctx"] = o.player.map(ctx)
    o["off_share"] = o.player.map(agg.offw / (agg.offw + agg.w))
    o["pos"] = o.player.map(pw.x / pw.wt)
    o["age_N"] = o.player.map(feat[feat.season == N].set_index("player").age)
    o.to_parquet(fn, index=False)
    print("done", N, len(o), "ctx sd", round(o.ctx.std(), 3), flush=True)
