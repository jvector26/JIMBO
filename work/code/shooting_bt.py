"""Game-level check: replace the 3PT efficiency rating component with a career+FT% talent estimate."""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet")
W = json.load(open(f"{D}/weights_bt.json")); W3 = W["Wo"]["o_3eff"] * 0.5
ps = pd.read_parquet(f"{D}/player_seasons.parquet"); f = pd.read_parquet(f"{D}/features.parquet")
p = ps.groupby(["player", "season"])[["fg3m", "fg3a", "ftm", "fta"]].sum().reset_index().sort_values(["player", "season"])
lg = p.groupby("season")[["fg3m", "fg3a"]].sum(); lg3 = lg.fg3m / lg.fg3a
for c in ["fg3m", "fg3a", "ftm", "fta"]: p["c_" + c] = p.groupby("player")[c].cumsum()
p = p.merge(f[["season", "player", "poss_off", "o_3eff"]], on=["season", "player"], how="left")
p["lg3"] = p.season.map(lg3)
gm = pd.read_parquet(f"{D}/games_all.parquet")
DEC = {0: 1.0, 1: 0.6, 2: 0.3}
out = []
for S in sorted(B.season.unique()):
    N = S - 1; w = p[p.season.between(N-2, N)].copy(); w["dw"] = (N - w.season).map(DEC) * w.poss_off.fillna(0)
    agg = w.assign(v=w.fg3a / (w.poss_off/100).clip(lower=0.01) * w.dw, e=w.o_3eff.fillna(0) * w.dw,
                   s=((w.fg3m + 250*w.lg3)/(w.fg3a + 250) - w.lg3) * w.dw).groupby("player")[["v", "e", "s", "dw"]].sum()
    vol = agg.v / agg.dw.clip(lower=1); poss = w.groupby("player").poss_off.sum()
    last = p[p.season <= N].groupby("player").last()                        # career-to-date through N
    car = (last.c_fg3m + 400*lg3[N]) / (last.c_fg3a + 400) - lg3[N]
    ft = (last.c_ftm + 0.75*50) / (last.c_fta + 50) - 0.75
    rel_new = 0.003 + 0.868*car + 0.011*ft + 0.183*ft*400/(last.c_fg3a + 400)
    rel_car = car
    shr = poss / (poss + 300)
    comp = pd.DataFrame({"old": agg.e / agg.dw.clip(lower=1),                                  # current: window-avg feature
                         "rebuilt": vol * 3 * agg.s / agg.dw.clip(lower=1) * shr,             # same idea, my rebuild
                         "career": vol * 3 * rel_car.reindex(vol.index) * shr,
                         "car_ft": vol * 3 * rel_new.reindex(vol.index) * shr}).fillna(0) * W3
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet", columns=["game", "player", "team", "min"])
    pg = pg[pg["min"] > 0]; pace = 100
    g = gm[gm.season == S][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    o = pd.DataFrame({"game": g.game.values})
    for c in comp.columns:
        t = (pg.player.map(comp[c]).fillna(0) * pg["min"]/48).groupby([pg.game, pg.team]).sum()
        o["x3_" + c] = t.reindex(hk).fillna(0).to_numpy() - t.reindex(ak).fillna(0).to_numpy()
    out.append(o)
A = B.merge(pd.concat(out), on="game")
comp_cols = [c for c in B.columns if c.startswith("f:") and c != "f:o_3eff"]
A["rest"] = A[comp_cols].sum(axis=1) + 0.6*A.lineup_impact + A.unrated
print("corr of my rebuild with production f:o_3eff:", np.corrcoef(A["f:o_3eff"], A.x3_old)[0, 1].round(3), np.corrcoef(A["f:o_3eff"], A.x3_rebuilt)[0, 1].round(3))
def loso(cols):
    X = np.c_[np.ones(len(A)), A[cols]]; y = A.margin.to_numpy(); pr = np.empty(len(A))
    for s in A.season.unique():
        tr = (A.season != s).to_numpy(); b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; pr[~tr] = X[~tr] @ b
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.sqrt(np.mean((y - pr)**2)), b
for lab, cols in [("production 3PT", ["rest", "f:o_3eff"]), ("no 3PT eff at all", ["rest"]), ("rebuilt (same method)", ["rest", "x3_rebuilt"]),
                  ("career k400", ["rest", "x3_career"]), ("career + FT%", ["rest", "x3_car_ft"])]:
    r, b = loso(cols); print(f"{lab:24s} LOSO {r:.4f}  coef on 3PT part {b[-1]/b[1] if len(b) > 2 else 0:.2f}x rest")
A.to_parquet(f"{D}/shooting_bt.parquet", index=False)
