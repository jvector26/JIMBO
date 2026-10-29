"""Does the raw on-court opponent shot profile (d_oppxq, also d_opprim) carry over when a player changes teams?
Split each game's f:<feat> diff into a stayer part and a mover part, by each player's share of his season-N minutes
played for his season-N+1 team. Then the standard bar: game LOSO (season-clustered t, halves 2015-19 / 2020-25).
python3 code/mover_context.py"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "code"); import impact as I
W = json.load(open(f"{I.D}/weights_bt.json")); Wd = {k: v * .5 for k, v in W["Wd"].items()}
FE = ["d_oppxq", "d_opprim"]; DEC = (.1, .4, 1.)
f = I.feats(); gm = pd.read_parquet(f"{I.D}/games_all.parquet")
mins = pd.read_parquet(f"{I.D}/game_minutes.parquet")[["season", "game", "player", "team", "min"]]
B = pd.read_parquet(f"{I.D}/backtest_components.parquet").merge(pd.read_parquet(f"{I.D}/bt_split.parquet"),
                                                                 on=["game", "home_team", "away_team"])
rows = []
for N in range(2014, 2025):
    fw = f[f.season.between(N - 2, N)].copy(); d = fw.season.map(dict(zip([N - 2, N - 1, N], DEC)))
    wt = fw.poss_def * d
    V = pd.DataFrame({x: (fw[x].fillna(0) * Wd[x] * wt).groupby(fw.player).sum() / wt.groupby(fw.player).sum() for x in FE}).fillna(0)
    # share of season-N minutes with each team
    mN = mins[mins.season == N].groupby(["player", "team"])["min"].sum()
    shN = (mN / mN.groupby(level=0).transform("sum"))
    pg = mins[mins.season == N + 1].copy()
    key = pd.MultiIndex.from_arrays([pg.player, pg.team])
    pg["share"] = shN.reindex(key).fillna(0).to_numpy()
    pg["hadN"] = pg.player.isin(mN.index.get_level_values(0)).to_numpy()
    # players with no season-N minutes (missed season / rookie): count as stayers (no context to discount)
    pg.loc[~pg.hadN, "share"] = 1.0
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    out = pd.DataFrame({"game": g.game.values})
    for x in FE:
        v = pg.player.map(V[x]).fillna(0).to_numpy() * pg["min"].to_numpy() / 48
        for part, wgt in [("all", 1.0), ("mov", 1 - pg.share.to_numpy())]:
            t = pd.Series(v * wgt).groupby([pg.game.values, pg.team.values]).sum()
            out[f"{x}_{part}"] = t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()
    # mover minutes share (for describing the sample)
    t = pd.Series((1 - pg.share.to_numpy()) * pg["min"].to_numpy()).groupby([pg.game.values, pg.team.values]).sum()
    out["mov_min"] = (t.reindex(hk).to_numpy() + t.reindex(ak).to_numpy()) / 480
    rows.append(out)
R = pd.concat(rows, ignore_index=True)
B = B.merge(R, on="game", how="left")
y = B.margin.to_numpy(); seas = B.season.to_numpy()
# reconstruction check: f:<x> = pace/100 * x_all (one scale per season)
for x in FE:
    r2 = []
    for s in np.unique(seas):
        m = seas == s; a = B.loc[m, f"{x}_all"].to_numpy(); b = B.loc[m, f"f:{x}"].to_numpy()
        k = (a @ b) / (a @ a); r2.append(1 - ((b - k * a) ** 2).sum() / ((b - b.mean()) ** 2).sum())
    print(f"reconstruction {x}: min R2 over seasons {min(r2):.4f}")
    # rescale mover part to the f: units per season
    sc = B.groupby("season").apply(lambda d: (d[f"{x}_all"] @ d[f"f:{x}"]) / (d[f"{x}_all"] @ d[f"{x}_all"]))
    B[f"M_{x}"] = B[f"{x}_mov"] * B.season.map(sc)
print(f"average share of minutes played by players on a new team: {B.mov_min.mean():.3f}")

F = [c for c in B.columns if c.startswith("f:")]
base = (B[F].sum(axis=1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy()

def loso(X):
    X = np.c_[np.ones(len(y)), X]; e = []
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    return np.sqrt(np.concatenate(e).mean())

def coefs(X, m):
    X = np.c_[np.ones(m.sum()), X[m]]; yy = y[m]; ss = seas[m]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss))
    return b, b / np.sqrt(np.diag(XtXi @ meat @ XtXi))

H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
print(f"\nbase LOSO {loso(base[:, None]):.4f}")
for x in FE + ["both"]:
    M = B[[f"M_{z}" for z in FE]].sum(axis=1).to_numpy() if x == "both" else B[f"M_{x}"].to_numpy()
    X = np.c_[base, M]
    b, t = coefs(X, ALL); b1, t1 = coefs(X, H1); b2, t2 = coefs(X, H2)
    # coefficient on M relative to base: -1 => movers' value worthless, 0 => carries over fully
    print(f"{x:9s} free term LOSO {loso(X):.4f} | M coef/base coef {b[2]/b[1]:+.2f} (t {t[2]:+.1f}); "
          f"halves {b1[2]/b1[1]:+.2f} (t {t1[2]:+.1f}) / {b2[2]/b2[1]:+.2f} (t {t2[2]:+.1f})")
    for k in [.25, .5, .75, 1.0]:
        print(f"           fixed: drop {int(k*100)}% of movers' value  LOSO {loso((base - k * M)[:, None]):.4f}")
B[["game", "season", "M_d_oppxq", "M_d_opprim", "mov_min"]].to_parquet(f"{I.D}/mover_context.parquet", index=False)
