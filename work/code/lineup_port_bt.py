"""Game-level test: shrink each player's lineup residual by window sample size and team change.
m = (1 - a*moved) * f(P),  f(P) = P/(P+c) * (Pref+c)/Pref   (Pref = 8000 decayed window possessions => f=1 there)
Uses saved window residuals (bt_split_ratings) + actual game minutes (game_minutes) for seasons 2015-2025."""
import sys, numpy as np, pandas as pd
D = "/home/claude/data"; DEC = (0.3, 0.6, 1.0); PREF = 8000
R = pd.read_parquet(f"{D}/bt_split_ratings.parquet")
gm = pd.read_parquet(f"{D}/game_minutes.parquet")[["season", "game", "player", "team", "min"]]
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
gid = pd.read_parquet(f"{D}/games_all.parquet")
B = pd.read_parquet(f"{D}/backtest_components.parquet")
F = [c for c in B.columns if c.startswith("f:")]; B["box"] = B[F].sum(axis=1) + B.unrated
# pace factor per season (bt_split res_o / raw res_o diff ratio, reproduced r=.9999)
pace = {2015: .8493, 2016: .8629, 2017: .8677, 2018: .8803, 2019: .892, 2020: .9018, 2021: .9087, 2022: .8965,
        2023: .8968, 2024: .8964, 2025: .8969}
# team id -> abbr mapping for player_seasons 'teams' (abbr) vs game team ids
id2abbr = pd.read_parquet(f"{D}/games_2025.parquet").drop_duplicates("home_team").set_index("home_team").htm.to_dict()
rows = []
for N in range(2014, 2025):
    r = R[R.season == N].set_index("player")
    h = ps[ps.season.between(N - 2, N)]
    P = sum(DEC[i] * h[h.season == N - 2 + i].groupby("player").on_poss.sum() for i in range(3))
    P = P.reindex(r.index).fillna(0) if False else (h.assign(w=h.season.map({N - 2: DEC[0], N - 1: DEC[1], N: DEC[2]}))
                                                    .eval("wp = w * on_poss").groupby("player").wp.sum())
    last = h.sort_values("season").groupby("player").teams.last().str.split("/").str[-1]
    g = gm[gm.season == N + 1].copy()
    g = g[g.player.isin(r.index)]
    g["res_o"] = g.player.map(r.res_o); g["res_d"] = g.player.map(r.res_d)
    g["P"] = g.player.map(P).fillna(0); g["moved"] = (g.team.map(id2abbr) != g.player.map(last)).astype(float)
    g["N"] = N; rows.append(g)
G = pd.concat(rows, ignore_index=True)
G["w48"] = G["min"] / 48 * G.season.map(pace)
gi = gid[["game", "home_team", "away_team"]]
G = G.merge(gi, on="game"); G["sgn"] = np.where(G.team == G.home_team, 1.0, -1.0)
B = B.set_index("game"); y = B.margin.values; seas = B.season.values

def diffs(m):
    v = G.sgn * G.w48 * m
    return (pd.DataFrame({"o": v * G.res_o, "d": v * G.res_d, "game": G.game}).groupby("game")[["o", "d"]].sum()
            .reindex(B.index).fillna(0))

def loso(X):
    X = np.c_[np.ones(len(y)), X]; e = []
    for s in np.unique(seas):
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
    return np.sqrt(np.concatenate(e).mean()), np.array([np.sqrt((ee).mean()) for ee in e])

def fixed(dd, wo=.6, wd=.8):
    return loso(np.c_[B.box + wo * dd.o + wd * dd.d])

if __name__ == "__main__":
    base = diffs(np.ones(len(G)))
    print("check vs bt_split res_o corr", np.corrcoef(base.o, pd.read_parquet(f"{D}/bt_split.parquet").set_index("game").reindex(B.index).res_o)[0, 1].round(5))
    b0, s0 = fixed(base); bf, sf = loso(np.c_[B.box, base.o, base.d])
    print(f"baseline fixed .6/.8 {b0:.4f}   free {bf:.4f}")
    res = []
    for c in [0, 1000, 2000, 4000, 8000]:
        f = G.P / (G.P + c) * (PREF + c) / PREF if c else np.ones(len(G))
        for a in [0, .2, .35, .5]:
            m = (1 - a * G.moved) * f
            dd = diffs(m); lf, sfx = fixed(dd); lr, sr = loso(np.c_[B.box, dd.o, dd.d])
            res.append(dict(c=c, a=a, fixed=lf, free=lr, d_fixed=lf - b0, wins=int((sfx < s0).sum()), mean_m=np.average(m, weights=G["min"])))
    print(pd.DataFrame(res).round(4).to_string())
