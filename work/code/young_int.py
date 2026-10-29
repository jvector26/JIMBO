"""Does the young-player age bump shrink as the player's window rating rises?
Variant: vets aged <= AGEMAX get rating += SLOPE * k * (win - c), win = pre-age window rating (pre-slope),
c = minutes-weighted mean win of those young vets in that season (so the average young bump is unchanged).
Game level: base (box + .6o + .8d, backtest_components/bt_split) + age curve + interaction term, LOSO.
Team level: rewrite pre_P/pre_T with TAG prefix -> score with age_eval.py TAG."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
D = "/home/claude/data"
SLOPE = float(pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]["slope"])
aging = pd.read_csv(f"{D}/aging.csv").set_index("bin").adj
BINS = [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]
def age_bin(a): return next((l for lo, hi, l in BINS if lo <= a <= hi), None)
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
def load_P(S):
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
    last = ps[ps.season < S].sort_values("season").groupby("player").last()
    P["age"] = P.player.map(last.age) + (S - P.player.map(last.season))
    P["ag"] = [aging.get(age_bin(a), 0.0) if pd.notna(a) else 0.0 for a in P.age]
    P["win"] = np.where(P.kind == "vet", (P.rating - P.returner) / SLOPE - P.ag, np.nan)
    return P
PP = {S: load_P(S) for S in range(2014, 2026)}
# sanity: young vets' window ratings
A = pd.concat(PP.values()); yv = A[(A.kind == "vet") & (A.age < 22)]
print("young vets:", len(yv), "win mean %.2f sd %.2f" % (np.average(yv.win, weights=yv.min_proj + 1), yv.win.std()))

def delta(P, k, agemax):
    m = (P.kind == "vet") & (P.age < agemax + 1)
    c = np.average(P.win[m], weights=P.min_proj[m] + 1) if m.any() else 0
    return np.where(m, SLOPE * k * (P.win - c), 0.0), m, c

if __name__ == "__main__" and sys.argv[1:2] == ["game"]:
    B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"])
    F = [c for c in B.columns if c.startswith("f:")]
    AD = pd.read_parquet(f"{D}/age_design2.parquet").set_index("game").reindex(B.game)
    agecurve = sum(AD[f"A{a}"].to_numpy() * aging.get(age_bin(a + .5), 0) for a in range(18, 45))
    base = (B[F].sum(1) + B.unrated + .6 * B.res_o + .8 * B.res_d).to_numpy() + SLOPE * 0 + agecurve * 1.0
    gm = pd.read_parquet(f"{D}/game_minutes.parquet")[["season", "game", "player", "team", "min"]]
    gi = pd.read_parquet(f"{D}/games_all.parquet")[["game", "home_team"]]
    pace = dict(zip(range(2015, 2026), [.8493, .8629, .8677, .8803, .892, .9018, .9087, .8965, .8968, .8964, .8969]))
    y = B.margin.to_numpy(); seas = B.season.to_numpy()
    def term(agemax):
        rows = []
        for S, P in PP.items():
            if S == 2014: continue
            m = (P.kind == "vet") & (P.age < agemax + 1)
            c = np.average(P.win[m], weights=P.min_proj[m] + 1)
            v = (P.win - c)[m.values]; v.index = P.player[m.values]
            g = gm[gm.season == S]; g = g[g.player.isin(v.index)].merge(gi, on="game")
            g["v"] = g.player.map(v) * g["min"] / 48 * pace[S] * np.where(g.team == g.home_team, 1, -1)
            rows.append(g.groupby("game").v.sum())
        return pd.concat(rows).reindex(B.game).fillna(0).to_numpy()
    def loso(X):
        X = np.c_[np.ones(len(y)), X]; e = []
        for s in np.unique(seas):
            tr, te = seas != s, seas == s; b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; e.append((y[te] - X[te] @ b) ** 2)
        return np.sqrt(np.concatenate(e).mean()), np.array([np.sqrt(x.mean()) for x in e])
    b0, s0 = loso(base[:, None])
    print(f"base (fixed weights + age curve) LOSO {b0:.4f}")
    for agemax in (21, 23):
        t = term(agemax)
        for k in (-.1, -.25, -.4):
            l, s = loso((base + k * t)[:, None])
            h = lambda sel: np.sqrt(np.mean(s[sel] ** 2)) - np.sqrt(np.mean(s0[sel] ** 2))
            print(f"  age<={agemax} k {k:+.2f}: LOSO {l:.4f} (d {l-b0:+.4f}) seasons better {int((s<s0).sum())}/11 halves {h(slice(0,5)):+.4f} {h(slice(5,11)):+.4f}")
        # free coefficient on the interaction, season-clustered t, and halves
        for nm, msk in (("all", seas > 0), ("2015-19", seas <= 2019), ("2020-25", seas >= 2020)):
            X = np.c_[np.ones(msk.sum()), base[msk], t[msk]]; bb = np.linalg.lstsq(X, y[msk], rcond=None)[0]; r = y[msk] - X @ bb
            ss = seas[msk]; XtXi = np.linalg.inv(X.T @ X); meat = sum(np.outer(X[ss == q].T @ r[ss == q], X[ss == q].T @ r[ss == q]) for q in np.unique(ss))
            se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
            print(f"    free {nm:8s}: interaction coef {bb[2]/bb[1]:+.3f} (relative to base), t {bb[2]/se[2]:+.1f}")

if __name__ == "__main__" and sys.argv[1:2] == ["team"]:
    k, agemax = float(sys.argv[2]), int(sys.argv[3]); TAG = sys.argv[4]
    for S, P in PP.items():
        T = pd.read_parquet(f"{D}/pre_T_{S}.parquet").set_index("team"); pace = (T.margin / T.net).mean() * 100
        d, m, c = delta(P, k, agemax); Q = P.drop(columns=["age", "ag", "win"]).copy(); Q["rating"] = Q.rating + d
        net = (Q.rating * Q.min_proj / 3936).groupby(Q.team).sum(); net -= net.mean()
        T["net"] = net; T["margin"] = net * pace / 100; T["model_wpct"] = norm.cdf(T.margin / 12.8)
        Q.to_parquet(f"{D}/{TAG}pre_P_{S}.parquet", index=False); T.reset_index().to_parquet(f"{D}/{TAG}pre_T_{S}.parquet", index=False)
