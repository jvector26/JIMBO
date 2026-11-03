"""Elo-style models vs JIMBO in the game backtest (session 44, 2026-11-03, user idea).

Base = JIMBO game backtest (ratings from the prior 3 seasons, ACTUAL minutes, box + 0.6 res_o + 0.8 res_d),
calibrated by leave-one-season-out (LOSO) linear fit = the research-bar baseline 13.4633.
NOTE: this base has NO in-season player-rating updates (production's M4), so any team-level layer here also
soaks up some of what M4 does. The fair comparison is residual Elo vs the production team-offset ridge (lam 80)
on the same base.

Variants (all strictly pre-game: predictions for date d use games before d only):
  R80   production team offsets: ridge on season-to-date residuals, lam_t 80, from 15 league games
  RELO  residual Elo: team adjustment r, pred = base + r_h - r_a, r_h += k*e, r_a -= k*e; season start r = c*r_end
  R80+RELO  residual Elo on top of the ridge layer
  ELO538  plain 538 Elo (MOV multiplier, K, carry-over toward 1505), from 1996, mapped to points by LOSO fit
  PELO  plain points Elo (linear update k*(margin - pred)), from 1996
  BASE+ELO  LOSO regression on JIMBO base and plain Elo together (does Elo carry independent info?)
k and c are picked LOSO (best on the other seasons, scored on the held-out one).
"""
import itertools, json, numpy as np, pandas as pd

D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"),
                                                                on=["game", "home_team", "away_team"])
G = pd.read_parquet(f"{D}/games_all.parquet")
G = G[G.score_ok].copy() if "score_ok" in G else G
G["margin"] = G.home_pts - G.away_pts
B = B.merge(G[["game", "date"]], on="game")
F = [c for c in B.columns if c.startswith("f:")]
B["raw"] = B[F].sum(axis=1) + B.unrated + 0.6 * B.res_o + 0.8 * B.res_d
B = B.sort_values(["date", "game"]).reset_index(drop=True)
y = B.margin.to_numpy(float); seas = B.season.to_numpy(); SEAS = np.unique(seas)


def loso_fit(X):
    X = np.c_[np.ones(len(X)), X]; p = np.empty(len(y))
    for s in SEAS:
        tr, te = seas != s, seas == s
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; p[te] = X[te] @ b
    return p


base = loso_fit(B.raw.to_numpy())
rm = lambda p: np.sqrt(((y - p) ** 2).mean())
print(f"base LOSO {rm(base):.4f}  (research bar 13.4633)")

teams = np.unique(np.r_[B.home_team, B.away_team]); ti = {t: i for i, t in enumerate(teams)}
hi = B.home_team.map(ti).to_numpy(); ai = B.away_team.map(ti).to_numpy()
dates = B.date.to_numpy()
day_groups = [np.flatnonzero(dates == d) for d in np.unique(dates)]  # in date order


def ridge_layer(ref, lam=80.0, min_games=15):
    """production team_offsets refit before every game day on season-to-date residuals vs ref."""
    out = np.zeros(len(y)); nT = len(teams)
    for s in SEAS:
        idx = np.flatnonzero(seas == s); A = lam * np.eye(nT); bvec = np.zeros(nT); n = 0
        for d in np.unique(dates[idx]):
            g = idx[dates[idx] == d]
            if n >= min_games:
                off = np.linalg.solve(A, bvec); out[g] = off[hi[g]] - off[ai[g]]
            e = y[g] - ref[g]
            for j, r in zip(g, e):
                h, a = hi[j], ai[j]
                A[h, h] += 1; A[a, a] += 1; A[h, a] -= 1; A[a, h] -= 1; bvec[h] += r; bvec[a] -= r
            n += len(g)
    return out


def relo_layer(ref, k, c):
    out = np.zeros(len(y)); r = np.zeros(len(teams)); last = None
    for g in day_groups:
        s = seas[g[0]]
        if s != last:
            r = c * r; last = s
        lay = r[hi[g]] - r[ai[g]]; out[g] = lay
        e = y[g] - ref[g] - lay
        np.add.at(r, hi[g], k * e); np.add.at(r, ai[g], -k * e)
    return out


def season_mse(p):
    return np.array([((y[seas == s] - p[seas == s]) ** 2).mean() for s in SEAS]), np.array([(seas == s).sum() for s in SEAS])


def pick_loso(runs):
    """runs: dict param -> pred. For each held-out season use the param best on the others."""
    M = {k: season_mse(p)[0] for k, p in runs.items()}; n = season_mse(base)[1]
    p_out = np.empty(len(y)); chosen = []
    for i, s in enumerate(SEAS):
        o = np.arange(len(SEAS)) != i
        best = min(M, key=lambda k: (M[k][o] * n[o]).sum())
        chosen.append(best); p_out[seas == s] = runs[best][seas == s]
    return p_out, chosen


def compare(p, ref, name, refname):
    d = (y - p) ** 2 - (y - ref) ** 2
    S = np.array([d[seas == s].sum() for s in SEAS]); N = len(d); m = d.mean()
    se = np.sqrt(((S - np.array([(seas == s).sum() for s in SEAS]) * m) ** 2).sum()) / N
    h1, h2 = seas <= 2019, seas >= 2020
    dr = lambda mk: np.sqrt(((y - p)[mk] ** 2).mean()) - np.sqrt(((y - ref)[mk] ** 2).mean())
    better = (S < 0).sum()
    print(f"  {name:22s} {rm(p):.4f}  vs {refname:10s} {rm(p)-rm(ref):+.4f}  t {m/se:+.2f}  better {better}/{len(SEAS)}"
          f"  halves {dr(h1):+.4f} / {dr(h2):+.4f}")
    return dict(name=name, ref=refname, rmse=rm(p), d=rm(p) - rm(ref), t=m / se, better=int(better), h1=dr(h1), h2=dr(h2))


rows = []
# ---------------------------------------------------------------- team-level layers on JIMBO
r80 = base + ridge_layer(base)
KS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06]; CS = [0.0, 0.25, 0.5]
relo_runs = {(k, c): base + relo_layer(base, k, c) for k in KS for c in CS}
relo, ch_relo = pick_loso(relo_runs)
both_runs = {(k, c): r80 + relo_layer(r80, k, c) for k in KS for c in CS}
both, ch_both = pick_loso(both_runs)
lam_runs = {lam: base + ridge_layer(base, lam) for lam in [20, 40, 80, 160, 320]}

print("\nin-sample grid, residual Elo on base (RMSE):")
print(pd.DataFrame({c: [rm(relo_runs[(k, c)]) for k in KS] for c in CS}, index=KS).round(4).to_string())
print("in-sample grid, residual Elo on top of ridge 80:")
print(pd.DataFrame({c: [rm(both_runs[(k, c)]) for k in KS] for c in CS}, index=KS).round(4).to_string())
print("ridge lam:", {l: round(rm(p), 4) for l, p in lam_runs.items()})
print("LOSO picks RELO:", sorted(set(ch_relo)), " R80+RELO:", sorted(set(ch_both)))

print("\nTEAM-LEVEL LAYERS ON JIMBO (preseason ratings + actual minutes)")
rows.append(compare(r80, base, "R80 (production)", "base"))
rows.append(compare(relo, base, "RELO", "base"))
rows.append(compare(relo, r80, "RELO", "R80"))
rows.append(compare(both, r80, "R80+RELO", "R80"))

# ---------------------------------------------------------------- plain Elo from 1996
GA = G.sort_values(["date", "game"]).reset_index(drop=True)
gh = GA.home_team.map(lambda t: t).to_numpy(); ga = GA.away_team.to_numpy(); gm = GA.margin.to_numpy(float)
gs = GA.season.to_numpy(); gd = GA.date.to_numpy()
allteams = np.unique(np.r_[gh, ga]); gi = {t: i for i, t in enumerate(allteams)}
gh = np.array([gi[t] for t in gh]); ga = np.array([gi[t] for t in ga])
gdays = [np.flatnonzero(gd == d) for d in np.unique(gd)]


def elo538(K=20, carry=0.75, hfa=100):
    e = np.full(len(allteams), 1505.0); out = np.zeros(len(GA)); last = None
    for g in gdays:
        s = gs[g[0]]
        if s != last and last is not None:
            e = carry * e + (1 - carry) * 1505
        last = s
        diff = e[gh[g]] + hfa - e[ga[g]]; out[g] = e[gh[g]] - e[ga[g]]
        pw = 1 / (1 + 10 ** (-diff / 400)); win = (gm[g] > 0).astype(float)
        wdiff = np.where(gm[g] > 0, diff, -diff)
        mult = (np.abs(gm[g]) + 3) ** 0.8 / (7.5 + 0.006 * wdiff)
        upd = K * mult * (win - pw)
        np.add.at(e, gh[g], upd); np.add.at(e, ga[g], -upd)
    return out


def pelo(k=0.04, carry=0.6, hca=2.5):
    r = np.zeros(len(allteams)); out = np.zeros(len(GA)); last = None
    for g in gdays:
        s = gs[g[0]]
        if s != last and last is not None:
            r = carry * r
        last = s
        out[g] = r[gh[g]] - r[ga[g]]
        e = gm[g] - (hca + out[g])
        np.add.at(r, gh[g], k * e); np.add.at(r, ga[g], -k * e)
    return out


pos = GA.reset_index().set_index("game")["index"]
take = pos.reindex(B.game).to_numpy()
e538_runs = {(K, c): loso_fit(elo538(K, c)[take]) for K in [10, 15, 20, 25, 30] for c in [0.5, 0.6, 0.75, 0.9]}
e538, ch538 = pick_loso(e538_runs)
pelo_raw = {(k, c): pelo(k, c)[take] for k in [0.02, 0.03, 0.04, 0.05, 0.07] for c in [0.4, 0.6, 0.75, 0.9]}
pelo_runs = {kc: loso_fit(v) for kc, v in pelo_raw.items()}
pel, chp = pick_loso(pelo_runs)
print("\nin-sample 538 Elo grid (rows K, cols carry):")
print(pd.DataFrame({c: [rm(e538_runs[(K, c)]) for K in [10, 15, 20, 25, 30]] for c in [0.5, 0.6, 0.75, 0.9]},
                   index=[10, 15, 20, 25, 30]).round(4).to_string())
print("in-sample points Elo grid (rows k, cols carry):")
print(pd.DataFrame({c: [rm(pelo_runs[(k, c)]) for k in [0.02, 0.03, 0.04, 0.05, 0.07]] for c in [0.4, 0.6, 0.75, 0.9]},
                   index=[0.02, 0.03, 0.04, 0.05, 0.07]).round(4).to_string())
print("LOSO picks 538:", sorted(set(ch538)), " points Elo:", sorted(set(chp)))

# combined: JIMBO raw + plain Elo, LOSO regression (Elo params = most-picked)
kc = max(set(chp), key=chp.count)
comb = loso_fit(np.c_[B.raw, pelo_raw[kc]])
X = np.c_[np.ones(len(y)), B.raw, pelo_raw[kc]]; bcoef = np.linalg.lstsq(X, y, rcond=None)[0]
comb2 = loso_fit(np.c_[B.raw, pelo_raw[kc], ridge_layer(base)])

print("\nPLAIN ELO (results only)")
rows.append(compare(e538, base, "ELO538", "base"))
rows.append(compare(pel, base, "PELO", "base"))
rows.append(compare(comb, base, "BASE+PELO regression", "base"))
rows.append(compare(comb, r80, "BASE+PELO regression", "R80"))
print(f"  joint coefs (all seasons): JIMBO raw {bcoef[1]:.3f}, points Elo {bcoef[2]:.3f}, params {kc}")

# ---------------------------------------------------------------- by stage of season
B["tg"] = B.groupby("season").cumcount() * 2 / 30  # approx league-average team games played
bins = [0, 10, 25, 45, 65, 83]
lab = pd.cut(B.tg, bins, right=False)
tab = pd.DataFrame({"games": lab.value_counts(sort=False),
                    "base": pd.Series((y - base) ** 2).groupby(lab).mean() ** .5,
                    "R80": pd.Series((y - r80) ** 2).groupby(lab).mean() ** .5,
                    "RELO": pd.Series((y - relo) ** 2).groupby(lab).mean() ** .5,
                    "R80+RELO": pd.Series((y - both) ** 2).groupby(lab).mean() ** .5,
                    "PELO": pd.Series((y - pel) ** 2).groupby(lab).mean() ** .5,
                    "BASE+PELO": pd.Series((y - comb) ** 2).groupby(lab).mean() ** .5})
print("\nRMSE by team games played:"); print(tab.round(3).to_string())
pd.DataFrame(rows).to_csv(f"{D}/elo_test_results.csv", index=False)
tab.to_csv(f"{D}/elo_test_by_stage.csv")
json.dump({"relo_picks": [list(x) for x in ch_relo], "both_picks": [list(x) for x in ch_both],
           "e538_picks": [list(x) for x in ch538], "pelo_picks": [list(x) for x in chp]}, open(f"{D}/elo_test_picks.json", "w"))
