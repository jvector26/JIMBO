"""Test: shrink 3P% toward a player-specific prior from FT% (+ 3PA rate, height) instead of the league average.
Target: NEXT season's 3P% (weighted by next-season 3PA). All inputs use data through the current season only."""
import numpy as np, pandas as pd
D = "/home/claude/data"
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
p = ps.groupby(["player", "season"]).agg(fg3m=("fg3m", "sum"), fg3a=("fg3a", "sum"), ftm=("ftm", "sum"), fta=("fta", "sum"),
                                          fga=("fga", "sum"), mn=("min", "sum"), h=("height_in", "first")).reset_index().sort_values(["player", "season"])
lg = p.groupby("season")[["fg3m", "fg3a", "ftm", "fta"]].sum(); p["lg3"] = p.season.map(lg.fg3m/lg.fg3a); p["lgft"] = p.season.map(lg.ftm/lg.fta)
g = p.groupby("player")
for c in ["fg3m", "fg3a", "ftm", "fta", "fga"]: p["c_" + c] = g[c].cumsum()          # career to date (incl. current)
p["ft_c"] = (p.c_ftm + 0.75*50) / (p.c_fta + 50)                                  # career FT%, lightly shrunk
p["ft_s"] = (p.ftm + 0.75*50) / (p.fta + 50)
p["rate3"] = np.log((p.c_fg3a + 1) / (p.c_fga + 10))                              # career 3PA share of FGA
p["hc"] = (p.h.fillna(p.h.median()) - 78) / 3
p["rel3"] = p.fg3m/p.fg3a.clip(lower=1) - p.lg3
nx = p.groupby("player").shift(-1); p["y3"] = nx.fg3m / nx.fg3a; p["ya"] = nx.fg3a; p["ylg"] = nx.lg3
T = p[(p.ya >= 50) & (p.season.between(2000, 2024))].copy()
T["yrel"] = T.y3 - T.ylg
# prior model for (true) 3P% relative to league: fit on career-to-date 3P% of high-volume careers (>=300 c_fg3a)
H = p[p.c_fg3a >= 300].copy(); H["c3rel"] = H.c_fg3m/H.c_fg3a - H.lg3
Xp = lambda d: np.c_[np.ones(len(d)), d.ft_c - 0.75, d.rate3, d.hc]
def fit_prior(d):
    w = d.c_fg3a.clip(upper=3000).to_numpy(); X = Xp(d)
    return np.linalg.solve((X*w[:, None]).T @ X, (X*w[:, None]).T @ d.c3rel.to_numpy())
def score(pred, d):
    w = d.ya.to_numpy(); return np.sqrt(np.sum(w*(d.yrel - pred)**2) / w.sum())
res = {}
for split, tr_s, te_s in [("all(in-sample prior)", (2000, 2024), (2000, 2024)), ("prior fit <=2012, test 2013-24", (1996, 2012), (2013, 2024))]:
    bp = fit_prior(H[H.season.between(*tr_s)]); d = T[T.season.between(*te_s)]
    prior = Xp(d) @ bp
    print(f"\n{split}: prior coefs (const, FT% per 1.0, log 3PA share, height/3in) {bp.round(3)}")
    for k in [100, 250, 400, 600, 1000]:
        cur = (d.fg3m + k*(d.lg3)) / (d.fg3a + k) - d.lg3                       # current: season 3P% shrunk to league
        car = (d.c_fg3m + k*(d.lg3)) / (d.c_fg3a + k) - d.lg3                   # career shrunk to league
        new = (d.c_fg3m + k*(d.lg3 + prior)) / (d.c_fg3a + k) - d.lg3           # career shrunk to FT-based prior
        news = (d.fg3m + k*(d.lg3 + prior)) / (d.fg3a + k) - d.lg3              # season shrunk to FT-based prior
        print(f"  k {k:5d}: season->league {score(cur, d)*100:.3f}  season->FTprior {score(news, d)*100:.3f}  "
              f"career->league {score(car, d)*100:.3f}  career->FTprior {score(new, d)*100:.3f}")
    print(f"  (no info: league avg {score(0*d.yrel, d)*100:.3f}; prior only {score(prior, d)*100:.3f})  [pct pts of next-season 3P%]")

# ---- direct approach: regress next-season relative 3P% on shrunk career 3P% + FT% + volume, CV by season
print("\nDIRECT REGRESSION (leave-one-season-out), next-season 3P% error in pct pts:")
T["car400"] = (T.c_fg3m + 400*T.lg3)/(T.c_fg3a + 400) - T.lg3
T["ftd"] = T.ft_c - 0.75; T["ftd_lo"] = T.ftd * 400/(T.c_fg3a + 400)     # FT info matters more when few 3PA
sets = {"career k400 only": ["car400"], "+ FT%": ["car400", "ftd"], "+ FT% x low-volume": ["car400", "ftd", "ftd_lo"],
        "+ 3PA share, height": ["car400", "ftd", "ftd_lo", "rate3", "hc"]}
for lab, cols in sets.items():
    for per, d in [("2000-24", T), ("2013-24", T[T.season >= 2013])]:
        pr = pd.Series(np.nan, index=d.index)
        for s in d.season.unique():
            tr = T[T.season != s]; w = tr.ya.to_numpy(); X = np.c_[np.ones(len(tr)), tr[cols]]
            b = np.linalg.solve((X*w[:, None]).T @ X, (X*w[:, None]).T @ tr.yrel.to_numpy())
            te = d.season == s; pr[te] = np.c_[np.ones(te.sum()), d.loc[te, cols]] @ b
        print(f"  {lab:24s} {per}: {score(pr, d)*100:.3f}", end="")
    print("   coefs", dict(zip(["c"]+cols, b.round(3))))
# low-volume subgroup (career 3PA < 300): where the prior should matter
L_ = T[T.c_fg3a < 300]; print("\nlow-volume shooters (career 3PA<300), n", len(L_))
