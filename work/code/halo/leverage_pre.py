"""Leverage test (user, 2026-10-06): do preseason team forecasts miss when a team's projection leans on thin-sample players
with big lineup residuals? Uses age_eval.py's scoring (model after minutes fix, Vegas, LOSO blend)."""
import runpy, numpy as np, pandas as pd, statsmodels.api as sm, io, contextlib
D = "/home/claude/data"
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("/home/claude/code/age_eval.py")
A, P, pblend = ns["A"].copy(), ns["P"], ns["p"]
A["e_mod"] = A.y - A["mod"]; A["e_blend"] = A.y - pblend
R = pd.read_parquet(f"{D}/bt_split_ratings.parquet"); ps = pd.read_parquet(f"{D}/player_seasons.parquet")
DEC = (0.3, 0.6, 1.0); rows = []
for S in range(2014, 2026):
    d = P[P.season == S].copy()
    k = ns["mults"](P[P.season != S]); d["min_adj"] = d.min_model * d.bin.map(k)
    d["min_proj2"] = d.groupby("team", group_keys=False).apply(lambda g: ns["allocate"](g, "min_adj"))
    r = R[R.season == S - 1].set_index("player"); h = ps[ps.season.between(S - 3, S - 1)]
    Pw = h.assign(w=h.season.map({S-3: DEC[0], S-2: DEC[1], S-1: DEC[2]})).eval("wp = w*on_poss").groupby("player").wp.sum()
    d["lin"] = (0.6 * d.player.map(r.res_o) + 0.8 * d.player.map(r.res_d)).fillna(0)
    d["Pw"] = d.player.map(Pw).fillna(0); rows.append(d)
P = pd.concat(rows)
# pace for points->wins: ~ margin/net; wins per point of margin ~ 82*phi(0)/12.8
WPP = 82 * 0.3989 / 12.8
P["min_use"] = P.min_proj2
def team_sum(mask, col="lin"):
    v = (P[col] * P.min_use / 3936).where(mask, 0)
    return v.groupby([P.season, P.team]).sum() * 0.96 * WPP     # 0.96 ~ pace/100
P["thin"] = P.Pw < 4000
feats = {"lineup_all": team_sum(P.Pw > 0),
         "lineup_thin": team_sum(P.thin & (P.Pw > 0)),
         "thin_high (lin>1.5)": team_sum(P.thin & (P.lin > 1.5)),
         "any_high (lin>2)": team_sum(P.lin > 2)}
F = pd.DataFrame(feats).reset_index().rename(columns={"level_0": "season", "level_1": "team"})
A = A.merge(F, on=["season", "team"], how="left").fillna(0)
print(f"{len(A)} team-seasons; model RMSE {np.sqrt((A.e_mod**2).mean()):.3f}, blend {np.sqrt((A.e_blend**2).mean()):.3f}")
print(A[list(feats)].describe().loc[["mean", "std", "max"]].round(2).to_string())
print("\nforecast error (actual - forecast, wins) on each feature (wins): coef ~ -1 = feature fully spurious, 0 = fine")
for f in feats:
    for e in ["e_mod", "e_blend"]:
        out = []
        for lab, sub in [("all", A), ("14-19", A[A.season <= 2019]), ("20-25", A[A.season >= 2020])]:
            r = sm.OLS(sub[e], sm.add_constant(sub[[f]])).fit(cov_type="cluster", cov_kwds={"groups": sub.season})
            out.append(f"{lab} {r.params[f]:+.2f} (t {r.tvalues[f]:+.1f})")
        print(f"  {f:20s} {e:8s} " + " | ".join(out))
# minutes realisation of thin-high players vs others
P["act"] = P.act_min; g = lambda m: (P.act[m].sum() / P.min_use[m].sum()).round(2)
print("\nactual/projected minutes: thin-high", g(P.thin & (P.lin > 1.5)), " thin other", g(P.thin & (P.lin <= 1.5) & (P.Pw > 0)),
      " thick", g(~P.thin), " n thin-high", int((P.thin & (P.lin > 1.5)).sum()))
P.to_parquet(f"{D}/leverage_pre_players.parquet", index=False); A.to_parquet(f"{D}/leverage_pre_teams.parquet", index=False)
