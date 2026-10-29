"""Better rookie + young-player priors, tested in the game-level calibration framework of aging.py:
margin ~ base (ratings thru N, actual minutes) + age bins + rookie/young exposures (minute-weighted, home minus away).
Coefficient / slope(base) = rating points per 100 for a unit of that feature."""
import numpy as np, pandas as pd, json
D = "/home/claude/data"
B = pd.read_parquet(f"{D}/backtest_components.parquet")
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
gm = pd.read_parquet(f"{D}/games_all.parquet")
dr = pd.read_parquet(f"{D}/draft_r1.parquet").dropna(subset=["player"])
dr = dr.assign(player=dr.player.astype(np.int64)).drop_duplicates("player")
W = json.load(open(f"{D}/weights_bt.json")); f = pd.read_parquet(f"{D}/features.parquet")
f["box"] = sum(f[c]*v for c, v in {**W["Wo"], **W["Wd"]}.items()) * 0.5
BOX = f.groupby(["season", "player"]).box.first(); FMIN = f.groupby(["season", "player"])["min"].first()
first_season = ps.groupby("player").season.min()
BINS = [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]
ht = lambda s: (int(s.split("-")[0])*12 + int(s.split("-")[1])) if isinstance(s, str) and "-" in s else np.nan
dr["h_in"] = dr.height.map(ht); dr["intl"] = (dr.origin_country.fillna("us") != "us").astype(float)
dr["young_cls"] = dr["class"].fillna("").str.startswith(("Fr", "So")).astype(float)
DR = dr.set_index("player")
SC = pd.read_parquet(f"{D}/ncaa_scores.parquet").dropna(subset=["player"]); SC = SC.assign(player=SC.player.astype(np.int64)).drop_duplicates("player").set_index("player")
CS_MEAN = SC.cscore[SC.college].mean()

def pace(s):
    st = pd.read_parquet(f"{D}/stints_{s}.parquet")
    poss = sum(st[f"fg2a{z}"] + st[f"fg3a{z}"] - st[f"oreb{z}"] + st[f"tov{z}"] + 0.44*st[f"fta{z}"] for z in "AB")
    return poss.sum() / 2 / st.game.nunique()

rows = []
for S in sorted(B.season.unique()):
    N = S - 1
    pgN = pd.concat([pd.read_parquet(f"{D}/player_games_{y}.parquet", columns=["player", "min", "date"]) for y in (N-2, N-1, N)])
    rated = set(pgN[pgN["min"] > 0].player)
    # late-season minutes rise in N: mpg after the season's midpoint date minus before
    p1 = pd.read_parquet(f"{D}/player_games_{N}.parquet", columns=["player", "min", "date"]); p1 = p1[p1["min"] > 0]
    mid = p1.date.median(); late = p1[p1.date > mid].groupby("player")["min"].mean() - p1[p1.date <= mid].groupby("player")["min"].mean()
    x = pd.read_parquet(f"{D}/player_games_{S}.parquet", columns=["game", "player", "team", "min"])
    x = x[x["min"] > 0].merge(ps[ps.season == S][["player", "age"]].drop_duplicates("player"), on="player", how="left")
    x["rated"] = x.player.isin(rated); x["w"] = x["min"] / 48 * pace(N) / 100
    x["yrs"] = S - x.player.map(first_season)               # 1 = second NBA season
    d = DR.reindex(x.player)
    x["pick"] = d.pick.values; x["dyear"] = d.year.values; x["dage"] = d.age.values
    x["h_in"] = d.h_in.values; x["intl"] = d.intl.values; x["ycls"] = d.young_cls.values
    x["rook1"] = (x.dyear == S) & ~x.rated
    x["lp"] = np.log(x.pick)
    # trend: box change N-1 -> N, both with 500+ min
    bN = np.array([BOX.get((N, p), np.nan) if FMIN.get((N, p), 0) >= 500 else np.nan for p in x.player])
    bM = np.array([BOX.get((N-1, p), np.nan) if FMIN.get((N-1, p), 0) >= 500 else np.nan for p in x.player])
    x["trend"] = np.nan_to_num(bN - bM); x["late"] = x.player.map(late).fillna(0).clip(-15, 15)
    young = x.rated & (x.age <= 24.99)
    g = gm[gm.season == S][["game", "home_team", "away_team"]]
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    out = pd.DataFrame({"game": g.game.values})
    def add(name, val):
        t = (x.w * val).groupby([x.game, x.team]).sum()
        out[name] = t.reindex(hk).fillna(0).to_numpy() - t.reindex(ak).fillna(0).to_numpy()
    for lo, hi, lab in BINS: add(lab, (x.rated & x.age.between(lo, hi)).astype(float))
    r1 = x.rook1
    add("rk_1_3", (r1 & (x.pick <= 3)).astype(float)); add("rk_4_10", (r1 & x.pick.between(4, 10)).astype(float))
    add("rk_11_30", (r1 & (x.pick >= 11)).astype(float))
    add("other_unrated", (~x.rated & ~r1).astype(float))
    cs = x.player.map(SC.cscore); col = cs.notna()
    add("rk_cs", np.where(r1 & col, cs - CS_MEAN, 0)); big = x.h_in.fillna(79) >= 82
    add("rk_cs_big", np.where(r1 & col & big, cs - CS_MEAN, 0)); add("rk_cs_sm", np.where(r1 & col & ~big, cs - CS_MEAN, 0)); add("rk_big", (r1 & big).astype(float))
    add("rk_cs_cap", np.where(r1 & col, np.clip(cs - CS_MEAN, -1, 1), 0)); add("rk_nocol", (r1 & ~col).astype(float))
    add("rk", r1.astype(float)); add("rk_lp", np.where(r1, x.lp - 2, 0)); add("rk_age", np.where(r1, x.dage.fillna(20) - 20, 0))
    add("rk_h", np.where(r1, (x.h_in.fillna(79) - 79)/3, 0)); add("rk_intl", np.where(r1, x.intl.fillna(0), 0))
    # young rated players
    add("y_trend", np.where(young, x.trend, 0)); add("y_late", np.where(young, x.late/5, 0))
    add("y_yr2", (young & (x.yrs == 1)).astype(float)); add("y_yr3", (young & (x.yrs == 2)).astype(float))
    fr = young & x.pick.notna() & (x.yrs <= 3)
    add("y_lp", np.where(fr, x.lp - 2, 0)); add("y_dage", np.where(fr, x.dage.fillna(20) - 20, 0))
    add("all_trend", np.where(x.rated & (x.age > 24.99), x.trend, 0))
    rows.append(out.assign(season=S))
A = pd.concat(rows).merge(B, on=["game", "season"])
comp = [c for c in B.columns if c.startswith("f:")]
A["base"] = A[comp].sum(axis=1) + 0.6*A.lineup_impact
A.to_parquet(f"{D}/young_prior_design.parquet", index=False)
