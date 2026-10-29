import json, numpy as np, pandas as pd
dr = pd.DataFrame(json.load(open("draftrepo/frontend/public/data/draft_history.json")))
dr = dr[(dr["round"] == 1) & dr.nba_id.notna()].copy(); dr["name"] = dr["player"]; dr["player"] = dr.nba_id.astype(np.int64)
dr.to_parquet("data/draft_r1.parquet", index=False)
ps = pd.read_parquet("data/player_seasons.parquet"); pg = pd.read_parquet("data/player_games_all.parquet")
gm = pd.read_parquet("data/games_all.parquet")
# team win% by season, keyed by abbreviation
res = pd.concat([gm.assign(abbr=gm.htm, w=gm.home_pts > gm.away_pts), gm.assign(abbr=gm.vtm, w=gm.away_pts > gm.home_pts)])
wp = res.groupby(["season", "abbr"]).w.mean()
ng = gm.groupby("season").size() * 2 / 30
h = dr[dr.year.between(1997, 2025)][["year", "pick", "player", "team", "age"]].rename(columns={"year": "season"})
h = h.merge(ps[["season", "player", "min", "gp"]], on=["season", "player"], how="left").fillna({"min": 0, "gp": 0})
main = pg.groupby(["season", "player", "abbr"])["min"].sum().reset_index().sort_values("min").drop_duplicates(["season", "player"], keep="last")
h = h.merge(main[["season", "player", "abbr"]], on=["season", "player"], how="left")
h["abbr"] = h.abbr.fillna(h.team)
# team quality: that team's win% the season before the rookie arrived
h["prev_wp"] = [wp.get((s - 1, a), np.nan) for s, a in zip(h.season, h.abbr)]
h = h.dropna(subset=["prev_wp"])
h["min82"] = h["min"] * 82 / h.season.map(ng)          # scale lockout/short seasons to 82 games
h["mpg82"] = h.min82 / 82
h["bucket"] = pd.cut(h.pick, [0, 3, 6, 10, 20, 30], labels=["1-3", "4-6", "7-10", "11-20", "21-30"])
h["team_q"] = pd.cut(h.prev_wp, [0, .35, .5, 1], labels=["bad (<35%)", "middle", "good (>50%)"])
print(f"{len(h)} first-round rookies, 1997-2025\n")
print("Rookie minutes per team game (82-game basis), by draft slot and quality of team joined:")
t = h.pivot_table(index="bucket", columns="team_q", values="mpg82", aggfunc="mean", observed=False).round(1)
t["all"] = h.groupby("bucket", observed=False).mpg82.mean().round(1)
t["n"] = h.groupby("bucket", observed=False).size()
print(t.to_string())
# regression for projection: minutes ~ log(pick) + prev win% + draft age
X = lambda d: np.c_[np.ones(len(d)), np.log(d.pick), d.prev_wp, d.age.fillna(d.age.median())]
tr, te = h[h.season <= 2015], h[h.season > 2015]
b = np.linalg.lstsq(X(tr), tr.min82, rcond=None)[0]
e = te.min82 - X(te) @ b
print(f"\nout-of-sample (2016-26 classes): typical error {np.sqrt((e**2).mean()):.0f} min; R^2 {1-(e**2).sum()/((te.min82-te.min82.mean())**2).sum():.2f}")
b = np.linalg.lstsq(X(h), h.min82, rcond=None)[0]
print("coefficients: per log(pick) %.0f min, per +10 pts of team win%% %.0f min, per year of draft age %.0f min" % (b[1], b[2] / 10, b[3]))
np.save("data/rookie_min_coef.npy", b)
h.to_parquet("data/rookie_hist.parquet", index=False)
