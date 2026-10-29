import glob, numpy as np, pandas as pd

D = "/home/claude/data"
S = range(1996, 2026)

pg = pd.concat([pd.read_parquet(f"{D}/player_games_{s}.parquet") for s in S], ignore_index=True)
gm = pd.concat([pd.read_parquet(f"{D}/games_{s}.parquet") for s in S], ignore_index=True)

# names: full names from shot files, fall back to pbp names
sd = pd.concat([pd.read_parquet(f"{D}/sdnames_{s}.parquet") for s in S]).sort_values("season")
names = sd.drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["PLAYER_NAME"]
pn = pd.concat([pd.read_parquet(f"{D}/names_{s}.parquet") for s in S]).drop_duplicates("pid", keep="first").set_index("pid")["name"]
names = names.combine_first(pn)

pg = pg[pg["sec"] > 0].copy()
abbr = pd.concat([gm[["season", "home_team", "htm"]].set_axis(["season", "team", "abbr"], axis=1),
                  gm[["season", "away_team", "vtm"]].set_axis(["season", "team", "abbr"], axis=1)]).drop_duplicates(["season", "team"])
pg = pg.merge(abbr, on=["season", "team"], how="left")
pg["name"] = pg["player"].map(names)
pg.to_parquet(f"{D}/player_games_all.parquet", index=False)
gm.to_parquet(f"{D}/games_all.parquet", index=False)

# ---------------- on-court / off-court team stats from stints
rows = []
for s in S:
    st = pd.read_parquet(f"{D}/stints_{s}.parquet")
    st["dur"] = st["t_end"] - st["t_start"]
    for z in "AB":
        st["fga" + z] = st["fg2a" + z] + st["fg3a" + z]
    for side, opp in (("A", "B"), ("B", "A")):
        x = st[["game", "dur", f"team{side}", f"lineup{side}", f"pts{side}", f"fga{side}", f"oreb{side}", f"tov{side}", f"fta{side}",
                f"pts{opp}", f"fga{opp}", f"oreb{opp}", f"tov{opp}", f"fta{opp}"]].copy()
        x.columns = ["game", "dur", "team", "lineup", "pts", "fga", "oreb", "tov", "fta", "opts", "ofga", "ooreb", "otov", "ofta"]
        x["player"] = x["lineup"].str.split("-")
        x = x.explode("player")
        x = x[x["player"] != ""]
        x["player"] = x["player"].astype(np.int64)
        x["season"] = s
        rows.append(x.drop(columns="lineup").groupby(["season", "player", "team"], as_index=False).sum(numeric_only=True).drop(columns="game"))
on = pd.concat(rows).groupby(["season", "player"], as_index=False).sum(numeric_only=True).drop(columns="team")
on["poss"] = on["fga"] - on["oreb"] + on["tov"] + 0.44 * on["fta"]
on["oposs"] = on["ofga"] - on["ooreb"] + on["otov"] + 0.44 * on["ofta"]

# team season totals (for off-court = team - on)
tt = []
for s in S:
    st = pd.read_parquet(f"{D}/stints_{s}.parquet")
    for z in "AB":
        st["fga" + z] = st["fg2a" + z] + st["fg3a" + z]
    for side, opp in (("A", "B"), ("B", "A")):
        x = st[[f"team{side}", f"pts{side}", f"fga{side}", f"oreb{side}", f"tov{side}", f"fta{side}",
                f"pts{opp}", f"fga{opp}", f"oreb{opp}", f"tov{opp}", f"fta{opp}"]].copy()
        x.columns = ["team", "pts", "fga", "oreb", "tov", "fta", "opts", "ofga", "ooreb", "otov", "ofta"]
        x["season"] = s
        tt.append(x.groupby(["season", "team"], as_index=False).sum())
tt = pd.concat(tt).groupby(["season", "team"], as_index=False).sum()
tt["poss"] = tt["fga"] - tt["oreb"] + tt["tov"] + 0.44 * tt["fta"]
tt["oposs"] = tt["ofga"] - tt["ooreb"] + tt["otov"] + 0.44 * tt["ofta"]

# ---------------- player-season box
stat = ["sec", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pm",
        "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "c3_fga", "astd_fgm"]
ps = pg.groupby(["season", "player"]).agg(gp=("game", "nunique"), gs=("started", "sum"), **{c: (c, "sum") for c in stat}).reset_index()
teams = pg.sort_values("date").groupby(["season", "player"])["abbr"].agg(lambda x: "/".join(dict.fromkeys(x.dropna()))).rename("teams")
ps = ps.merge(teams, on=["season", "player"], how="left")
ps["name"] = ps["player"].map(names)
ps["min"] = ps["sec"] / 60
ps["mpg"] = ps["min"] / ps["gp"]
ps = ps.merge(on[["season", "player", "pts", "opts", "poss", "oposs", "fga", "fta", "tov"]].rename(columns={
    "pts": "on_pts", "opts": "on_opts", "poss": "on_poss", "oposs": "on_oposs", "fga": "on_tm_fga", "fta": "on_tm_fta", "tov": "on_tm_tov"}),
    on=["season", "player"], how="left")

# player's primary team for off-court calc (team with most minutes)
prim = pg.groupby(["season", "player", "team"])["sec"].sum().reset_index().sort_values("sec").drop_duplicates(["season", "player"], keep="last")
ps = ps.merge(prim[["season", "player", "team"]], on=["season", "player"], how="left")
# on-court for primary team only, to subtract cleanly
onp = pd.concat(rows).groupby(["season", "player", "team"], as_index=False).sum(numeric_only=True)
onp["poss"] = onp["fga"] - onp["oreb"] + onp["tov"] + 0.44 * onp["fta"]
onp["oposs"] = onp["ofga"] - onp["ooreb"] + onp["otov"] + 0.44 * onp["ofta"]
onp = onp.merge(prim[["season", "player", "team"]], on=["season", "player", "team"])
off = onp.merge(tt, on=["season", "team"], suffixes=("_on", "_tm"))
off["off_net"] = 100 * ((off.pts_tm - off.pts_on) / (off.poss_tm - off.poss_on).clip(lower=1)
                        - (off.opts_tm - off.opts_on) / (off.oposs_tm - off.oposs_on).clip(lower=1))
off["off_poss"] = off.poss_tm - off.poss_on
ps = ps.merge(off[["season", "player", "off_net", "off_poss"]], on=["season", "player"], how="left")

# ---------------- rates
p36 = 36 * 60 / ps["sec"]
for c in ["pts", "fga", "fg3a", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov"]:
    ps[f"{c}_36"] = ps[c] * p36
ps["ts_pct"] = ps["pts"] / (2 * (ps["fga"] + 0.44 * ps["fta"])).replace(0, np.nan)
ps["efg_pct"] = (ps["fgm"] + 0.5 * ps["fg3m"]) / ps["fga"].replace(0, np.nan)
ps["fg3_pct"] = ps["fg3m"] / ps["fg3a"].replace(0, np.nan)
ps["ft_pct"] = ps["ftm"] / ps["fta"].replace(0, np.nan)
ps["fg3a_rate"] = ps["fg3a"] / ps["fga"].replace(0, np.nan)
ps["fta_rate"] = ps["fta"] / ps["fga"].replace(0, np.nan)
ps["usg_pct"] = (ps["fga"] + 0.44 * ps["fta"] + ps["tov"]) / (ps["on_tm_fga"] + 0.44 * ps["on_tm_fta"] + ps["on_tm_tov"])
ps["ast_to"] = ps["ast"] / ps["tov"].replace(0, np.nan)
ps["on_ortg"] = 100 * ps["on_pts"] / ps["on_poss"]
ps["on_drtg"] = 100 * ps["on_opts"] / ps["on_oposs"]
ps["on_net"] = ps["on_ortg"] - ps["on_drtg"]
ps["on_off_diff"] = ps["on_net"] - ps["off_net"]
ps["pace_on"] = 48 * 60 * (ps["on_poss"] + ps["on_oposs"]) / 2 / ps["sec"]

cols = ["season", "player", "name", "teams", "gp", "gs", "min", "mpg", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pm", "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "c3_fga", "astd_fgm",
        "pts_36", "fga_36", "fg3a_36", "fta_36", "oreb_36", "dreb_36", "ast_36", "stl_36", "blk_36", "tov_36",
        "ts_pct", "efg_pct", "fg3_pct", "ft_pct", "fg3a_rate", "fta_rate", "usg_pct", "ast_to",
        "on_poss", "on_ortg", "on_drtg", "on_net", "off_net", "on_off_diff", "pace_on"]
ps = ps[cols].sort_values(["season", "min"], ascending=[True, False])
ps.to_parquet(f"{D}/player_seasons.parquet", index=False)
print(ps.shape)
