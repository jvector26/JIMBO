"""College player-seasons from ESPN box scores (sportsdataverse/hoopR-mbb-data), opponent-adjusted team strength,
and a match to NBA first-round picks (draft_r1)."""
import numpy as np, pandas as pd, re, unicodedata, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
D = "/home/claude/data"
S_COLS = ["minutes", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
          "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "assists", "steals",
          "blocks", "turnovers", "fouls", "points"]
SHORT = ["min", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "pts"]

def team_ratings(x):
    g = x.drop_duplicates(["game_id", "team_id"])[["game_id", "team_id", "opponent_team_id", "team_score", "opponent_team_score", "home_away"]]
    n = g.team_id.value_counts(); d1 = set(n[n >= 10].index)
    g = g[g.team_id.isin(d1)].copy()
    g["opp"] = np.where(g.opponent_team_id.isin(d1), g.opponent_team_id, -1)
    teams = sorted(d1) + [-1]; ix = {t: i for i, t in enumerate(teams)}
    r = np.arange(len(g)); T = len(teams)
    X = sp.csr_matrix((np.r_[np.ones(len(g)), -np.ones(len(g)), np.where(g.home_away == "home", 1.0, -1.0)],
                       (np.r_[r, r, r], np.r_[g.team_id.map(ix), g.opp.map(ix), np.full(len(g), T)])), shape=(len(g), T + 1))
    y = (g.team_score - g.opponent_team_score).clip(-40, 40).to_numpy(float)
    b = lsqr(X, y, damp=0.3)[0]
    rat = pd.Series(b[:T], index=teams); rat -= rat[rat.index != -1].mean()
    sos = g.assign(o=g.opp.map(rat)).groupby("team_id").o.mean()
    return rat.drop(-1), sos

rows = []
for y in range(2006, 2027):
    x = pd.read_parquet(f"{D}/ncaa/pb_{y}.parquet"); x = x[~x.did_not_play & x.minutes.notna()]
    rat, sos = team_ratings(x)
    x = x.rename(columns=dict(zip(S_COLS, SHORT)))
    tg = x.groupby(["game_id", "team_id"])[["min", "fga", "orb", "tov", "fta"]].sum()
    tg["poss"] = tg.fga - tg.orb + tg.tov + 0.44 * tg.fta
    x = x.join(tg[["min", "poss"]].rename(columns={"min": "tmin", "poss": "tposs"}), on=["game_id", "team_id"])
    x["pon"] = x.tposs * x["min"] / (x.tmin / 5).clip(lower=1)
    agg = x.groupby("athlete_id").agg(name=("athlete_display_name", "last"), team_id=("team_id", lambda s: s.mode().iat[0]),
                                      team=("team_location", "last"), pos=("athlete_position_abbreviation", "last"),
                                      g=("game_id", "nunique"), gs=("starter", "sum"), pon=("pon", "sum"),
                                      tgames=("game_id", "nunique"), **{c: (c, "sum") for c in SHORT})
    agg["team_g"] = agg.team_id.map(x.drop_duplicates(["game_id", "team_id"]).groupby("team_id").size())
    agg["srs"] = agg.team_id.map(rat); agg["sos"] = agg.team_id.map(sos); agg["season"] = y
    rows.append(agg.reset_index())
    print(y, len(agg), "teams rated", len(rat), "top", rat.sort_values().index[-1], round(rat.max(), 1), flush=True)
C = pd.concat(rows)
C["seasons_in"] = C.groupby("athlete_id").cumcount() + 1
C.to_parquet(f"{D}/ncaa_player_seasons.parquet", index=False)
