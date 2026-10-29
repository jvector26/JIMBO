"""Team-season tank descriptors from games_all: wpct, margin, early (first 55 team games) vs late margin."""
import pandas as pd, numpy as np
D = "/home/claude/data"
g = pd.read_parquet(f"{D}/games_all.parquet")
g = g[g.score_ok].copy()
h = pd.DataFrame({"season": g.season, "date": g.date, "game": g.game, "team": g.htm, "m": g.home_pts - g.away_pts})
a = pd.DataFrame({"season": g.season, "date": g.date, "game": g.game, "team": g.vtm, "m": g.away_pts - g.home_pts})
t = pd.concat([h, a]).sort_values(["season", "team", "date", "game"])
t["k"] = t.groupby(["season", "team"]).cumcount()
t["n"] = t.groupby(["season", "team"]).k.transform("size")
t["late"] = t.k >= (t.n * 55 / 82).round()
r = t.groupby(["season", "team"]).agg(n=("m", "size"), wpct=("m", lambda x: (x > 0).mean()), margin=("m", "mean")).reset_index()
e = t[~t.late].groupby(["season", "team"]).m.mean().rename("m_early")
l = t[t.late].groupby(["season", "team"]).m.mean().rename("m_late")
wl = t[t.late].groupby(["season", "team"]).m.apply(lambda x: (x > 0).mean()).rename("w_late")
r = r.join(e, on=["season", "team"]).join(l, on=["season", "team"]).join(wl, on=["season", "team"])
r["drop"] = r.m_late - r.m_early
r.to_parquet(f"{D}/tank_team_seasons.parquet")
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(r.describe().round(2))
    print(r.sort_values("drop").head(25).round(2).to_string())
    for s, tm in [(2025, "IND"), (2025, "SAC"), (2025, "MEM"), (2023, "TOR"), (2023, "WAS"), (2014, "PHI"), (2022, "SAS"), (2022, "POR"), (2022, "UTA"), (2020, "OKC"), (2020, "HOU"), (2024, "UTA"), (2024, "POR"), (2025, "WAS")]:
        print(s, tm, r[(r.season == s) & (r.team == tm)][["wpct", "margin", "m_early", "m_late", "drop"]].round(2).to_dict("records"))
