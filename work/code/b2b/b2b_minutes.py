"""Player minutes on the 2nd night of back-to-backs, 2014-25 (session 37). Roster span = team games between a player's first
and last appearance for that team that season; non-appearance in the span = 0 minutes (injury or rest, can't tell apart).
Within player-team-season comparisons (B2B night 2 vs his other games)."""
import numpy as np, pandas as pd
D = "/home/claude/data"
GM = pd.read_parquet(f"{D}/game_minutes.parquet")
G = pd.read_parquet(f"{D}/b2b_games.parquet")
G = G[G.season.between(2014, 2025)]
TG = pd.concat([G[["season", "game", "d", "home_team", "rest_h", "g34_h"]].rename(columns={"home_team": "team", "rest_h": "rest", "g34_h": "g34"}),
                G[["season", "game", "d", "away_team", "rest_a", "g34_a"]].rename(columns={"away_team": "team", "rest_a": "rest", "g34_a": "g34"})])
TG = TG.sort_values(["season", "team", "d"]); TG["k"] = TG.groupby(["season", "team"]).cumcount()
# also: is the NEXT game a B2B (this is night 1)?
TG["night1"] = (TG.groupby(["season", "team"]).rest.shift(-1) == 0).astype(int)
GM = GM.merge(TG[["game", "team", "k"]], on=["game", "team"])
span = GM.groupby(["season", "team", "player"]).k.agg(["min", "max"]).reset_index()
rows = []
for (S, t), sp_ in span.groupby(["season", "team"]):
    tg = TG[(TG.season == S) & (TG.team == t)].set_index("k")
    for _, r in sp_.iterrows():
        ks = np.arange(r["min"], r["max"] + 1)
        rows.append(pd.DataFrame({"season": S, "team": t, "player": r.player, "k": ks}))
R = pd.concat(rows, ignore_index=True).merge(TG[["season", "team", "k", "game", "d", "rest", "night1", "g34"]], on=["season", "team", "k"])
R = R.merge(GM[["game", "team", "player", "min", "started"]], on=["game", "team", "player"], how="left")
R["min"] = R["min"].fillna(0); R["app"] = (R["min"] > 0).astype(int)
R = R[R.rest.notna()]
R["b2b"] = (R.rest == 0).astype(int)
bd = pd.read_parquet(f"{D}/birthdates.parquet").set_index("player").birthdate
R["age"] = (R.d - pd.to_datetime(R.player.map(bd))).dt.days / 365.25
grp = R.groupby(["season", "team", "player"])
R["mpg_wp"] = grp["min"].transform("sum") / grp.app.transform("sum")
R["ngames"] = grp.k.transform("size")
R = R[R.ngames >= 20]
R["age_b"] = pd.cut(R.age, [0, 24.99, 29.99, 32.99, 34.99, 50], labels=["<=24", "25-29", "30-32", "33-34", "35+"])
R["mp_b"] = pd.cut(R.mpg_wp, [0, 15, 24, 30, 60], labels=["<15", "15-24", "24-30", "30+"])
print("player-team-games", len(R), " B2B night-2 share %.3f" % R.b2b.mean())
def within(df, ycol, xcol="b2b"):
    """within player-team-season: mean y on x=1 minus mean y on x=0, weighted by player-season."""
    a = df.groupby(["season", "team", "player", xcol])[ycol].mean().unstack()
    a = a.dropna(); n = len(a)
    d = a[1] - a[0]; return d.mean(), d.std() / np.sqrt(n), a[0].mean(), n
print("\nNight 2 of a B2B vs his other games (within player-season). Absence = not playing at all (injury OR rest).")
print(f"{'group':14s} {'base abs%':>9s} {'d abs%':>8s} {'base MIN':>8s} {'d MIN':>7s} {'d MIN|played':>12s}   n")
def show(lab, df):
    a = within(df, "app"); m = within(df, "min")
    dp = within(df[df.app == 1], "min")
    print(f"{lab:14s} {100*(1-a[2]):9.1f} {-100*a[0]:+8.1f} {m[2]:8.1f} {m[0]:+7.2f} {dp[0]:+12.2f}   {a[3]}")
show("all", R)
for c in ["age_b", "mp_b"]:
    for v in R[c].cat.categories: show(f"{c[:3]} {v}", R[R[c] == v])
print("\nStarters-level (30+ mpg) by age:")
for v in R.age_b.cat.categories: show(f"30+ & {v}", R[(R.mp_b == "30+") & (R.age_b == v)])
print("\nby era (all players):")
for lab, m in [("2014-19", R.season <= 2019), ("2020-25", R.season >= 2020)]: show(lab, R[m])
print("\n30+ mpg, 30+ yrs, by era:")
for lab, m in [("2014-19", R.season <= 2019), ("2020-25", R.season >= 2020)]: show(lab, R[m & (R.mp_b == "30+") & (R.age >= 30)])
print("\nNight 1 (game BEFORE a B2B) vs other non-B2B games:")
show("night1", R[R.b2b == 0].assign(b2b=lambda x: x.night1))
R.to_parquet(f"{D}/b2b_player_games.parquet", index=False)
