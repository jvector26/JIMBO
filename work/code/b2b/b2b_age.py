"""Does the team B2B penalty grow with roster age? team-season minutes-weighted age (2014-25)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/code/b2b"); from fe import fe_fit
D = "/home/claude/data"
R = pd.read_parquet(f"{D}/b2b_player_games.parquet")
ta = R[R["min"] > 0].groupby(["season", "team"]).apply(lambda x: np.average(x.age, weights=x["min"])).rename("tage").reset_index()
G = pd.read_parquet(f"{D}/b2b_games.parquet"); G = G[G.season.between(2015, 2025)]
for side, col in [("h", "home_team"), ("a", "away_team")]:
    G = G.merge(ta.rename(columns={"team": col, "tage": f"age_{side}"}), on=["season", col], how="left")
G = G.dropna(subset=["age_h", "age_a"]); mu = np.r_[G.age_h, G.age_a].mean()
G["bxa_h"] = G.b2b_h * (G.age_h - mu); G["bxa_a"] = G.b2b_a * (G.age_a - mu)
G["bdiff"] = G.b2b_h - G.b2b_a; G["bxa_diff"] = G.bxa_h - G.bxa_a
print("team age sd %.2f" % np.r_[G.age_h, G.age_a].std())
for lab, m in [("2015-25", G.season >= 2015), ("2015-19", G.season <= 2019), ("2020-25", G.season >= 2020)]:
    b, se = fe_fit(G[m], ["bdiff", "bxa_diff"])
    print(f"{lab}: B2B {b[0]:+.2f} (t {b[0]/se[0]:.1f}); per year of team age {b[1]:+.2f} (t {b[1]/se[1]:.1f})")
