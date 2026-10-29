"""Team-direction features per team-season, using only information available in October of season S."""
import numpy as np, pandas as pd
D = "/home/claude/data"
ps = pd.read_parquet(f"{D}/player_seasons.parquet")[["season", "player", "age"]]
draft = pd.read_parquet(f"{D}/draft_r1.parquet")
id2abbr = pd.read_parquet(f"{D}/games_2025.parquet").drop_duplicates("home_team").set_index("home_team").htm.to_dict()

def features(S):
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet")
    T = pd.read_parquet(f"{D}/pre_T_{S}.parquet").set_index("team")
    last = ps[ps.season < S].sort_values("season").groupby("player").last()
    age = P.player.map(last.age) + (S - P.player.map(last.season))
    age = age.fillna(P.player.map(draft[draft.year == S].set_index("player").age)).fillna(21.0)
    P["age"] = age
    prev = pd.read_parquet(f"{D}/player_games_{S-1}.parquet").sort_values(["date", "game"]).drop_duplicates("player", keep="last")
    P["was_here"] = P.player.map(prev.set_index("player").team.map(id2abbr)) == P.team
    P["m"] = P.min_proj / 19680
    g = P.groupby("team")
    F = pd.DataFrame({
        "mw_age": g.apply(lambda d: (d.age * d.m).sum()),
        "young_share": g.apply(lambda d: d.m[d.age <= 24].sum()),
        "young_quality": g.apply(lambda d: (d.m * d.rating)[d.age <= 24].sum()),   # net contribution from <=24s
        "vet_quality": g.apply(lambda d: (d.m * d.rating.clip(lower=0))[d.age >= 28].sum()),  # tradeable positive vets
        "continuity": g.apply(lambda d: d.m[d.was_here].sum()),
        "rookie_share": g.apply(lambda d: d.m[d.kind != "vet"].sum()),
    })
    F["season"] = S
    return F.reset_index()

if __name__ == "__main__":
    F = pd.concat([features(S) for S in range(2014, 2026)])
    F.to_parquet(f"{D}/direction_features.parquet", index=False)
    print(F.describe().round(3).to_string())
