"""Positions (1-5 scale) for 2026 players with no NBA history, from their last college season (ESPN box scores,
hoopR-mbb-data): regression of rookie-season pos_est on college per-40 rates + listed G/F/C, fit on the 2024-25 and
2025-26 NBA rookie classes. Non-college newcomers: listed-position default. -> data/pos_2026.csv (all 2026 players)."""
import re, unicodedata, numpy as np, pandas as pd
D = "data"
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s); return re.sub(r"[^a-z]", "", s)
def college(y):
    b = pd.read_parquet(f"{D}/ncaa/player_box_{y}.parquet")
    b = b[b.minutes.fillna(0) > 0]
    num = ["minutes", "assists", "offensive_rebounds", "defensive_rebounds", "blocks", "steals",
           "three_point_field_goals_attempted", "field_goals_attempted", "free_throws_attempted", "fouls"]
    for c in num: b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0)
    g = b.groupby("athlete_id").agg(name=("athlete_display_name", "last"), pos=("athlete_position_abbreviation", "last"),
                                    **{c: (c, "sum") for c in num})
    g = g[g.minutes >= 200]
    per = lambda c: g[c] / g.minutes * 40
    X = pd.DataFrame({"key": g.name.map(norm), "ast": per("assists"), "oreb": per("offensive_rebounds"),
                      "dreb": per("defensive_rebounds"), "blk": per("blocks"), "stl": per("steals"),
                      "tpr": g.three_point_field_goals_attempted / g.field_goals_attempted.clip(lower=1),
                      "isG": (g.pos == "G").astype(float), "isC": (g.pos == "C").astype(float), "cmin": g.minutes})
    return X.sort_values("cmin").drop_duplicates("key", keep="last")
FX = ["ast", "oreb", "dreb", "blk", "stl", "tpr", "isG", "isC"]
po = pd.read_parquet(f"{D}/positions.parquet")
first = po.groupby("player").season.min()
tr = []
for s in (2024, 2025):  # NBA rookie season s <- college spring s
    rk = po[(po.season == s) & po.player.map(first).eq(s) & (po["min"] >= 300)].copy(); rk["key"] = rk.name.map(norm)
    tr.append(rk.merge(college(s), on="key"))
tr = pd.concat(tr)
A = np.c_[np.ones(len(tr)), tr[FX]]; b = np.linalg.lstsq(A, tr.pos_est, rcond=None)[0]
res = tr.pos_est - A @ b
print(f"train n={len(tr)}, R2 {1 - res.var() / tr.pos_est.var():.2f}, rmse {res.std():.2f}")
# 2026 players
P = pd.concat([pd.read_csv(f"{D}/players_proj_2026.csv"), pd.read_csv(f"{D}/players_fa_2026.csv")], ignore_index=True)
w = po[po.season >= 2023].copy(); w["wt"] = w["min"] * w.season.map({2023: .3, 2024: .6, 2025: 1.})
hist = (w.pos_est * w.wt).groupby(w.player).sum() / w.wt.groupby(w.player).sum()
hist = hist.combine_first(po.sort_values("season").groupby("player").pos_est.last())
col = pd.concat([college(2026), college(2025), college(2024)]).drop_duplicates("key")
P["key"] = P.name.map(norm)
out = P[["team", "name", "player", "key"]].copy()
out["pos_est"] = P.player.map(hist); out["pos_src"] = np.where(out.pos_est.notna(), "nba", None)
m = out.pos_est.isna()
c = out[m].merge(col, on="key", how="left")
pred = np.c_[np.ones(len(c)), c[FX]] @ b
out.loc[m, "pos_est"] = np.clip(pred, 1, 5); out.loc[m, "pos_src"] = np.where(np.isnan(pred), None, "college")
MAN = {"Khalifa Diop": 4.8, "Tarik Biberovic": 3.0, "Sergio De Larrea": 1.7, "Alpha Diallo": 3.0,  # internationals:
       "Alex Toohey": 3.2, "Karim Lopez": 3.6, "Bogoljub Markovic": 4.3}                          # listed position
for n, v in MAN.items():
    mm = out.name.eq(n) & out.pos_est.isna(); out.loc[mm, "pos_est"] = v; out.loc[mm, "pos_src"] = "manual (listed position)"
left = out.pos_est.isna(); print("no college match:", out[left].name.tolist())
out.loc[left, "pos_est"] = 3.0; out.loc[left, "pos_src"] = "default"
out["pos5"] = pd.cut(out.pos_est, [0, 1.5, 2.5, 3.45, 4.4, 9], labels=["PG", "SG", "SF", "PF", "C"]).astype(str)
out["pos3"] = pd.cut(out.pos_est, [0, 2.5, 3.9, 9], labels=["G", "W", "B"]).astype(str)
out.drop(columns="key").to_csv(f"{D}/pos_2026.csv", index=False)
print(out.pos_src.value_counts().to_dict())
print(out[out.pos_src == "college"][["name", "pos_est", "pos5"]].round(2).head(25).to_string(index=False))
