import numpy as np, pandas as pd, re, unicodedata
D = "/home/claude/data"
C = pd.read_parquet(f"{D}/ncaa_player_seasons.parquet")
dr = pd.read_parquet(f"{D}/draft_r1.parquet"); dr = dr[dr.year.between(2006, 2026)].copy()
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-,]", "", s); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()
C["nk"] = C.name.map(norm); dr["nk"] = dr.name.map(norm)
C["lk"] = C.nk.str.split().str[-1]; dr["lk"] = dr.nk.str.split().str[-1]
last = C.sort_values("min").drop_duplicates(["season", "nk"], keep="last")
m = dr.merge(last, left_on=["year", "nk"], right_on=["season", "nk"], how="left", suffixes=("", "_c"))
# fallback: same last name + school token in team name, same season
miss = m.athlete_id.isna()
for i in m[miss].index:
    r = m.loc[i]; c = C[(C.season == r.year) & (C.lk == r.lk)]
    c = c[c.team.fillna("").str.lower().apply(lambda t: bool(t) and (t in str(r.preDraftTeam).lower() or str(r.preDraftTeam).lower() in t))]
    if len(c) == 1:
        for k in c.columns: m.at[i, k] = c.iloc[0][k] if k in m.columns else None
        m.at[i, "athlete_id"] = c.iloc[0].athlete_id
m["college"] = m.athlete_id.notna()
coll_class = m["class"].fillna("").str.match(r"^(Fr|So|Jr|Sr)")
print("draft 2006-26 first-rounders", len(m), "| class says college", coll_class.sum(), "| matched", m.college.sum(),
      "| matched & college", (m.college & coll_class).sum())
print(m[coll_class & ~m.college][["year", "pick", "name", "preDraftTeam"]].to_string())
m.to_parquet(f"{D}/ncaa_draft_match.parquet", index=False)
