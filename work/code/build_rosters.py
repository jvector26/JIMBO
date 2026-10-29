import re, unicodedata, pandas as pd, numpy as np
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)
ALIAS = {"ronholland": "ronaldholland", "ejharkless": "elijahharkless", "jameshuff": "jayhuff", "cameronchristie": "camchristie", "nicolasclaxton": "nicclaxton"}
pg = pd.read_parquet("data/player_games_all.parquet")
names = pg.sort_values("date").drop_duplicates("player", keep="last")[["player", "name"]]
names["key"] = names.name.map(norm)
key2id = {}
for k, g in names.groupby("key"):
    key2id[k] = g.player.tolist()
last26 = pg[pg.season == 2025].sort_values("date").groupby("player").abbr.last()
active = set(pg[pg.season >= 2023].player)

rows, adds, drops = [], {}, {}
for line in open("rosters_2026_raw.txt"):
    line = line.strip()
    if not line or line.startswith("#"): continue
    team, role, ns = line.split("|")
    for n in ns.split(";"):
        n = n.strip()
        if role in ("S", "R", "L"): rows.append((team, role, n))
        elif role == "ADD": adds.setdefault(team, []).append(n)
        else: drops.setdefault(team, []).append(n)

def lookup(n):
    k = norm(n); k = ALIAS.get(k, k)
    ids = key2id.get(k, [])
    if len(ids) > 1:  # prefer recently active
        ids = [i for i in ids if i in active] or ids
    return ids[0] if ids else None

R = pd.DataFrame(rows, columns=["team", "role", "name"])
for t, lst in adds.items():
    R = pd.concat([R, pd.DataFrame({"team": t, "role": "?", "name": lst})])
R["player"] = R.name.map(lookup)
placed = set(R.player.dropna())
dropped = {lookup(n) for lst in drops.values() for n in lst}
for t in ["POR", "SAC", "SAS", "UTA", "WAS"]:
    base = last26[last26 == t].index
    extra = [p for p in base if p not in placed and p not in dropped]
    R = pd.concat([R, pd.DataFrame({"team": t, "role": "?", "player": extra,
                                    "name": [names.set_index("player").name.get(p) for p in extra]})])
has = R.player.notna()
R = pd.concat([R[has].drop_duplicates(subset=["player"], keep="first"), R[~has]]).reset_index(drop=True)  # a player can't be on two teams
R.to_csv("data/rosters_2026_base.csv", index=False)  # then code/depth_apply.py -> data/rosters_2026.csv
print("roster rows", len(R), "| matched to NBA history", R.player.notna().sum(), "| no NBA history (rookies/new)", R.player.isna().sum())
print(R.groupby("team").size().describe().round(1).to_dict())
print("unmatched:", sorted(R[R.player.isna()].name.tolist()))
