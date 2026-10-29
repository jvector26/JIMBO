import json, re, unicodedata, numpy as np, pandas as pd
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)
bio = pd.DataFrame(json.load(open("bio.json")))
bio["bd"] = pd.to_datetime(bio["BIRTHDAY"], format="%m/%d/%Y", errors="coerce")
bio["key"] = bio["PLAYER"].map(norm)
bio["h_in"] = bio["HEIGHT"].str.extract(r"(\d+)-(\d+)").astype(float).pipe(lambda x: x[0] * 12 + x[1])
bio["wt"] = pd.to_numeric(bio["WEIGHT"], errors="coerce")
bio["draft"] = pd.to_numeric(bio["DRAFT"], errors="coerce")

ps = pd.read_parquet("data/player_seasons.parquet")
first = ps.groupby("player").agg(name=("name", "first"), first_season=("season", "min"))
ALIAS = {"Herbert Jones": "Herb Jones", "Santi Aldama": "Santiago Aldama", "Bub Carrington": "Carlton Carrington",
         "Brandon Boston": "BJ Boston", "Ish Wainright": "Ishmail Wainright", "Joshua Primo": "Josh Primo",
         "Matt Hurt": "Matthew Hurt", "Cat Barber": "Anthony Barber", "Cui Yongxi": "Yongxi Cui"}
first["key"] = first["name"].map(lambda n: norm(ALIAS.get(n, n)))
old = pd.read_parquet("data/birthdates.parquet").set_index("player")["birthdate"]

m = first.reset_index().merge(bio[["key", "bd", "h_in", "wt", "draft", "POS"]], on="key", how="left")
# resolve same-name collisions: age at debut must be 17-24ish window widened to 17-40; prefer closest draft year
m["age_debut"] = (pd.to_datetime((m.first_season + 1).astype(str) + "-02-01") - m.bd).dt.days / 365.25
m["ok"] = m.age_debut.between(17.5, 40) | m.bd.isna()
m["dd"] = (m.draft - m.first_season).abs().fillna(5)
m = m[m.ok].sort_values(["player", "dd"]).drop_duplicates("player")
m = m.set_index("player")
bd = old.combine_first(m["bd"])  # keep ID-matched birthdates first, fill from name match
# consistency check where both exist
both = old.index.intersection(m.dropna(subset=["bd"]).index)
agree = ((old[both] - m.loc[both, "bd"]).abs().dt.days <= 1).mean()
print("ID-matched vs name-matched birthdate agreement:", round(agree, 4), "on", len(both), "players")
ref = pd.to_datetime((ps.season + 1).astype(str) + "-02-01")
ps["age"] = ((ref - ps.player.map(bd)).dt.days / 365.25).round(1)
for c, s in [("height_in", "h_in"), ("weight_lb", "wt"), ("pos", "POS")]:
    ps[c] = ps.player.map(m[s])
cov = ps.groupby("season").apply(lambda d: (d.age.notna() * d["min"]).sum() / d["min"].sum())
print("age coverage (share of minutes) by season:", cov.round(3).tail(8).to_dict())
miss = ps[ps.age.isna()].groupby("name")["min"].sum().sort_values(ascending=False)
print("unmatched players (by total minutes):", miss.head(15).round(0).to_dict())
ps.to_parquet("data/player_seasons.parquet", index=False)
pd.Series(bd, name="birthdate").rename_axis("player").reset_index().to_parquet("data/birthdates.parquet")
