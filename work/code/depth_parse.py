"""Parse archived RealGM depth charts -> data/depth_realgm.parquet
(season, snap, team, player(NBA id), tier S/R/L, row, col, depth_rank within team). Name match to player_seasons."""
import gzip, glob, re, os, unicodedata, pandas as pd, numpy as np
D = "/home/claude/data"; H = "/home/claude/gh/research/depth_wayback/html"
TEAMS = {"Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BKN","Charlotte Hornets":"CHA","Chicago Bulls":"CHI",
 "Cleveland Cavaliers":"CLE","Dallas Mavericks":"DAL","Denver Nuggets":"DEN","Detroit Pistons":"DET","Golden State Warriors":"GSW",
 "Houston Rockets":"HOU","Indiana Pacers":"IND","Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL","Memphis Grizzlies":"MEM",
 "Miami Heat":"MIA","Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN","New Orleans Pelicans":"NOP","New York Knicks":"NYK",
 "Oklahoma City Thunder":"OKC","Orlando Magic":"ORL","Philadelphia Sixers":"PHI","Philadelphia 76ers":"PHI","Phoenix Suns":"PHX",
 "Portland Trail Blazers":"POR","Sacramento Kings":"SAC","San Antonio Spurs":"SAS","Toronto Raptors":"TOR","Utah Jazz":"UTA",
 "Washington Wizards":"WAS"}
TIER = {"depth_starters": "S", "depth_rotation": "R", "depth_limpt": "L"}
def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-]", " ", s); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"[^a-z]", "", s)
def parse(html):
    out = []
    for m in re.finditer(r"<h2[^>]*>(?:\s*<img[^>]*>)?\s*(\d{4})-(\d{4}) ([^<]+?) Depth Chart\s*</h2>(.*?)</table>", html, re.S):
        yr, team, body = int(m.group(1)), m.group(3).strip(), m.group(4)
        for ri, tr in enumerate(re.finditer(r'<tr class="(depth_\w+)">(.*?)</tr>', body, re.S)):
            tier = TIER.get(tr.group(1))
            cells = re.findall(r'<td data-th="[^"]*" class="depth-chart-cell"[^>]*>(.*?)</td>', tr.group(2), re.S)
            for ci, c in enumerate(cells):
                for a in re.finditer(r'/player/([^/]+)/Summary/(\d+)"', c):
                    out.append(dict(label_year=yr, team_name=team, tier=tier, row=ri, col=ci, slug=a.group(1), rgm=int(a.group(2))))
    return out
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
namecol = "name" if "name" in ps else [c for c in ps.columns if "name" in c.lower()][0]
rows = []
for f in sorted(glob.glob(f"{H}/*_realgm_*.html.gz")):
    season, _, ts = os.path.basename(f).split(".")[0].split("_")
    r = parse(gzip.open(f).read().decode("latin-1"))
    for x in r: x.update(season=int(season), snap=ts[:8])
    rows += r
    lab = {x["label_year"] for x in r}
    print(os.path.basename(f), len(r), "teams", len({x["team_name"] for x in r}), "label", lab)
R = pd.DataFrame(rows)
R["team"] = R.team_name.map(TEAMS)
print("unmapped teams", R[R.team.isna()].team_name.unique())
R = R[R.label_year == R.season]  # chart must be for the upcoming season
ALIAS = {"Enes-Kanter": "Enes Freedom", "Moe-Wagner": "Moritz Wagner", "Dennis-Schroeder": "Dennis Schroder",
         "Wesley-Iwundu": "Wes Iwundu", "BJ-Boston-Jr": "Brandon Boston"}
R["key"] = R.slug.map(lambda x: ALIAS.get(x, x.replace("-", " "))).map(norm)
# name index: players active in seasons S-4..S+1 (rookies appear in S)
ps = ps.dropna(subset=[namecol]); ps["key"] = ps[namecol].astype(str).map(norm)
out = []
for S, g in R.groupby("season"):
    cand = ps[ps.season.between(S - 4, S + 1)].sort_values("season").drop_duplicates("key", keep="last")[["key", "player"]]
    g = g.merge(cand, on="key", how="left")
    # fallback: unique last-name match among players who played for that team in S or S-1
    last = lambda k: k
    miss = g.player.isna()
    pool = ps[ps.season.between(S - 1, S)].copy()
    pool["last"] = pool[namecol].astype(str).str.split().str[-1].map(norm)
    for i in g.index[miss]:
        ln = norm(g.at[i, "slug"].split("-")[-1]); tm = g.at[i, "team"]
        c = pool[(pool["last"] == ln) & pool.teams.astype(str).str.contains(tm)].player.unique()
        if len(c) == 1: g.at[i, "player"] = c[0]
    out.append(g)
R = pd.concat(out)
print("match rate", R.player.notna().mean().round(3)); print(R[R.player.isna()].slug.value_counts().head(25))
R = R.dropna(subset=["player"]); R["player"] = R.player.astype(int)
R = R.sort_values(["season", "snap", "team", "row", "col"])
R = R.drop_duplicates(["season", "snap", "player"])  # a player listed at 2 positions keeps his first (highest) listing
R["depth_rank"] = R.groupby(["season", "snap", "team"]).cumcount() + 1
R.to_parquet(f"{D}/depth_realgm.parquet", index=False)
print(R.groupby(["season", "snap"]).agg(n=("player", "size"), teams=("team", "nunique")).to_string())
