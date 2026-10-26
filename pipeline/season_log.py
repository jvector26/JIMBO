"""Player-page season data (2026-10-26, "This season" tab): per-player game lines + nightly rating history.

  python pipeline/season_log.py --season 2026 [--out site/data]

Runs after nightly.py (reads its latest.json). Writes:
  gamelog.json     {"asof", "cols", "g": {game: [date, home, away, home_pts, away_pts]},
                    "p": {player_id: [[game, team_is_home, starter, min, pts, fgm, fga, 3pm, 3pa, ftm, fta, oreb, dreb,
                                       ast, stl, blk, tov, pm], ...]}}       # played games, oldest first
                    DNP rows: [game, team_is_home, -1, "reason"]   (only when the box lists a reason)
  rating_hist.json {"d": [asof dates], "r": {player key: [model rating on each date or null]}, "pre": {key: preseason},
                    "s": {key: [pure 2026-27 season rating (box score only) on each date or null]}}
Box points from play-by-play shots are checked against the official box line; a mismatch keeps the box points.
Display only: nothing here feeds the model.
"""
import argparse, csv, gzip, json, os, re, collections
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--out", default=os.path.join(ROOT, "site", "data"))
a = ap.parse_args()
L = os.path.join(ROOT, "live", str(a.season))
COLS = ["game", "home", "gs", "min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
        "ast", "stl", "blk", "tov", "pm"]


def mins(s):
    m = re.match(r"PT(\d+)M([\d.]+)S", str(s))
    return int(m.group(1)) + float(m.group(2)) / 60 if m else 0.0


def pbp_counts(path):
    st = collections.defaultdict(collections.Counter)
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            t, res = r["actionType"], r["shotResult"]
            pid = int(float(r["personId"])) if r["personId"] else None
            if not pid:
                continue
            if t in ("2pt", "3pt"):
                st[pid]["fga"] += 1
                if t == "3pt": st[pid]["fg3a"] += 1
                if res == "Made":
                    st[pid]["fgm"] += 1
                    if t == "3pt": st[pid]["fg3m"] += 1
                    if r["assistPersonId"]:
                        st[int(float(r["assistPersonId"]))]["ast"] += 1
            elif t == "freethrow":
                st[pid]["fta"] += 1
                if res == "Made": st[pid]["ftm"] += 1
            elif t == "rebound":
                st[pid]["oreb" if r["subType"] == "offensive" else "dreb"] += 1
            elif t == "steal": st[pid]["stl"] += 1
            elif t == "block": st[pid]["blk"] += 1
            elif t == "turnover": st[pid]["tov"] += 1
    return st


meta = pd.read_csv(os.path.join(L, "games_meta.csv"))
meta = meta[meta.status == 3].sort_values(["date_et", "game"])
G, P, bad = {}, collections.defaultdict(list), 0
for _, g in meta.iterrows():
    gid = f"00{int(g.game)}"
    bf, pf = os.path.join(L, "box", f"{gid}.csv.gz"), os.path.join(L, "pbp", f"{gid}.csv.gz")
    if not (os.path.exists(bf) and os.path.exists(pf)):
        continue
    G[int(g.game)] = [g.date_et, g.htm, g.vtm, int(g.home_pts), int(g.away_pts)]
    st = pbp_counts(pf)
    box = pd.read_csv(bf)
    for _, b in box.iterrows():
        pid, home = int(b.player), int(bool(b.home))
        if not b.played:
            why = b.not_playing_desc if isinstance(b.not_playing_desc, str) and b.not_playing_desc.strip() else ""
            if why: P[pid].append([int(g.game), home, -1, why.strip()[:40]])
            continue
        s = st[pid]
        pts_pbp = 2 * (s["fgm"] - s["fg3m"]) + 3 * s["fg3m"] + s["ftm"]
        pts = int(b.pts) if pd.notna(b.pts) else pts_pbp
        bad += pts != pts_pbp
        P[pid].append([int(g.game), home, int(b.starter), round(mins(b.minutes), 1), pts,
                       *[int(s[k]) for k in COLS[5:17]], int(b.pm) if pd.notna(b.pm) else 0])
print(f"season_log: {len(G)} games, {len(P)} players, {sum(len(v) for v in P.values())} rows, pts mismatches {bad}")

LT = json.load(open(os.path.join(a.out, "latest.json")))
json.dump({"asof": LT["asof"], "cols": COLS, "g": G, "p": P},
          open(os.path.join(a.out, "gamelog.json"), "w"), separators=(",", ":"), allow_nan=False)

# rating history: one entry per asof date (later runs that day overwrite it)
hf = os.path.join(a.out, "rating_hist.json")
H = json.load(open(hf)) if os.path.exists(hf) else {"d": [], "r": {}, "pre": {}}
d = LT["asof"]
if d not in H["d"]:                       # insert in date order (a replay with --asof can be earlier than the last date)
    j = sum(x < d for x in H["d"])
    H["d"].insert(j, d)
    for k in H["r"]:
        H["r"][k].insert(j, None)
    for k in H.get("s", {}):
        H["s"][k].insert(j, None)
i, n = H["d"].index(d), len(H["d"])
for k in H["r"]:
    H["r"][k] += [None] * (n - len(H["r"][k]))
for p in LT["players"]:
    k = p["k"]
    row = H["r"].setdefault(k, [None] * n)
    row += [None] * (n - len(row))
    row[i] = p["r"]
    H["pre"][k] = p["r_pre"]
    if p.get("sr") is not None:                       # pure 2026-27 season rating (nightly.py sr), same dates
        srow = H.setdefault("s", {}).setdefault(k, [None] * n)
        srow += [None] * (n - len(srow))
        srow[i] = p["sr"]
for k in H.get("s", {}):
    H["s"][k] += [None] * (n - len(H["s"][k]))
json.dump(H, open(hf, "w"), separators=(",", ":"), allow_nan=False)
