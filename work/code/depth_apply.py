"""Apply the latest RealGM depth chart to the 2026-27 roster (session 15, 2026-09-23).
  data/rosters_2026_base.csv (build_rosters.py) + gh/live/2026/depth/realgm_*.html.gz (latest)
  -> data/rosters_2026.csv (team moved per chart, role/tier from chart) + data/depth_2026_log.csv (every change).
Rules (data/depth_config.json):
  - player on ANOTHER team's chart -> moved there (trades/signings RealGM already has)
  - player with NBA history on a chart but on no roster of ours (FA pool / unrostered) -> added to that team
  - chart player with no NBA history and not on our rosters -> added as newcomer (role = tier), as build_rosters did
  - carry-over teams (no chart when the base roster was built) : players on NO chart -> dropped (-> FA pool)
  - elsewhere, rostered player on no chart -> dropped too (released/waived), except exempt keys (e.g. unsigned RFA Duren)
  - tier = listing on his own team's chart (S/R/L); exempt -> X (no minutes change). A team with fewer than
    min_team_players on the chart is left untouched (partial page guard).
project.py then sets want = a[tier] + b[tier] * m0 (data/depth_tier.json) before the team allocation."""
import glob, json, os, sys, pandas as pd, numpy as np
sys.path.insert(0, "/home/claude/gh/model"); import depth_chart as dc
D = "/home/claude/data"
CFG = json.load(open(f"{D}/depth_config.json")); EX = set(CFG["exempt"])
_fs = sorted(glob.glob("/home/claude/gh/live/2026/depth/realgm_*.html.gz"))
f = [x for x in _fs if os.environ.get("DEPTH_DATE", "") in x][-1]  # DEPTH_DATE=YYYYMMDD pins a chart (regression checks); default = latest
cdate = os.path.basename(f)[7:15]
R = dc.load_chart(f, 2026)
B = pd.read_csv(f"{D}/rosters_2026_base.csv")
B["key"] = [f"p{int(p)}" if pd.notna(p) else f"n:{n}|{t}" for p, n, t in zip(B.player, B["name"], B.team)]
ps = pd.read_parquet(f"{D}/player_seasons.parquet", columns=["player", "name", "season", "teams"])
last = ps[ps.season >= 2020].sort_values("season").drop_duplicates("player", keep="last")
Hh = pd.DataFrame({"key": "p" + last.player.astype(int).astype(str), "name": last["name"].values,
                   "team": last.teams.astype(str).str[-3:].values, "player": last.player.values})
people = pd.concat([B[["key", "name", "team"]], Hh[~Hh.key.isin(B.key)][["key", "name", "team"]]], ignore_index=True)
R = dc.match(R, people)
size = R.groupby("team").size(); good = set(size[size >= CFG["min_team_players"]].index)
print(f"chart {cdate}: {len(R)} players, {len(good)}/30 teams usable, matched {R.key.notna().sum()}")
on = R.dropna(subset=["key"]).set_index("key")
log = []
out = B.copy()
for i, r in out.iterrows():
    k = r.key
    if k in EX: continue
    if k in on.index:
        ct = on.team[k]
        if ct != r.team and ct in good:
            log.append(dict(change="moved", name=r["name"], key=k, frm=r.team, to=ct, tier=on.tier[k])); out.at[i, "team"] = ct
    elif r.team in good:
        why = "carry-over roster, on no chart" if r.team in CFG["carryover_teams"] else "on no chart (released?)"
        log.append(dict(change="dropped", name=r["name"], key=k, frm=r.team, to="FA", tier="", why=why)); out.at[i, "team"] = None
out = out[out.team.notna()].copy()
# additions: on a chart, not on any roster of ours
nm = Hh.set_index("key")
for ci, c in R[R.team.isin(good)].iterrows():
    if pd.notna(c.key) and c.key in set(out.key): continue
    if pd.notna(c.key) and str(c.key).startswith("p"):
        out = pd.concat([out, pd.DataFrame([{"team": c.team, "role": c.tier, "name": nm.name.get(c.key, c.slug.replace("-", " ")),
                                             "player": float(c.key[1:]), "key": c.key}])], ignore_index=True)
        log.append(dict(change="added", name=nm.name.get(c.key), key=c.key, frm="", to=c.team, tier=c.tier, why="NBA history"))
    elif pd.isna(c.key):
        n = c.slug.replace("-", " "); R.at[ci, "key"] = f"n:{n}|{c.team}"
        out = pd.concat([out, pd.DataFrame([{"team": c.team, "role": c.tier, "name": n, "player": np.nan, "key": f"n:{n}|{c.team}"}])],
                        ignore_index=True)
        log.append(dict(change="added", name=n, key=f"n:{n}|{c.team}", frm="", to=c.team, tier=c.tier, why="no NBA history"))
# tiers (after moves) ; role = tier for everyone on a chart (role drives no-history newcomer minutes in project.py)
out["tier"] = dc.tiers(out, R[R.team.isin(good)], EX).values
keep_old = ~out.team.isin(good)                       # untouched teams keep their base role and no tier change
out.loc[keep_old & (out.tier != "X"), "tier"] = "X"
# user tier overrides (depth_config.json tier_override: key -> {"tier": S/R/L/N, "note"}); win over chart and exempt
TO = {k: v["tier"] for k, v in CFG.get("tier_override", {}).items()}
for k, t in TO.items():
    hit = out.key == k
    if hit.any():
        log.append(dict(change="tier override", name=out.loc[hit, "name"].iloc[0], key=k, frm=out.loc[hit, "tier"].iloc[0], to=t, tier=t, why="user"))
        out.loc[hit, "tier"] = t
    else: print("tier override key not on any roster:", k)
out["role"] = np.where(out.tier.isin(["S", "R", "L"]), out.tier, out.role)  # N (user: out of rotation) keeps base role
out["depth_date"] = cdate
out = out.drop_duplicates("key")
out.drop(columns=["key"]).to_csv(f"{D}/rosters_2026.csv", index=False)
L = pd.DataFrame(log); L.to_csv(f"{D}/depth_2026_log.csv", index=False)
print(L.to_string() if len(L) else "no changes")
print("tiers:", out.tier.value_counts().to_dict(), "| rows", len(out), "| per team", out.groupby("team").size().describe()[["min", "max"]].to_dict())
