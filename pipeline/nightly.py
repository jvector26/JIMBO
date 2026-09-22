"""JIMBO nightly update: parse season-to-date games -> update ratings (M4) -> rest-of-season minutes (injuries,
usage, user overrides) -> team strengths -> simulate the rest of the season on the real schedule -> site/data JSON.

  python pipeline/nightly.py --season 2026 [--asof 2026-11-15] [--sims 10000]
Replay/testing: --state state/2025_replay --parsed DIR --schedule-from-parsed (uses DIR/games_{S}.parquet as schedule)
"""
import argparse, datetime as dt, json, os, re, sys, unicodedata, numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model")); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import jimbo_update as U, jimbo_sim as SIM

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026); ap.add_argument("--asof")
ap.add_argument("--state"); ap.add_argument("--parsed"); ap.add_argument("--schedule-from-parsed", action="store_true")
ap.add_argument("--sims", type=int, default=10000); ap.add_argument("--out", default=os.path.join(ROOT, "site", "data"))
ap.add_argument("--no-history", action="store_true")
a = ap.parse_args()
S = a.season
st_dir = a.state or os.path.join(ROOT, "state", str(S))
cfg = json.load(open(os.path.join(st_dir, "config.json")))
PL = pd.read_parquet(os.path.join(st_dir, "players.parquet"))
H = pd.read_parquet(os.path.join(st_dir, "hist_box.parquet"))
TP = pd.read_csv(os.path.join(st_dir, "teams_pre.csv"), index_col=0)
live = os.path.join(ROOT, "live", str(S))
et_now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=4)
asof = a.asof or et_now.strftime("%Y-%m-%d")        # games dated before asof are complete
asof_i = int(asof.replace("-", ""))
SIG, HCA, SL = cfg["sig"], cfg["hca"], cfg["slope"]

def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s); return re.sub(r"[^a-z]", "", s)

# ------------------------------------------------------------------ schedule + results
if a.schedule_from_parsed:
    g = pd.read_parquet(os.path.join(a.parsed, f"games_{S}.parquet"))
    sched = pd.DataFrame({"game": g.game, "date": g.date.astype(int), "home": g.htm, "away": g.vtm,
                          "home_pts": g.home_pts, "away_pts": g.away_pts})
else:
    s0 = pd.read_csv(os.path.join(live, "schedule.csv"), dtype={"gameId": str})
    s0 = s0[s0.kind == "reg"]
    sched = pd.DataFrame({"game": s0.game, "date": s0.date_et.str.replace("-", "").astype(int), "home": s0.home,
                          "away": s0.away, "home_pts": s0.home_pts, "away_pts": s0.away_pts, "time_utc": s0.time_utc})
done = sched.date < asof_i
# parsed season-to-date play-by-play
pg = st = games = None
parsed = a.parsed or os.path.join(ROOT, "state", "season_parse")
if not a.parsed and os.path.isdir(os.path.join(live, "pbp")) and len(os.listdir(os.path.join(live, "pbp"))):
    import parse_live
    parse_live.parse(S, parsed)
if os.path.exists(os.path.join(parsed, f"games_{S}.parquet")):
    gm = pd.read_parquet(os.path.join(parsed, f"games_{S}.parquet"))
    gm = gm[gm.date < asof_i]
    pg = pd.read_parquet(os.path.join(parsed, f"player_games_{S}.parquet")); pg = pg[pg.game.isin(gm.game)]
    st = pd.read_parquet(os.path.join(parsed, f"stints_{S}.parquet")); st = st[st.game.isin(gm.game)]
    games = pd.DataFrame({"game": gm.game, "date": gm.date, "home": gm.htm, "away": gm.vtm, "home_id": gm.home_team,
                          "margin": gm.home_pts - gm.away_pts})
    abbr_of = pd.concat([gm.set_index("home_team").htm, gm.set_index("away_team").vtm]).groupby(level=0).first()
n_done = int(done.sum()); n_parsed = 0 if games is None else len(games)
print(f"asof {asof}: {n_done} games final in schedule, {n_parsed} parsed")
# results from the schedule (authoritative for W/L; parsed games carry the same scores)
res = sched[done & sched.home_pts.notna()]
wins = pd.Series(0, index=SIM.TEAMS); losses = pd.Series(0, index=SIM.TEAMS)
for h_, a_, hp, ap_ in zip(res.home, res.away, res.home_pts, res.away_pts):
    if hp > ap_: wins[h_] += 1; losses[a_] += 1
    else: wins[a_] += 1; losses[h_] += 1
played = wins + losses
rem = sched[~done & sched.home.isin(SIM.TEAMS) & sched.away.isin(SIM.TEAMS)]   # TBD (NBA Cup) slots filled in the sim
# ------------------------------------------------------------------ ratings
prior = PL.dropna(subset=["player"]).drop_duplicates("player").set_index("player").rating_pre
prior.index = prior.index.astype(np.int64)
if games is not None and len(games):
    r, det = U.update_ratings(prior, H, pg, st, games, cfg)
else:
    r, det = prior.copy(), pd.DataFrame({"prior": prior, "box_delta": 0.0, "rapm_delta": 0.0, "poss": 0.0})
pace = cfg["pace"]
if st is not None and games is not None and len(games) >= 60:
    pa = sum(st[f"fg2a{z}"] + st[f"fg3a{z}"] - st[f"oreb{z}"] + st[f"tov{z}"] + 0.44 * st[f"fta{z}"] for z in "AB")
    pace = float(pa.sum() / 2 / st.game.nunique())
# team offsets from past-game residuals (predicted with current ratings and the minutes actually played)
off = pd.Series(0.0, index=SIM.TEAMS)
if games is not None and len(games) >= cfg["min_games"]:
    x = pg[pg["min"] > 0].merge(games[["game", "home_id"]], on="game")
    x["side"] = np.where(x.team == x.home_id, 1.0, -1.0)
    x["ws"] = 5 * x["min"] / x.groupby(["game", "team"])["min"].transform("sum") * x.side
    pr = (x.ws * x.player.map(r).fillna(SL * cfg["unrated"])).groupby(x.game).sum()
    past = games.assign(pred=games.game.map(pr).fillna(0) * pace / 100 + HCA)
    off = U.team_offsets(past, cfg["lam_t"]).reindex(SIM.TEAMS).fillna(0)
# ------------------------------------------------------------------ rosters, minutes, injuries
P = PL.copy()
P["pid"] = P.player.fillna(-1).astype(np.int64)
P["rating"] = P.pid.map(r).where(P.pid >= 0, P.rating_pre).fillna(P.rating_pre)
gp = pd.Series(dtype=float); smin = pd.Series(dtype=float)
if pg is not None and len(pg):
    q = pg[pg["min"] > 0].sort_values("date")
    gp = q.groupby("player").size(); smin = q.groupby("player")["min"].sum()
    last_team = q.groupby("player").team.last().map(abbr_of)
    P["team"] = P.pid.map(last_team).fillna(P.team)
    newp = last_team.index.difference(P.pid)            # played this season but not in the state
    if len(newp):
        P = pd.concat([P, pd.DataFrame({"key": ["p%d" % p for p in newp], "player": newp, "pid": newp, "name": "",
                                        "team": last_team[newp].values, "rating": r.reindex(newp).fillna(SL * cfg["unrated"]).values,
                                        "rating_pre": SL * cfg["unrated"], "mpg_pre": 10.0, "avail_pre": .6,
                                        "pos_est": 3.0, "hustle": 0.0})], ignore_index=True)
P["gp"] = P.pid.map(gp).fillna(0); P["mpg_season"] = (P.pid.map(smin) / P.gp).fillna(0)
k = cfg["mpg_k"]; P["mpg_est"] = (k * P.mpg_pre.fillna(10) + P.gp * P.mpg_season) / (k + P.gp)
tg = P.team.map(played).fillna(0)
P["avail"] = ((cfg["avail_k"] * P.avail_pre.fillna(.7) + P.gp) / (cfg["avail_k"] + tg)).clip(0, 1)
# user overrides (dashboard settings, migrated to overrides.json)
ovf = os.path.join(ROOT, "overrides.json")
OV = json.load(open(ovf)).get("players", {}) if os.path.exists(ovf) else {}
for kk, o in OV.items():
    m = P.key == kk
    if not m.any(): continue
    if "t" in o: P.loc[m, "team"] = o["t"]
    if "r" in o: P.loc[m, "rating"] = float(o["r"])
# injuries (ESPN): games missed from now
inj_f = os.path.join(live, "injuries.csv")
P["status"] = ""; P["miss"] = 0.0
G_rem = (82 - played).clip(lower=0).astype(int)   # includes TBD Cup-week games
if os.path.exists(inj_f) and not a.schedule_from_parsed:
    I = pd.read_csv(inj_f); I["k"] = I.name.map(norm_name)
    P["k"] = P.name.map(norm_name)
    for _, row in I.iterrows():
        m = P.k == row.k
        if not m.any(): continue
        t = P.loc[m, "team"].iloc[0]
        stt = str(row.status); cm = str(row.get("comment", "")).lower()
        if stt.lower().startswith("out") or "suspen" in stt.lower():
            rd = pd.to_datetime(row.get("return_date"), errors="coerce")
            if pd.notna(rd):   # an explicit return date beats comment text (old notes say 'remainder of the season')
                rdi = int(rd.strftime("%Y%m%d"))
                miss = int((((rem.home == t) | (rem.away == t)) & (rem.date < rdi)).sum())
            elif "season" in cm and ("out for the season" in cm or "rest of the season" in cm or "remainder" in cm):
                miss = G_rem.get(t, 0)
            else:
                miss = cfg["out_default_games"]
        elif "day" in stt.lower():
            miss = 0.5
        else:
            miss = 0
        P.loc[m, "status"] = stt; P.loc[m, "miss"] = min(miss, G_rem.get(t, 0))
P["g_rem"] = P.team.map(G_rem).fillna(0)
P["min_want"] = ((P.g_rem - P.miss).clip(lower=0) * P.avail * P.mpg_est).where(P.team != "FA", 0)
for kk, o in OV.items():   # minutes overrides: season games g (of 82) and mpg -> rest-of-season share
    m = P.key == kk
    if m.any() and ("g" in o or "mpg" in o):
        gr = P.loc[m, "g_rem"].iloc[0]
        g_ = o.get("g", 82 * P.loc[m, "avail"].iloc[0]); mp = o.get("mpg", P.loc[m, "mpg_est"].iloc[0])
        P.loc[m, "min_want"] = g_ / 82 * gr * mp; P.loc[m, "locked"] = True
P["locked"] = P.get("locked", False)
P["locked"] = P["locked"].fillna(False).astype(bool)

def allocate(df, budget, cap):
    lock = df.locked; mins = df.min_want.clip(upper=cap).copy()
    free_b = budget - mins[lock].sum(); fr = ~lock
    if free_b <= 0: mins[fr] = 0; return mins
    tot = mins[fr].sum()
    if tot > free_b:  # trim from the bottom of the rotation
        excess = tot - free_b
        for i in mins[fr].sort_values().index:
            cut = min(excess, mins[i]); mins[i] -= cut; excess -= cut
            if excess <= 0: break
    elif tot > 0:
        for _ in range(10):
            room = mins[fr & (mins < cap)].sum(); fixed = mins[fr & (mins >= cap)].sum()
            if room <= 0: break
            f = (free_b - fixed) / room; mins[fr & (mins < cap)] = (mins[fr & (mins < cap)] * f).clip(upper=cap)
    return mins
P["min_rem"] = 0.0
for t in SIM.TEAMS:
    m = P.team == t
    if G_rem[t] > 0 and m.any():
        P.loc[m, "min_rem"] = allocate(P[m], 240 * G_rem[t], 3000 / 82 * G_rem[t])   # cap as project.py
# ------------------------------------------------------------------ team strengths
net = pd.Series({t: (P.rating[P.team == t] * P.min_rem[P.team == t]).sum() / max(48 * G_rem[t], 1) for t in SIM.TEAMS})
if (G_rem == 0).all():
    net[:] = 0
net = net - net.mean()
model_m = net * pace / 100 + off
gap0 = TP.blend_margin_pre - TP.model_margin_pre
decay = cfg["gap_k"] / (cfg["gap_k"] + played)
fc_m = model_m + gap0.reindex(SIM.TEAMS).fillna(0) * decay
fc_m = fc_m - fc_m.mean()
avg_played = float(played.mean())
tau = np.interp(avg_played, [0, 41, 70, 82], [cfg["tau_pre"], cfg["tau_mid"], cfg["tau_late"], cfg["tau_late"]])
remA = np.array([[SIM.IDX[h], SIM.IDX[w]] for h, w in zip(rem.home, rem.away)], int).reshape(-1, 2)
out = {}
for mode, mg in (("forecast", fc_m), ("model", model_m - model_m.mean())):
    W, TRUE, rng = SIM.simulate_rest(mg.reindex(SIM.TEAMS).to_numpy(), tau, remA, wins.to_numpy(), played.to_numpy(), a.sims)
    post = SIM.postseason(W, TRUE, rng)
    out[mode] = dict(mean=W.mean(0), p10=np.percentile(W, 10, 0), p90=np.percentile(W, 90, 0), **post,
                     over=np.array([(W[:, i] > TP.line.get(t, 99)).mean() for i, t in enumerate(SIM.TEAMS)]))
# ------------------------------------------------------------------ today's games
def game_strength(t):
    m = (P.team == t)
    per = (P.mpg_est * np.where(P.status.str.lower().str.startswith("out"), 0, np.where(P.status.str.lower().str.contains("day"), .5, 1)) * m).where(m, 0)
    if per.sum() <= 0: return 0.0
    per = per * 240 / per.sum()
    return float((P.rating * per).sum() / 48)
game_day = asof_i
if not (sched.date == asof_i).any() and (sched.date > asof_i).any():
    game_day = int(sched.date[sched.date > asof_i].min())   # no games today: show the next game day
today = sched[sched.date == game_day]
odds_f = os.path.join(live, "odds", f"{game_day}.csv")
OD = pd.read_csv(odds_f) if os.path.exists(odds_f) else pd.DataFrame(columns=["home", "away", "spread", "details", "over_under", "home_ml", "away_ml"])
netm = {t: game_strength(t) for t in SIM.TEAMS}; nm_mean = np.mean(list(netm.values()))
tonight = []
for _, gg in today.iterrows():
    mh = (netm[gg.home] - nm_mean) * pace / 100 + off[gg.home] + gap0.get(gg.home, 0) * decay[gg.home]
    ma = (netm[gg.away] - nm_mean) * pace / 100 + off[gg.away] + gap0.get(gg.away, 0) * decay[gg.away]
    sp_ = mh - ma + HCA
    o = OD[(OD.home == gg.home) & (OD.away == gg.away)]
    tonight.append({"game": int(gg.game), "time_utc": gg.get("time_utc"), "home": gg.home, "away": gg.away,
                    "margin": round(float(sp_), 1), "p_home": round(float(norm.cdf(sp_ / SIG)), 3),
                    "line": (o.details.iloc[0] if len(o) else None), "total": (float(o.over_under.iloc[0]) if len(o) and pd.notna(o.over_under.iloc[0]) else None)})
# ------------------------------------------------------------------ write
os.makedirs(a.out, exist_ok=True)
F, M = out["forecast"], out["model"]
teams = []
for i, t in enumerate(SIM.TEAMS):
    teams.append({"t": t, "conf": SIM.conf_of[t], "w": int(wins[t]), "l": int(losses[t]),
                  "proj_w": round(float(F["mean"][i]), 1), "p10": int(F["p10"][i]), "p90": int(F["p90"][i]),
                  "model_w": round(float(M["mean"][i]), 1), "line": (None if pd.isna(TP.line.get(t)) else float(TP.line[t])),
                  "pre_w": round(float(TP.blend_w_pre.get(t, np.nan)), 1),
                  **{k: round(float(F[k][i]), 4) for k in ["playoffs", "playin", "top6", "seed1", "r2", "cf", "finals", "title", "over"]},
                  "margin": round(float(fc_m[t]), 2), "model_margin": round(float(model_m[t]), 2), "offset": round(float(off[t]), 2),
                  "g_rem": int(G_rem[t]), "gp": int(played[t]), "gap": round(float(gap0.get(t, 0)), 3),
                  "dec": round(float(decay[t]), 4)})
pl = P[(P.team != "FA") | (P.gp > 0)].copy()
pl["rating_delta"] = pl.rating - pl.rating_pre
players = [{"k": row.key, "n": row["name"], "t": row.team, "pos": round(float(row.pos_est), 2) if pd.notna(row.pos_est) else 3.0,
            "r": round(float(row.rating), 2), "r_pre": round(float(row.rating_pre), 2), "gp": int(row.gp),
            "mpg": round(float(row.mpg_season), 1), "min_rem": round(float(row.min_rem)), "status": row.status or "",
            # app fields: pre-allocation minutes wanted, mpg estimate, availability, games out
            "mw": round(float(row.min_want)), "me": round(float(row.mpg_est), 2), "av": round(float(row.avail), 3),
            "miss": round(float(row.miss), 1)}
           for _, row in pl.sort_values("rating", ascending=False).iterrows() if row.team != "FA" or row.gp > 0]   # every rostered player (injured ones carry their status to the app)
latest = {"asof": asof, "season": S, "games_final": n_done, "games_parsed": n_parsed, "pace": round(pace, 1), "tau": round(float(tau), 2),
          "teams": teams, "players": players, "tonight": tonight, "sims": a.sims,
          "game_day": f"{str(game_day)[:4]}-{str(game_day)[4:6]}-{str(game_day)[6:]}",
          "cfg": {"gap_k": cfg["gap_k"], "sig": SIG, "hca": HCA},
          "rem": [f"{h}-{w}" for h, w in zip(rem.home, rem.away)],   # remaining schedule (TBD Cup slots not included)
          "generated_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%MZ")}
json.dump(latest, open(os.path.join(a.out, "latest.json"), "w"), separators=(",", ":"))
if not a.no_history:
    hf = os.path.join(a.out, "history.json")
    hist = json.load(open(hf)) if os.path.exists(hf) else {}
    hist[asof] = {t["t"]: [t["proj_w"], t["playoffs"], t["title"], t["w"], t["l"]] for t in teams}
    json.dump(hist, open(hf, "w"), separators=(",", ":"))
tb = pd.DataFrame(teams).set_index("t").sort_values("proj_w", ascending=False)
print(tb[["w", "l", "proj_w", "model_w", "pre_w", "playoffs", "title", "offset"]].head(8).to_string())
print(f"pace {pace:.1f} tau {tau:.2f}; players out: {(P.status.str.lower().str.startswith('out')).sum()}; overrides {len(OV)}")
