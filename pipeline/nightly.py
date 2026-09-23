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
# ------------------------------------------------------------------ depth chart (preseason roles; session 15)
# Re-apply the latest RealGM chart dated before the opener (live/{S}/depth, pipeline/depth_fetch.py): players on another
# team's chart move there, rostered players on no chart go to FA (released), everyone's preseason minutes =
# _tier model a[_tier] + b[_tier]*m0 (model/depth_chart.py), team allocation to 19,680. Same rules as the sandbox
# (code/depth_apply.py + project.py), so the chart the state was built with reproduces the state's min_proj.
# After the opener the last pre-opener chart stays in force (the backtested 'late' snapshot); games then take over.
DEPTH = {"chart": None, "moved": [], "dropped": [], "added": []}
_DC = cfg.get("depth")
if _DC and "m0" in PL and os.path.isdir(os.path.join(live, "depth")):
    import glob, depth_chart as DCH
    _first = int(sched.date.min()) if len(sched) else 99999999
    _cut = min(asof_i, _first - 1)
    _fs = [f for f in sorted(glob.glob(os.path.join(live, "depth", "realgm_*.html.gz"))) if int(os.path.basename(f)[7:15]) <= _cut]
    if _fs:
        _R = DCH.load_chart(_fs[-1], S)
        if len(_R):
            _R = DCH.match(_R, PL[["key", "name", "team"]].assign(name=PL["name"].fillna("")))
            _sz = _R.groupby("team").size(); _good = set(_sz[_sz >= _DC["min_team_players"]].index)
            _Rg = _R[_R.team.isin(_good)]; _on = _Rg.dropna(subset=["key"]).set_index("key"); _ex = set(_DC["exempt"])
            PL = PL.copy()
            for i_, k_, t_ in zip(PL.index, PL.key, PL.team):
                if k_ in _ex: continue
                if k_ in _on.index:
                    if _on.team[k_] != t_ and (t_ == "FA" or t_ in _good):
                        (DEPTH["added"] if t_ == "FA" else DEPTH["moved"]).append([PL.at[i_, "name"], t_, _on.team[k_]]); PL.at[i_, "team"] = _on.team[k_]
                elif t_ in _good:
                    DEPTH["dropped"].append([PL.at[i_, "name"], t_]); PL.at[i_, "team"] = "FA"
            _ros = PL.team.isin(_good)
            _tier = DCH.tiers(PL[_ros], _Rg, _ex)
            _w = pd.Series(DCH.want(PL.loc[_ros, "m0"].fillna(0), _tier, _DC["coef"]), index=_tier.index)
            _nl = PL.loc[_ros, "newc"].fillna(False).astype(bool) & (_tier == "L")
            _w[_nl] = _DC["newc_L"]; _w[_tier == "X"] = PL.loc[_tier.index[_tier == "X"], "m0"].fillna(0)
            PL.loc[_ros, "tier"] = _tier.values
            PL.loc[_ros, "min_proj"] = _w.groupby(PL.loc[_ros, "team"]).transform(lambda x: DCH.allocate(x)).values
            PL.loc[PL.team == "FA", "tier"] = None
            DEPTH["chart"] = os.path.basename(_fs[-1])[7:15]
            print(f"depth chart {DEPTH['chart']}: {len(_good)} teams, moved {len(DEPTH['moved'])}, added {len(DEPTH['added'])}, "
                  f"dropped {len(DEPTH['dropped'])}", DEPTH["moved"][:10], DEPTH["added"][:10], DEPTH["dropped"][:10])
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
k = cfg.get("mpg_k", 10); P["mpg_est"] = (k * P.mpg_pre.fillna(10) + P.gp * P.mpg_season) / (k + P.gp)
# expected minutes per team game (backtested rule, code/inseason_minutes.py -> data/inseason.json, session 5):
#   m = w*(RW*last10 + (1-RW)*season_to_date) + (1-w)*preseason, w = n/(n+K), n = team games since joining, DNPs count 0;
#   (optional, off by default: no injury-report entry but absent K_STREAK+ straight -> x0.5); team rescaled to 240 per game.
K_MIN, RW, K_STREAK = cfg.get("min_k", 5), cfg.get("min_rec", 0.5), cfg.get("min_streak", 999)  # streak fallback off: hurt player minutes in 9-season replay; injury reports cover absences
P["n_tm"] = 0.0; P["std_tg"] = 0.0; P["l10_tg"] = 0.0; P["streak"] = 0.0; P["np_c"] = 0.0; P["std_p"] = 0.0; P["l5_p"] = 0.0
if pg is not None and len(pg) and games is not None and len(games):
    tgm = pd.concat([games[["game", "date", "home"]].rename(columns={"home": "t"}),
                     games[["game", "date", "away"]].rename(columns={"away": "t"})]).sort_values(["date", "game"])
    tgm["k"] = tgm.groupby("t").cumcount(ascending=False)          # 0 = team's most recent game
    xq = q.assign(t=q.team.map(abbr_of)).merge(tgm[["game", "t", "k"]], on=["game", "t"])
    xq = xq[xq.t == xq.player.map(last_team)]                      # current team only
    rows_ = {}
    for p_, xp in xq.groupby("player"):
        n_ = int(xp.k.max()) + 1; mk = xp.groupby("k")["min"].sum(); n10 = min(10, n_)
        pl_ = mk[mk > 0].sort_index()                                # played games only, most recent first
        rows_[p_] = (n_, mk.sum() / n_, mk[mk.index < n10].sum() / n10, float(mk.index.min()),
                     len(pl_), pl_.mean() if len(pl_) else 0.0, pl_.iloc[:5].mean() if len(pl_) else 0.0)
    R_ = pd.DataFrame.from_dict(rows_, orient="index", columns=["n", "std", "l10", "streak", "np", "std_p", "l5_p"])
    hit = P.pid.isin(R_.index)
    for c_, cc in (("n_tm", "n"), ("std_tg", "std"), ("l10_tg", "l10"), ("streak", "streak"), ("np_c", "np"), ("std_p", "std_p"), ("l5_p", "l5_p")):
        P.loc[hit, c_] = P.loc[hit, "pid"].map(R_[cc]).values
    ngm = tgm.groupby("t").size()
    P.loc[~hit, "streak"] = P.loc[~hit, "team"].map(ngm).fillna(0).values   # on a roster, not appeared yet
MIN_ALLOC = cfg.get("min_alloc", "rescale")   # 9-season replay: rescale best (trim = worse player minutes early, teams equal)
if MIN_ALLOC == "rescale":   # prior = preseason ALLOCATED minutes; team rescaled to 240 (as inseason_minutes.py)
    pre_tg = (P.min_proj / 82).where(P.min_proj.notna(), P.std_tg) if "min_proj" in P else P.std_tg
    fa0 = PL.set_index("key").team.reindex(P.key).values == "FA"   # free agents (if signed): their unconstrained projection
    pre_tg = pre_tg.where(~fa0, (P.mpg_pre * P.avail_pre).where(P.mpg_pre.notna(), pre_tg))
else:                        # prior = preseason WANTED minutes (before the roster trim); allocate() trims/scales later,
    pre_tg = (P.mpg_pre * P.avail_pre).where(P.mpg_pre.notna(), P.std_tg)   # so day 0 = old nightly (injured -> next man up)
w_ = P.n_tm / (P.n_tm + K_MIN)
P["m_tg"] = w_ * (RW * P.l10_tg + (1 - RW) * P.std_tg) + (1 - w_) * pre_tg.fillna(0)   # FA zeroed after team overrides
tg = P.team.map(played).fillna(0)
P["avail"] = ((cfg.get("avail_k", 20) * P.avail_pre.fillna(.7) + P.gp) / (cfg.get("avail_k", 20) + tg)).clip(0, 1)
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
P["status"] = ""; P["miss"] = 0.0; P["inj"] = ""; P["ret"] = ""
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
        # game-page detail: injury type/side and expected return (factual fields only, no report text)
        det_ = " ".join(str(x) for x in (row.get("detail"), row.get("side")) if pd.notna(x) and str(x).strip() and str(x) != "nan")
        rd_ = pd.to_datetime(row.get("return_date"), errors="coerce")
        P.loc[m, "inj"] = det_; P.loc[m, "ret"] = rd_.strftime("%Y-%m-%d") if pd.notna(rd_) else ""
P["g_rem"] = P.team.map(G_rem).fillna(0)
P.loc[(P.status.fillna("") == "") & (P.streak >= K_STREAK), "m_tg"] *= 0.5
P["m_tg"] = P.m_tg.where(P.team != "FA", 0)
P["m_raw"] = P.m_tg   # before the team rescale (the app redoes the rescale after its own team moves)
if MIN_ALLOC == "rescale":
    P["m_tg"] = (P.m_tg * 240 / P.groupby("team").m_tg.transform("sum").replace(0, np.nan)).fillna(0).where(P.team != "FA", 0)
P["min_want"] = ((P.g_rem - P.miss).clip(lower=0) * P.m_tg).where(P.team != "FA", 0)
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
# tonight's minutes (session 13, code/game_minutes_active_score.py): minutes WHEN PLAYING, not season averages with missed
# games baked in. c = w*(0.5*last5 played + 0.5*season played) + (1-w)*preseason mpg when playing, w = np/(np+GK), np = games
# played for the current team; preseason prior = mpg when playing WITHOUT the age multiplier (better calibrated). Injury report: out 0, day-to-day x0.5; team trimmed to 240 from the bottom of the rotation
# (scaled up if short). Backtest 2014-25: game margin RMSE 13.265 -> 13.225 (t -3.0, both halves), player-game min RMSE 7.53 -> 7.08.
GK = cfg.get("game_k", 5)
_wg = P.np_c / (P.np_c + GK)
_pre_c = (P.mpg_wp if "mpg_wp" in P else P.mpg_pre).fillna(P.mpg_pre).fillna(10.0).clip(0, 40)
P["m_c"] = (_wg * (0.5 * P.l5_p + 0.5 * P.std_p) + (1 - _wg) * _pre_c).where(P.team != "FA", 0).fillna(0)
P["c_lock"] = False
for kk, o in OV.items():   # user edits: g = 0 -> not playing; mpg -> his minutes when playing (kept by the trim)
    m = P.key == kk
    if m.any() and o.get("g", None) == 0: P.loc[m, "m_c"] = 0.0
    elif m.any() and "mpg" in o: P.loc[m, "m_c"] = float(o["mpg"]); P.loc[m, "c_lock"] = True
def trim240(per, lock=None, total=240.0):
    """Bottom-trim the rotation to 240 (smallest expected minutes cut first); scale up if short. Edited (locked) players keep
    their minutes unless the edits alone exceed 240."""
    per = per.clip(lower=0).astype(float); lock = pd.Series(False, index=per.index) if lock is None else lock.reindex(per.index).fillna(False).astype(bool)
    L = per[lock].sum(); out = per.copy()
    if L >= total:
        out[lock] = per[lock] * total / L if L > 0 else 0; out[~lock] = 0; return out
    R = total - L; fr = per[~lock]; U = fr.sum()
    if U <= 0: return out
    if U < R: out[~lock] = (fr * R / U).clip(upper=48); return out
    ex = U - R
    for i in fr.sort_values().index:
        cut = min(ex, out[i]); out[i] -= cut; ex -= cut
        if ex <= 0: break
    return out
def tonight_minutes(t):
    """Expected minutes tonight: out = 0; each day-to-day player plays (full minutes) with prob 0.5 -> average the trimmed
    rotation over those scenarios (a DTD star is not treated as a 17-minute bench player)."""
    m = (P.team == t)
    st = P.status.fillna("").str.lower()[m]
    lk = P.c_lock[m]
    base = P.m_c[m].where(lk | ~(st.str.startswith("out") | st.str.contains("suspen")), 0.0)
    dtd = list(base.index[st.str.contains("day") & (base > 0) & ~lk])[:8]
    if not dtd: return trim240(base, lk)
    acc = base * 0.0
    for mask in range(1 << len(dtd)):
        b = base.copy()
        for j, i in enumerate(dtd):
            if not (mask >> j) & 1: b[i] = 0.0
        acc += trim240(b, lk)
    return acc / (1 << len(dtd))
def game_strength(t):
    per = tonight_minutes(t)
    if per.sum() <= 0: return 0.0
    return float((P.rating[per.index] * per).sum() / 48)
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
    def _rest(t):   # days since the team's previous game (None = no earlier game this season)
        prev = sched[((sched.home == t) | (sched.away == t)) & (sched.date < game_day)].date
        if not len(prev): return None
        return int((pd.Timestamp(str(game_day)) - pd.Timestamp(str(int(prev.max())))).days)
    mt = res[((res.home == gg.home) & (res.away == gg.away)) | ((res.home == gg.away) & (res.away == gg.home))].sort_values("date")
    tonight.append({"rest_h": _rest(gg.home), "rest_a": _rest(gg.away),
                    "meet": [[int(d), h_, a_, int(hp), int(ap_)] for d, h_, a_, hp, ap_ in zip(mt.date, mt.home, mt.away, mt.home_pts, mt.away_pts)],
                    "game": int(gg.game), "time_utc": gg.get("time_utc"), "home": gg.home, "away": gg.away,
                    "margin": round(float(sp_), 1), "p_home": round(float(norm.cdf(sp_ / SIG)), 3),
                    "line": (o.details.iloc[0] if len(o) else None), "total": (float(o.over_under.iloc[0]) if len(o) and pd.notna(o.over_under.iloc[0]) else None)})
# ------------------------------------------------------------------ write
os.makedirs(a.out, exist_ok=True)
F, M = out["forecast"], out["model"]
teams = []
def last10(t):   # [date, opponent, 'H'/'A', team pts, opp pts], newest last
    r_ = res[(res.home == t) | (res.away == t)].sort_values("date").tail(10)
    return [[int(d), (a_ if h_ == t else h_), ("H" if h_ == t else "A"), int(hp if h_ == t else ap_), int(ap_ if h_ == t else hp)]
            for d, h_, a_, hp, ap_ in zip(r_.date, r_.home, r_.away, r_.home_pts, r_.away_pts)]
for i, t in enumerate(SIM.TEAMS):
    teams.append({"t": t, "conf": SIM.conf_of[t], "w": int(wins[t]), "l": int(losses[t]),
                  "proj_w": round(float(F["mean"][i]), 1), "p10": int(F["p10"][i]), "p90": int(F["p90"][i]),
                  "model_w": round(float(M["mean"][i]), 1), "line": (None if pd.isna(TP.line.get(t)) else float(TP.line[t])),
                  "pre_w": round(float(TP.blend_w_pre.get(t, np.nan)), 1),
                  **{k: round(float(F[k][i]), 4) for k in ["playoffs", "playin", "top6", "seed1", "r2", "cf", "finals", "title", "over"]},
                  "margin": round(float(fc_m[t]), 2), "model_margin": round(float(model_m[t]), 2), "offset": round(float(off[t]), 2),
                  "g_rem": int(G_rem[t]), "gp": int(played[t]), "gap": round(float(gap0.get(t, 0)), 3),
                  "dec": round(float(decay[t]), 4), "l10": last10(t)})
pl = P[(P.team != "FA") | (P.gp > 0)].copy()
pl["rating_delta"] = pl.rating - pl.rating_pre
players = [{"k": row.key, "n": row["name"], "t": row.team, "pos": round(float(row.pos_est), 2) if pd.notna(row.pos_est) else 3.0,
            "r": round(float(row.rating), 2), "r_pre": round(float(row.rating_pre), 2), "gp": int(row.gp),
            "mpg": round(float(row.mpg_season), 1), "min_rem": round(float(row.min_rem)), "status": row.status or "",
            # app fields: pre-allocation minutes wanted, mpg estimate, availability, games out
            "mw": round(float(row.min_want)), "me": round(float(row.mpg_est), 2), "mg": round(float(row.m_tg), 3), "mr": round(float(row.m_raw), 3), "mc": round(float(row.m_c), 2), "av": round(float(row.avail), 3),
            "miss": round(float(row.miss), 1), **({"inj": row.inj, "ret": row.ret} if row.status else {})}
           for _, row in pl.sort_values("rating", ascending=False).iterrows() if row.team != "FA" or row.gp > 0]   # every rostered player (injured ones carry their status to the app)
latest = {"asof": asof, "season": S, "games_final": n_done, "games_parsed": n_parsed, "pace": round(pace, 1), "tau": round(float(tau), 2),
          "teams": teams, "players": players, "tonight": tonight, "sims": a.sims,
          "game_day": f"{str(game_day)[:4]}-{str(game_day)[4:6]}-{str(game_day)[6:]}", "depth": DEPTH,
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
