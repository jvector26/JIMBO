import os
"""Build the v2 dashboard: data/*.csv|parquet -> dashboard.html (template code/dash_v2.html, data at __DATA__).
Everything the page recomputes live (ratings under sliders, minutes allocation, team wins, blend, season sim,
backtest) is re-implemented in the page's JS; test_dashboard.js checks it reproduces project.py/simulate.py."""
import json, base64, sys
import numpy as np, pandas as pd
sys.path.insert(0, "code")
from features import COMPONENTS
import simulate as SIM

D = "data"
F = [f for fs in COMPONENTS.values() for f in fs]
COMP_OF = {f: c for c, fs in COMPONENTS.items() for f in fs}
EFF = ["o_3eff", "o_rimeff", "o_mideff", "o_fteff"]
cal = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]
aging = pd.read_csv(f"{D}/aging.csv").set_index("bin").adj
B = json.load(open(f"{D}/blend.json"))
pace = json.load(open(f"{D}/pace_2025.json"))["pace"]
rt = pd.read_parquet(f"{D}/ratings_2025.parquet").set_index("player")
import lineup_shrink; rt = lineup_shrink.apply(rt)  # judgment shrink, thin windows (2026-10-07)
import json as _j, shooting  # same 3PT talent adjustment as project.py
rt = shooting.apply(rt, 2025, 0.5 * _j.load(open(f"{D}/weights_prod.json"))["Wo"]["o_3eff"])
P = pd.concat([pd.read_csv(f"{D}/players_proj_2026.csv"), pd.read_csv(f"{D}/players_fa_2026.csv")], ignore_index=True)
L = pd.read_csv(f"{D}/win_totals_2026.csv").set_index("team")
S = pd.read_csv(f"{D}/sim_2026.csv").set_index("team")
SLOPE = float(cal["slope"])

r4 = lambda v: None if pd.isna(v) else round(float(v), 4)
POS = pd.read_csv(f"{D}/pos_2026.csv")  # newcomer_pos.py: NBA lineup-rank position (1 PG .. 5 C), college model or listed
POS_BY = {(r.name, r.team): round(float(r.pos_est), 2) for r in POS.itertuples()}
# role (depth-chart tier) overrides in the Edit screen: want = a[tier] + b[tier] * m0 (project.py / nightly.py rule)
_rc = json.load(open(f"{D}/min_recal.json")); _c, (_k1, _k2) = _rc["coef"], _rc["knots"]
recal = lambda m: float(np.clip(_c[0] + _c[1] * m + _c[2] * max(m - _k1, 0) + _c[3] * max(m - _k2, 0), 0, 3000))
NEWC_M0 = recal(400 * B["min_age_mult"]["<=22"]); NEWC_L = recal(250.0)
DEPTH = {"coef": json.load(open(f"{D}/depth_tier.json"))["coef"], "newc_L": round(NEWC_L, 2)}
HUB = pd.read_csv(f"{D}/hustle_2026.csv", index_col=0); HJ = json.load(open(f"{D}/hustle.json"))  # hustle = c_lbr*lbr + c_c3*c3
players = []
for _, r in P.iterrows():
    rated = pd.notna(r.player) and r.player in rt.index
    rec = {"k": f"p{int(r.player)}" if pd.notna(r.player) else f"n:{r['name']}|{r.team}",
           "n": r["name"], "t": r.team, "r": r.role, "a": r4(r.age) and round(r.age, 1),
           "mm": round(float(r.min_model), 1), "note": r.note if isinstance(r.note, str) else "",
           "ps": POS_BY.get((r["name"], r.team), 3.0)}
    newc = str(r.note).startswith("no NBA history")
    m0 = NEWC_M0 if newc else (r.min_model_nodepth if pd.notna(r.get("min_model_nodepth")) else r.min_model)
    rec["m0"] = round(float(m0), 1)
    if newc: rec["nc"] = 1
    dt = r.get("depth_tier")
    if isinstance(dt, str) and r.team != "FA":
        rec["dt"] = "L" if (newc and dt == "X" and r.role == "L") else dt   # L newcomers are stored as X (not tiered)
    if rated:
        x = rt.loc[r.player]
        rec["f"] = [r4(x[f"f:{f}"]) for f in F]
        rec["lo"], rec["ld"] = r4(x.res_o), r4(x.res_d)
        rec["ag"] = r4(r.aging / SLOPE)                          # raw aging adj (pre-slope)
        if r.returner: rec["rt"] = r4(r.returner)                # returner adj (post-slope, returner.json)
        if pd.notna(r.get("hustle")) and r.hustle: rec["hu"] = r4(r.hustle)  # hustle.py (post-slope)
        if pd.notna(r.get("hustle")) and r.hustle and r.player in HUB.index:  # skill drill-down: loose balls / 3PT contests split
            rec["hb"] = [r4(HJ["c_lbr"] * HUB.lbr[r.player]), r4(HJ["c_c3"] * HUB.c3[r.player])]
        rec["m25"] = round(float(x.min_2025))
        # editor defaults: games and minutes-per-game that multiply to the model's season minutes
        g = float(r.exp_games) if pd.notna(r.exp_games) else 70.0
        rec["g"] = round(g, 1); rec["mpg"] = round(float(r.min_model) / g, 2) if g > 0 else 0
    else:
        rec["fx"] = r4(r.rating)                                  # fixed newcomer / rookie value (post-slope)
        rec["g"] = 70.0; rec["mpg"] = round(float(r.min_model) / 70, 2)
    players.append(rec)

# backtest columns (per-game feature diffs, 2014-15..2025-26) as base64 float32
bt = pd.read_parquet(f"{D}/backtest_components.parquet")
bt = bt.merge(pd.read_parquet(f"{D}/bt_split.parquet"), on=["game", "home_team", "away_team"], how="left", validate="1:1")  # bt_split.py
assert bt.res_o.notna().all() and (bt.res_o + bt.res_d - bt.lineup_impact).abs().max() < 0.01
b64 = lambda a: base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()
BT = {"n": len(bt), "margin": b64(bt.margin), "lineup_o": b64(bt.res_o), "lineup_d": b64(bt.res_d), "unrated": b64(bt.unrated),
      "cols": {f: b64(bt[f"f:{f}"]) for f in F}}

sched = SIM.make_schedule(np.random.default_rng(2026))            # one fixed NBA-formula schedule for the page sim
if os.path.exists(f"{D}/dash_sched_2026.json"):  # session 17: make_schedule(seed 2026) is not reproducible across environments;
    sched = np.array(json.load(open(f"{D}/dash_sched_2026.json"))).reshape(-1, 2)  # pin the published one
teams = [{"t": t, "conf": SIM.conf_of[t], "div": SIM.div_of[t], "line": float(L.line[t]),
          "mkt": round(float(S.mkt_mean[t]), 2)} for t in SIM.TEAMS]

meta = {"feats": F, "comp_of": COMP_OF, "eff": EFF, "components": list(COMPONENTS),
        "depth": DEPTH, "slope": SLOPE, "lw_o": 0.6, "lw_d": 0.8, "sig": SIM.SIG, "hca": SIM.HCA, "pace": pace,
        "blend": {"wm": B["w_model"], "wv": B["w_vegas"], "wa": B["w_age"], "tau": B["tau_blend"], "tau_model": B["tau_model"], "curve": json.load(open(f"{D}/blend_curve.json"))},
        "asof": "2026-09-19", "lines_src": "BetMGM, about Sep 11",
        "py_sim": {t: {"w": round(float(S.blend_mean_w[t]), 2), "po": round(float(S.blend_playoffs[t]), 4),
                       "title": round(float(S.blend_title[t]), 4)} for t in SIM.TEAMS}}
# ---- player view: career box-score contributions by season (same lens as the projection window: 0.5 x production W,
# each season alone, centered on that season's possession-weighted league average) + rookie college profiles
Wp = json.load(open(f"{D}/weights_prod.json")); Wwin = {x: 0.5 * {**Wp["Wo"], **Wp["Wd"]}[x] for x in F}  # = fit_window W
fe = pd.read_parquet(f"{D}/features.parquet"); fe = fe[fe["min"] > 0].copy()
for x in F:
    wt = fe.poss_off if x.startswith("o_") else fe.poss_def
    fe[x] = fe[x] * Wwin[x]
    fe[x] -= (fe[x] * wt).groupby(fe.season).transform("sum") / wt.groupby(fe.season).transform("sum")
gp = pd.read_parquet(f"{D}/player_seasons.parquet").groupby(["season", "player"]).gp.sum()
want = {int(k[1:]) for k in (p["k"] for p in players) if k.startswith("p")}
HIST = {}
LH = pd.read_parquet(f"{D}/lineup_hist.parquet").set_index(["season", "player"])  # lineup_history.py (raw, pre-weight)
L1O, L1D, LWO, LWD = LH.one_o.to_dict(), LH.one_d.to_dict(), LH.win_o.to_dict(), LH.win_d.to_dict()
q = lambda M_, s, pid: None if pd.isna(M_.get((s, pid), np.nan)) else int(round(M_[(s, pid)] * 1000))
for pid, h in fe[fe.player.isin(want)].sort_values("season").groupby("player"):
    HIST[f"p{int(pid)}"] = {"s": h.season.astype(int).tolist(), "tm": h.teams.fillna("").tolist(),
        "a": [None if pd.isna(v) else round(float(v), 1) for v in h.age], "g": [int(gp.get((s, pid), 0)) for s in h.season],
        "mn": [int(round(v)) for v in h["min"]],
        "l1o": [q(L1O, s, pid) for s in h.season], "l1d": [q(L1D, s, pid) for s in h.season],
        "lwo": [q(LWO, s, pid) for s in h.season], "lwd": [q(LWD, s, pid) for s in h.season], "v": np.round(h[F].to_numpy() * 1000).astype(int).ravel().tolist()}
# ---- skill drill-down (user request 2026-10-18): last season's actual stats behind each box-score skill
RAWC = ["fg3m", "fg3a", "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "ftm", "fta", "fgm", "astd_fgm", "ast", "tov", "oreb",
        "dreb", "blk", "stl", "pf"]
LASTS = int(fe.season.max())
_ps = pd.read_parquet(f"{D}/player_seasons.parquet"); _ps = _ps[_ps.season == LASTS].set_index("player")
_fr = pd.read_parquet(f"{D}/features.parquet"); _fr = _fr[(_fr.season == LASTS) & (_fr["min"] > 0)].set_index("player")
_j = _ps[RAWC].join(_fr[["poss_off", "poss_def", "d_opprim", "d_oppxq"]], how="inner")
_DEFC = {"dreb", "blk", "stl", "pf"}
LG = {c: round(float(_j[c].sum() / _j["poss_def" if c in _DEFC else "poss_off"].sum() * 100), 3) for c in RAWC}
LG.update(p3=round(float(_j.fg3m.sum() / _j.fg3a.sum()), 4), prim=round(float(_j.rim_fgm.sum() / _j.rim_fga.sum()), 4),
          pmid=round(float(_j.mid_fgm.sum() / _j.mid_fga.sum()), 4), pft=round(float(_j.ftm.sum() / _j.fta.sum()), 4))
for pid, x in _j.iterrows():
    k = f"p{int(pid)}"
    if k in HIST:
        HIST[k]["raw"] = [int(x[c]) for c in RAWC] + [int(round(x.poss_off)), int(round(x.poss_def)), round(float(x.d_opprim), 2), round(float(x.d_oppxq), 2)]
SC = pd.read_parquet(f"{D}/ncaa_scores.parquet"); RV = json.load(open(f"{D}/rookie_value.json"))
allc = SC.cscore.dropna(); ROOK = {}
d26 = SC[SC.year == 2026].set_index("pick")
MT = pd.read_parquet(f"{D}/ncaa_draft_match.parquet"); MT = MT[MT.year == 2026].drop_duplicates("pick").set_index("pick")
PG = (MT.pon / MT.g / 100).where(MT.g > 0)   # on-court possessions per game / 100: per-100 rate x PG = per game
for p in players:
    note = p.get("note") or ""
    if note.startswith("2026 rookie, pick "):
        pk = int(__import__("re").search(r"pick (\d+)", note).group(1)); x = d26.loc[pk] if pk in d26.index else None
        rec = {"pick": pk}
        if x is not None and bool(x.college) and pd.notna(x.cscore):
            rec.update(college=True, cs=round(float(x.cscore), 3), pct=round(float((allc < x.cscore).mean()) * 100),
                       capped=bool(abs(x.cscore - RV["col_mean"]) > RV["col_cap"]),
                       st={k: round(float(x[f"x_{k}"]), 3) for k in ["pts", "ts", "tpa", "tp", "ft", "ftr", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "usg", "mpg", "srs", "age", "h"]})
            if pk in PG.index and pd.notna(PG[pk]): rec["st"]["pg"] = round(float(PG[pk]), 4); rec["st"]["g"] = int(MT.g[pk])
        else:
            rec.update(college=False)
        ROOK[p["k"]] = rec
meta["raw"] = {"y": LASTS, "cols": RAWC + ["poss_off", "poss_def", "d_opprim", "d_oppxq"], "lg": LG}  # skill drill-down
meta["po"] = SIM.PO_META  # playoff-strength curve + series home court (session 21); the page's poDelta() reads it
meta["rv"] = RV; meta["decay"] = [0.3, 0.6, 1.0]; meta["box_decay"] = [0.1, 0.4, 1.0]
meta["aging_bins"] = {str(k): round(float(v), 3) for k, v in aging.items()}
data = {"meta": meta, "players": players, "teams": teams, "bt": BT, "sched": sched.ravel().tolist(), "hist": HIST, "rook": ROOK}
js = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
html = open("code/dash_v2.html").read().replace("__DATA__", js)
open("dashboard.html", "w").write(html)
print(f"players {len(players)} (FA {sum(p['t']=='FA' for p in players)}), payload {len(js)/1e6:.2f} MB")
