import numpy as np, pandas as pd
from math import erf, sqrt
from features import COMPONENTS
D = "data"
R = pd.read_csv(f"{D}/rosters_2026.csv")
rt = pd.read_parquet(f"{D}/ratings_2025.parquet").set_index("player")
import lineup_shrink; rt = lineup_shrink.apply(rt)  # judgment shrink, thin windows (2026-10-07)
import json as _j, shooting  # 3PT talent: career 3P% + FT% prior, 1.5x (shooting_bt.py)
rt = shooting.apply(rt, 2025, 0.5 * _j.load(open(f"{D}/weights_prod.json"))["Wo"]["o_3eff"])
# free-agent pool: rated players on no 2026-27 roster (projected as if signed; excluded from team totals)
_fa = rt[~rt.index.isin(R.player.dropna()) & (rt.min_2025 >= 100)]
R = pd.concat([R, pd.DataFrame({"team": "FA", "role": "F", "name": _fa["name"].values, "player": _fa.index.values})], ignore_index=True)
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
cal = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]
aging = pd.read_csv(f"{D}/aging.csv").set_index("bin").adj
import json
MIN_AGE = json.load(open(f"{D}/blend.json"))["min_age_mult"]  # actual/projected minutes by age, preseason backtest
SLOPE, UNRATED = cal["slope"], cal["unrated"]
RET = json.load(open(f"{D}/returner.json"))  # return from (nearly) full missed season
HU = pd.read_csv(f"{D}/hustle_2026.csv", index_col=0).hustle  # hustle.py: loose balls + 3PT contests (defense #7), post-slope
LW_O, LW_D = 0.6, 0.8  # lineup-impact weights, offense / defense (bt_split.py, 2026-09-20)
bm, ba, brk = np.load(f"{D}/min_bm.npy"), np.load(f"{D}/min_ba.npy"), np.load(f"{D}/rookie_min_coef.npy")
draft = pd.read_parquet(f"{D}/draft_r1.parquet")
d26 = draft[draft.year == 2026].set_index("player")
gm = pd.read_parquet(f"{D}/games_all.parquet"); g25 = gm[gm.season == 2025]
wp25 = pd.concat([g25.assign(a=g25.htm, w=g25.home_pts > g25.away_pts), g25.assign(a=g25.vtm, w=g25.away_pts > g25.home_pts)]).groupby("a").w.mean()
name2draft = {n: pid for pid, n in zip(d26.index, d26["name"])}

def age_bin(a):
    for lo, hi, lab in [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
                        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]:
        if lo <= a <= hi: return lab
def age_bin5(a):
    return "<=22" if a <= 22 else "23-25" if a <= 25 else "26-28" if a <= 28 else "29-31" if a <= 31 else "32-34" if a <= 34 else "35+"
last = ps.sort_values("season").groupby("player").last()
m_by = ps.pivot_table(index="player", columns="season", values="min", aggfunc="sum").fillna(0)
gs_by = ps.pivot_table(index="player", columns="season", values="gs", aggfunc="sum").fillna(0)
gp_by = ps.pivot_table(index="player", columns="season", values="gp", aggfunc="sum").fillna(0)

out = []
comps = list(COMPONENTS)
for _, r in R.iterrows():
    p = r.player
    rec = {"team": r.team, "name": r["name"], "player": p, "role": r.role, "depth_tier": r.get("tier", np.nan)}
    has = pd.notna(p) and p in rt.index
    if has:
        x = rt.loc[p]
        age = last.loc[p, "age"] + (2026 - last.loc[p, "season"]) if pd.notna(last.loc[p, "age"]) else np.nan
        ag = aging.get(age_bin(age), 0.0) if pd.notna(age) else 0.0
        for c in comps: rec[c] = SLOPE * x[c]
        rec["lineup_impact"] = SLOPE * (LW_O * x.res_o + LW_D * x.res_d)
        rec["aging"] = SLOPE * ag
        rec["off"] = SLOPE * (x["off"] - (1 - LW_O) * x.res_o + ag / 2); rec["def"] = SLOPE * (x["def"] - (1 - LW_D) * x.res_d + ag / 2)
        rec["age"] = age
        g24, g25 = gp_by.loc[p].get(2024, 0), gp_by.loc[p].get(2025, 0)
        is_ret = g24 >= RET["min_gp"] and m_by.loc[p].get(2024, 0) / max(g24, 1) >= RET["min_mpg"] and g25 / 82 < RET["max_avail"]
        rec["returner"] = RET["adj"] if is_ret else 0.0
        rec["off"] += rec["returner"] / 2; rec["def"] += rec["returner"] / 2
        rec["hustle"] = float(HU.get(p, 0.0)); rec["def"] += rec["hustle"]
        yrs = [y for y in (2025, 2024, 2023, 2022) if gp_by.loc[p].get(y, 0) > 0]
        injury_year = gp_by.loc[p].get(2025, 0) == 0 and len(yrs) > 0
        use = (yrs + [None] * 3)[:3] if yrs else [None] * 3
        def S(y, what):
            if y is None: return None
            g, m, gs = gp_by.loc[p].get(y, 0), m_by.loc[p].get(y, 0), gs_by.loc[p].get(y, 0)
            return {"mpg": m / g, "avail": min(g / 82, 1), "gsr": gs / g}[what]
        mpg1 = S(use[0], "mpg") if use[0] else 12.0
        mpg2 = S(use[1], "mpg") if use[1] else mpg1
        mpg3 = S(use[2], "mpg") if use[2] else mpg2
        gsr1 = S(use[0], "gsr") if use[0] else 0.0
        av = [S(y, "avail") if y else None for y in use]
        av1 = 0.0 if injury_year else (av[0] if av[0] is not None else 0.62)
        a = min(max(age if pd.notna(age) else 27, 19), 40)
        # model A: learned from history (availability counts only from seasons as a regular, 20+ mpg)
        mA = float(np.clip(bm @ [1, mpg1, mpg2, mpg3, gsr1 * mpg1, av1 * mpg1, max(a - 32, 0), max(24 - a, 0)], 0, 40))
        regs = [(av1 if k == 0 else (av[k] or 0.0), (mpg1, mpg2, mpg3)[k] >= 20 and (k == 0 or av[k] is not None)) for k in range(3)]
        xa = [1]
        for v, r_ in regs: xa += [v if r_ else 0.0, float(r_)]
        xa += [mpg1, max(a - 30, 0)]
        aA = float(np.clip(np.dot(ba, xa), 0.05, 1))
        # model B: recency-weighted rule
        mB = 0.75 * mpg1 + 0.25 * mpg2 - max(a - 33, 0)
        aB = 0.4 * av1 + 0.25 * (av[1] if av[1] is not None else av1) + 0.15 * (av[2] if av[2] is not None else av1) + 0.2 * 0.62
        rec["mpg_when_playing"] = 0.5 * mA + 0.5 * mB
        rec["exp_games"] = 82 * (0.5 * aA + 0.5 * aB)
        rec["min_age_mult"] = MIN_AGE[age_bin5(age)] if pd.notna(age) else 1.0
        rec["min_model"] = float(np.clip(rec["mpg_when_playing"] * rec["exp_games"] * rec["min_age_mult"], 0, 3000))
        rec["note"] = "missed all of 2025-26" if injury_year else ""
        if is_ret: rec["note"] = (rec["note"] + "; " if rec["note"] else "") + f"returner ({int(g25)} gp in 2025-26)"
    else:
        for c in comps: rec[c] = 0.0
        rec["lineup_impact"] = SLOPE * UNRATED; rec["aging"] = 0.0; rec["returner"] = 0.0; rec["hustle"] = 0.0
        rec["off"] = SLOPE * UNRATED / 2; rec["def"] = SLOPE * UNRATED / 2
        rec["age"] = np.nan
        pid = name2draft.get(r["name"])
        if pid is not None:  # 2026 first-round pick: minutes from draft slot, team quality, draft age
            dd = d26.loc[pid]
            RV = json.load(open(f"{D}/rookie_value.json"))  # college score if NCAA player, else pick+age (young_prior.py, ncaa_model.py)
            SC = pd.read_parquet(f"{D}/ncaa_scores.parquet"); cs = SC[(SC.year == 2026) & (SC.pick == dd.pick)].cscore
            if len(cs) and pd.notna(cs.iloc[0]):
                val = RV["col_a"] + RV["col_b"] * float(np.clip(cs.iloc[0] - RV["col_mean"], -RV["col_cap"], RV["col_cap"]))
            else:
                val = RV["rk"] + RV["rk_lp"] * (np.log(dd.pick) - 2) + RV["rk_age"] * ((dd.age if pd.notna(dd.age) else 20) - 20)
            rec["lineup_impact"] = SLOPE * val; rec["off"] = rec["def"] = SLOPE * val / 2
            rec["min_model"] = float(np.clip(brk @ [1, np.log(dd.pick), wp25.get(r.team, 0.5), dd.age if pd.notna(dd.age) else 19.5], 50, 2600)) * MIN_AGE["<=22"]
            rec["age"] = dd.age; rec["note"] = f"2026 rookie, pick {int(dd.pick)}"
            rec["mpg_when_playing"] = np.nan; rec["exp_games"] = np.nan
        else:
            # with a depth-chart tier: backtest convention (preseason_backtest.py newcomer 400 x age mult), the tier model
            # below does the rest (newcomer S/R -> ~760/600); old role heuristic only without a tier
            # L newcomers keep 250 (NOT tiered below): the backtest's L newcomers (act ~414/82) only include players who
            # appeared, while production L newcomers are mostly camp invitees / two-ways, many never play
            if r.get("tier") in ("S", "R", "N"): rec["min_model"] = 400.0 * MIN_AGE["<=22"]
            else: rec["min_model"] = {"S": 2000, "R": 1200, "L": 250}.get(r.role, 700); rec["depth_tier"] = "X" if r.get("tier") == "L" else rec["depth_tier"]
            rec["mpg_when_playing"] = np.nan; rec["exp_games"] = np.nan
            rec["note"] = "no NBA history: newcomer value"
    rec["rating"] = rec["off"] + rec["def"]
    out.append(rec)
P = pd.DataFrame(out)
# season-minutes recalibration (min_recal_fit.py, session 13): high projections were too high (mostly games), so map
# min_model -> expected actual minutes; applied as a games factor (mpg when playing unchanged).
_RC = json.load(open(f"{D}/min_recal.json")); _c = _RC["coef"]; _k1, _k2 = _RC["knots"]
P["min_model_raw"] = P.min_model
P["min_model"] = (_c[0] + _c[1] * P.min_model_raw + _c[2] * (P.min_model_raw - _k1).clip(lower=0) + _c[3] * (P.min_model_raw - _k2).clip(lower=0)).clip(lower=0, upper=3000)
# expected games = projected minutes / mpg when playing, so the age multiplier and this recalibration both sit in GAMES
# (game-level check: mpg when playing is calibrated without the age multiplier)
P["exp_games"] = (P.min_model / P.mpg_when_playing.replace(0, np.nan)).clip(upper=82).where(P.mpg_when_playing.notna(), P.exp_games)
# RealGM depth-chart tier (code/depth_apply.py -> rosters_2026.csv 'tier'; backtest code/depth_test.py, fit depth_fit.py,
# session 14-15): want = a[tier] + b[tier] * m0 with m0 = the recalibrated projection above; S/R/L = starter / rotation /
# limited on his team's chart, X = exempt (unchanged), FA unchanged. The change sits in minutes per game (games unchanged).
_DT = json.load(open(f"{D}/depth_tier.json"))["coef"]
P["min_model_nodepth"] = P.min_model
_t = P.depth_tier.where(P.team != "FA")
_has = _t.isin(list(_DT))
P.loc[_has, "min_model"] = [float(np.clip(_DT[t][0] + _DT[t][1] * m, 0, 3000)) for t, m in zip(_t[_has], P.min_model[_has])]
_TN = {"S": "starter", "R": "rotation", "L": "limited minutes"}
P.loc[_has, "note"] = [(n + "; " if isinstance(n, str) and n else "") + f"depth chart: {_TN.get(t, 'not listed')}" for n, t in zip(P.note[_has], _t[_has])]
# team minutes must total 82 games x 240. Crowded rosters lose minutes from the bottom of the rotation first;
# thin rosters scale everyone up. Nobody exceeds 3,000.
def allocate(g):
    m = g.min_model.clip(upper=3000).copy()
    excess = m.sum() - 19680
    if excess > 0:
        for i in m.sort_values().index:
            cut = min(excess, m[i]); m[i] -= cut; excess -= cut
            if excess <= 0: break
    else:
        for _ in range(10):
            room = m < 3000
            m[room] *= (19680 - m[~room].sum()) / m[room].sum()
            m = m.clip(upper=3000)
    return m
FA = P[P.team == "FA"].copy(); P = P[P.team != "FA"].copy()
P["min_proj"] = P.groupby("team", group_keys=False).apply(allocate)
P["mpg_proj"] = P.min_proj / 82
T = P.assign(w=P.rating * P.min_proj / 3936, wo=P.off * P.min_proj / 3936, wd=P["def"] * P.min_proj / 3936) \
     .groupby("team")[["w", "wo", "wd"]].sum()
T.columns = ["net", "off", "def"]
T[["net", "off", "def"]] -= T[["net", "off", "def"]].mean()  # league is zero-sum
import os
if os.path.exists(f"{D}/stints_2025.parquet"):
    st = pd.read_parquet(f"{D}/stints_2025.parquet")
    pace = sum(st[f"fg2a{z}"] + st[f"fg3a{z}"] - st[f"oreb{z}"] + st[f"tov{z}"] + 0.44 * st[f"fta{z}"] for z in "AB").sum() / 2 / st.game.nunique()
    json.dump({"pace": float(pace)}, open(f"{D}/pace_2025.json", "w"))
else:
    pace = json.load(open(f"{D}/pace_2025.json"))["pace"]  # cached from stints_2025
T["margin"] = T.net * pace / 100
SIG = 12.8
T["wins"] = 82 * T.margin.map(lambda m: 0.5 * (1 + erf(m / SIG / sqrt(2))))
T = T.sort_values("wins", ascending=False)
FA["min_proj"] = FA.min_model; FA["mpg_proj"] = FA.min_proj / 82
P.to_csv(f"{D}/players_proj_2026.csv", index=False); FA.to_csv(f"{D}/players_fa_2026.csv", index=False); T.to_csv(f"{D}/teams_proj_2026.csv")
print(f"pace {pace:.1f}; sum of projected wins {T.wins.sum():.0f} (should be ~1230)")
print(T.round(1).to_string())
