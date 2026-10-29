"""True preseason backtest: for each season S (2014-15 .. 2025-26), project every team as of October using only
information available then, and compare with actual results and the Vegas preseason win total.

Mirrors project.py (same rating recipe, lineup weight, aging, rookie values, minutes model, team-minute allocation).
Leak-free pieces:  component weights W (fit 1996-2013 only), player ratings (seasons S-3..S-1 only),
                   minutes model and rookie-minutes model (refit on seasons < S each year).
Known leaks (small): calibration slope / aging / rookie & unrated values come from calib.csv + aging.csv, fit on
                   2014-15..2025-26 games (13 parameters on 13k games). Opening rosters are inferred from who played
                   for whom early in S, so players who missed the ENTIRE season are absent (a real preseason
                   projection would have had to guess about them) -> flatters the model slightly.
"""
import json, numpy as np, pandas as pd, sys
from scipy.stats import norm
sys.path.insert(0, "/home/claude/code")
import impact as I
from features import COMPONENTS
D = "/home/claude/data"
LW, SIG = 0.6, 12.8
import os as _os
LW_O, LW_D = float(_os.environ.get("LW_O", 0.6)), float(_os.environ.get("LW_D", 0.8))
cal = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]
import os
TAG = os.environ.get("BT_TAG", "")
aging = pd.read_csv(os.environ.get("AGING_CSV", f"{D}/aging.csv")).set_index("bin").adj
RET = json.load(open(f"{D}/returner.json"))
SLOPE, UNRATED = float(os.environ.get("SLOPE_OVR", cal["slope"])), cal["unrated"]
W = json.load(open(f"{D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
Wd["d_dreb"] *= float(os.environ.get("DREB_MULT", 1.0))  # weights_bt.json already has dreb x0.5 baked in (2026-09-20)
ps = pd.read_parquet(f"{D}/player_seasons.parquet")
gm = pd.read_parquet(f"{D}/games_all.parquet")
draft = pd.read_parquet(f"{D}/draft_r1.parquet")
ng = gm.groupby("season").size() * 2 / 30  # games per team each season
id2abbr = pd.read_parquet(f"{D}/games_2025.parquet").drop_duplicates("home_team").set_index("home_team").htm.to_dict()
BINS = [(0, 21.99, "<=21"), (22, 23.99, "22-23"), (24, 25.99, "24-25"), (26, 27.99, "26-27"), (28, 29.99, "28-29"),
        (30, 31.99, "30-31"), (32, 33.99, "32-33"), (34, 35.99, "34-35"), (36, 99, "36+")]
age_bin = lambda a: next(lab for lo, hi, lab in BINS if lo <= a <= hi)

# ---------------- minutes model training table (same design as minutes3.py) ----------------
pz = ps[["season", "player", "min", "gp", "gs", "age"]].copy()
pz["mpg"] = pz["min"] / pz.gp; pz["avail"] = (pz.gp / pz.season.map(ng)).clip(upper=1); pz["gsr"] = pz.gs / pz.gp
def lag(k):
    x = pz[["season", "player", "mpg", "avail", "gsr", "age", "gp"]].copy(); x["season"] += k
    return x.rename(columns={c: f"{c}{k}" for c in ["mpg", "avail", "gsr", "age", "gp"]})
dm = pz.merge(lag(1), on=["season", "player"]).merge(lag(2), on=["season", "player"], how="left").merge(lag(3), on=["season", "player"], how="left")
dm = dm.dropna(subset=["age1"]); dm["age_now"] = dm.age1 + 1
for k in (2, 3):
    dm[f"mpg{k}"] = dm[f"mpg{k}"].fillna(dm.mpg1); dm[f"gsr{k}"] = dm[f"gsr{k}"].fillna(dm.gsr1)
    dm[f"reg{k}"] = (dm[f"mpg{k}"] >= 20) & dm[f"avail{k}"].notna()
dm["reg1"] = dm.mpg1 >= 20
def Xm(df):
    a = df.age_now.clip(19, 40)
    return np.c_[np.ones(len(df)), df.mpg1, df.mpg2, df.mpg3, df.gsr1 * df.mpg1, df.avail1 * df.mpg1, np.clip(a - 32, 0, None), np.clip(24 - a, 0, None)]
def Xa(df):
    a = df.age_now.clip(19, 40); cols = [np.ones(len(df))]
    for k in (1, 2, 3):
        r = df[f"reg{k}"]; cols += [np.where(r, df[f"avail{k}"], 0.0), r.astype(float)]
    return np.column_stack(cols + [df.mpg1, np.clip(a - 30, 0, None)])

# rookie minutes table (same as rookies.py)
res = pd.concat([gm.assign(abbr=gm.htm, w=gm.home_pts > gm.away_pts), gm.assign(abbr=gm.vtm, w=gm.away_pts > gm.home_pts)])
wp = res.groupby(["season", "abbr"]).w.mean()
wp_id = pd.concat([gm.assign(t=gm.home_team, w=gm.home_pts > gm.away_pts), gm.assign(t=gm.away_team, w=gm.away_pts > gm.home_pts)]).groupby(["season", "t"]).w.mean()

def pace_of(season):
    st = pd.read_parquet(f"{D}/stints_{season}.parquet")
    poss = sum(st[f"fg2a{z}"] + st[f"fg3a{z}"] - st[f"oreb{z}"] + st[f"tov{z}"] + 0.44 * st[f"fta{z}"] for z in "AB")
    return poss.sum() / 2 / st.game.nunique()

def opening_rosters(S):
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet")[["game", "player", "team", "min", "date"]]
    pg = pg[pg["min"] > 0]
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team"]]
    tg = pd.concat([g.assign(team=g.home_team), g.assign(team=g.away_team)]).sort_values(["date", "game"])
    tg["gnum"] = tg.groupby("team").cumcount()
    pg = pg.merge(tg[["game", "team", "gnum"]], on=["game", "team"])
    first = pg.sort_values(["date", "game"]).drop_duplicates("player")
    prev = pd.read_parquet(f"{D}/player_games_{S - 1}.parquet").sort_values(["date", "game"]).drop_duplicates("player", keep="last").set_index("player").team
    keep = (first.gnum < 15) | (first.player.map(prev) == first.team)
    r = first[keep][["player", "team"]].copy()
    r["abbr"] = r.team.map(id2abbr)
    return r

def project_season(S):
    N = S - 1
    # ratings as of end of N (window N-2..N), backtest weights, re-centered on season-N minutes (as production)
    _bd = os.environ.get("BOX_DECAY", "0.1,0.4,1"); r = I.fit_window(N, Wo, Wd, lam=2000, box_decay=tuple(map(float, _bd.split(","))) if _bd else None)
    import shooting; r = shooting.apply(r, N, Wo["o_3eff"], index_col="player")
    f = I.feats(); last_f = f[f.season == N].set_index("player")
    w = r.player.map(last_f["min"]).fillna(0)
    oc = [c for c, v in COMPONENTS.items() if v[0].startswith("o_")]; dc = [c for c, v in COMPONENTS.items() if v[0].startswith("d_")]
    for side, comps, rc in (("off", oc, "res_o"), ("def", dc, "res_d")):
        r[rc] -= (r[side] * w).sum() / w.sum()
        r[side] = r[comps].sum(axis=1) + r[rc]
    r = r.set_index("player")
    # minutes models refit on data before S
    tr = dm[dm.season.between(2000, N)]
    bm = np.linalg.lstsq(Xm(tr) * np.sqrt(tr.gp.values)[:, None], tr.mpg * np.sqrt(tr.gp.values), rcond=None)[0]
    ba = np.linalg.lstsq(Xa(tr), tr.avail, rcond=None)[0]
    h = draft[draft.year.between(1997, N)][["year", "pick", "player", "team", "age"]].rename(columns={"year": "season"})
    h = h.merge(ps[["season", "player", "min"]], on=["season", "player"], how="left").fillna({"min": 0})
    h["prev_wp"] = [wp.get((s - 1, a), np.nan) for s, a in zip(h.season, h.team)]
    h = h.dropna(subset=["prev_wp"]); h["min82"] = h["min"] * 82 / h.season.map(ng)
    brk = np.linalg.lstsq(np.c_[np.ones(len(h)), np.log(h.pick), h.prev_wp, h.age.fillna(h.age.median())], h.min82, rcond=None)[0]
    dS = draft[draft.year == S].set_index("player")

    R = opening_rosters(S)
    hist = ps[ps.season <= N]
    lastrow = hist.sort_values("season").groupby("player").last()
    gp_by = hist.pivot_table(index="player", columns="season", values="gp", aggfunc="sum").fillna(0)
    m_by = hist.pivot_table(index="player", columns="season", values="min", aggfunc="sum").fillna(0)
    gs_by = hist.pivot_table(index="player", columns="season", values="gs", aggfunc="sum").fillna(0)
    out = []
    for p, tid, ab in R.itertuples(index=False):
        rec = {"player": p, "team": ab}
        if p in r.index and p in gp_by.index:
            x = r.loc[p]
            age = lastrow.loc[p, "age"] + (S - lastrow.loc[p, "season"]) if pd.notna(lastrow.loc[p, "age"]) else np.nan
            ag = aging.get(age_bin(age), 0.0) if pd.notna(age) else 0.0
            rec["rating"] = SLOPE * (x["off"] + x["def"] - (1 - LW_O) * x.res_o - (1 - LW_D) * x.res_d + ag)
            g1 = gp_by.loc[p].get(N - 1, 0)  # returner from a (nearly) full missed season N (returner.json)
            rec["returner"] = RET["adj"] if (g1 >= RET["min_gp"] and m_by.loc[p].get(N - 1, 0) / max(g1, 1) >= RET["min_mpg"]
                                             and gp_by.loc[p].get(N, 0) / ng[N] < RET["max_avail"]) else 0.0
            rec["rating"] += rec["returner"]
            yrs = [y for y in (N, N - 1, N - 2, N - 3) if gp_by.loc[p].get(y, 0) > 0]
            injury_year = gp_by.loc[p].get(N, 0) == 0 and len(yrs) > 0
            use = (yrs + [None] * 3)[:3] if yrs else [None] * 3
            def St(y, what):
                g, m, gs = gp_by.loc[p].get(y, 0), m_by.loc[p].get(y, 0), gs_by.loc[p].get(y, 0)
                return {"mpg": m / g, "avail": min(g / ng[y], 1), "gsr": gs / g}[what]
            mpg1 = St(use[0], "mpg") if use[0] else 12.0
            mpg2 = St(use[1], "mpg") if use[1] else mpg1
            mpg3 = St(use[2], "mpg") if use[2] else mpg2
            gsr1 = St(use[0], "gsr") if use[0] else 0.0
            av = [St(y, "avail") if y else None for y in use]
            av1 = 0.0 if injury_year else (av[0] if av[0] is not None else 0.62)
            a = min(max(age if pd.notna(age) else 27, 19), 40)
            mA = float(np.clip(bm @ [1, mpg1, mpg2, mpg3, gsr1 * mpg1, av1 * mpg1, max(a - 32, 0), max(24 - a, 0)], 0, 40))
            regs = [(av1 if k == 0 else (av[k] or 0.0), (mpg1, mpg2, mpg3)[k] >= 20 and (k == 0 or av[k] is not None)) for k in range(3)]
            xa = [1]
            for v, rr in regs: xa += [v if rr else 0.0, float(rr)]
            xa += [mpg1, max(a - 30, 0)]
            aA = float(np.clip(np.dot(ba, xa), 0.05, 1))
            mB = 0.75 * mpg1 + 0.25 * mpg2 - max(a - 33, 0)
            aB = 0.4 * av1 + 0.25 * (av[1] if av[1] is not None else av1) + 0.15 * (av[2] if av[2] is not None else av1) + 0.2 * 0.62
            rec["min_model"] = float(np.clip((0.5 * mA + 0.5 * mB) * 82 * (0.5 * aA + 0.5 * aB), 0, 3000))
            rec["kind"] = "vet"
        elif p in dS.index:
            dd = dS.loc[p]
            RV = json.load(open(f"{D}/rookie_value.json"))  # college score if NCAA player, else pick+age (young_prior.py, ncaa_model.py)
            SC = pd.read_parquet(f"{D}/ncaa_scores.parquet"); cs = SC[(SC.year == S) & (SC.pick == dd.pick)].cscore
            if len(cs) and pd.notna(cs.iloc[0]):
                val = RV["col_a"] + RV["col_b"] * float(np.clip(cs.iloc[0] - RV["col_mean"], -RV["col_cap"], RV["col_cap"]))
            else:
                val = RV["rk"] + RV["rk_lp"] * (np.log(dd.pick) - 2) + RV["rk_age"] * ((dd.age if pd.notna(dd.age) else 20) - 20)
            rec["rating"] = SLOPE * val
            rec["min_model"] = float(np.clip(brk @ [1, np.log(dd.pick), wp_id.get((N, tid), 0.5), dd.age if pd.notna(dd.age) else 19.5], 50, 2600))
            rec["kind"] = "rookie"
        else:
            rec["rating"] = SLOPE * UNRATED; rec["min_model"] = 400.0; rec["kind"] = "newcomer"
        out.append(rec)
    P = pd.DataFrame(out)
    def allocate(g):
        m = g.min_model.clip(upper=3000).copy(); excess = m.sum() - 19680
        if excess > 0:
            for i in m.sort_values().index:
                cut = min(excess, m[i]); m[i] -= cut; excess -= cut
                if excess <= 0: break
        else:
            for _ in range(10):
                room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
        return m
    P["min_proj"] = P.groupby("team", group_keys=False).apply(allocate)
    T = (P.rating * P.min_proj / 3936).groupby(P.team).sum().rename("net").to_frame()
    T["net"] -= T.net.mean()
    T["margin"] = T.net * pace_of(N) / 100
    T["model_wpct"] = norm.cdf(T.margin / SIG)
    T["season"] = S; T["n_players"] = P.groupby("team").size()
    T["rookie_min_share"] = P[P.kind != "vet"].groupby("team").min_proj.sum() / 19680
    return T.reset_index(), P.assign(season=S)

if __name__ == "__main__":
    seasons = list(map(int, sys.argv[1:])) or list(range(2014, 2026))
    for S in seasons:
        T, P = project_season(S)
        T.to_parquet(f"{D}/{TAG}pre_T_{S}.parquet", index=False); P.to_parquet(f"{D}/{TAG}pre_P_{S}.parquet", index=False)
        print(S, "teams", len(T), "players/team %.1f" % T.n_players.mean(), "sd margin %.2f" % T.margin.std(), flush=True)
