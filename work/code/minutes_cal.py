"""Minutes calibration (session 13): replay the CURRENT preseason minutes model for each October 2014-2025 and keep its
pieces (mpg when playing, expected games, age mult, allocation). Adds full-season absentees (played N and later, 0 games in S)
that pre_P lacks. -> data/minutes_cal.parquet (one row per projected player-season)."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
import preseason_backtest as PB
D = PB.D; ps, ng, dm, Xm, Xa = PB.ps, PB.ng, PB.dm, PB.Xm, PB.Xa
MIN_AGE = json.load(open(f"{D}/blend.json"))["min_age_mult"]
bin5 = lambda a: "<=22" if a <= 22 else "23-25" if a <= 25 else "26-28" if a <= 28 else "29-31" if a <= 31 else "32-34" if a <= 34 else "35+"
R26 = set(pd.read_csv(f"{D}/rosters_2026.csv").player.dropna().astype(int))
id2abbr = PB.id2abbr
out = []
for S in range(2014, 2026):
    N = S - 1
    P = pd.read_parquet(f"{D}/pre_P_{S}.parquet")[["player", "team", "kind", "min_model", "rating"]]
    hist = ps[ps.season <= N]
    gp_by = hist.pivot_table(index="player", columns="season", values="gp", aggfunc="sum").fillna(0)
    m_by = hist.pivot_table(index="player", columns="season", values="min", aggfunc="sum").fillna(0)
    gs_by = hist.pivot_table(index="player", columns="season", values="gs", aggfunc="sum").fillna(0)
    last = hist.sort_values("season").groupby("player").last()
    # full-season absentees: played in N, 0 games in S, played again later (or on a 2026-27 roster)
    later = set(ps[ps.season > S].player) | (R26 if S == 2025 else set())
    inS = set(ps[ps.season == S].player)
    ab = [p for p in gp_by.index[gp_by.get(N, pd.Series(0, index=gp_by.index)) > 0] if p not in inS and p in later and p not in set(P.player)]
    lt = pd.read_parquet(f"{D}/player_games_{N}.parquet").sort_values(["date", "game"]).drop_duplicates("player", keep="last").set_index("player").team
    P = pd.concat([P, pd.DataFrame({"player": ab, "team": [id2abbr.get(lt.get(p)) for p in ab], "kind": "vet", "min_model": np.nan, "rating": np.nan})], ignore_index=True)
    P["absentee"] = P.player.isin(ab)
    tr = dm[dm.season.between(2000, N)]
    bm = np.linalg.lstsq(Xm(tr) * np.sqrt(tr.gp.values)[:, None], tr.mpg * np.sqrt(tr.gp.values), rcond=None)[0]
    ba = np.linalg.lstsq(Xa(tr), tr.avail, rcond=None)[0]
    recs = []
    for p in P.player:
        rec = {"player": p}
        if p in gp_by.index:
            age = last.loc[p, "age"] + (S - last.loc[p, "season"]) if pd.notna(last.loc[p, "age"]) else np.nan
            yrs = [y for y in (N, N - 1, N - 2, N - 3) if gp_by.loc[p].get(y, 0) > 0]
            injury_year = gp_by.loc[p].get(N, 0) == 0 and len(yrs) > 0
            use = (yrs + [None] * 3)[:3] if yrs else [None] * 3
            St = lambda y, w: {"mpg": m_by.loc[p].get(y, 0) / gp_by.loc[p].get(y, 0), "avail": min(gp_by.loc[p].get(y, 0) / ng[y], 1), "gsr": gs_by.loc[p].get(y, 0) / gp_by.loc[p].get(y, 0)}[w]
            mpg1 = St(use[0], "mpg") if use[0] else 12.0; mpg2 = St(use[1], "mpg") if use[1] else mpg1; mpg3 = St(use[2], "mpg") if use[2] else mpg2
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
            # durability history: seasons (of last 3 league seasons N..N-2) as a 20+ mpg regular and availability in each
            hv = []
            for y in (N, N - 1, N - 2):
                g = gp_by.loc[p].get(y, 0)
                hv.append(min(g / ng[y], 1) if g > 0 and m_by.loc[p].get(y, 0) / g >= 20 else np.nan)
            rec.update(age=age, mpg1=mpg1, av1=av1, gsr1=gsr1, mA=mA, mB=mB, aA=aA, aB=aB, injury_year=injury_year,
                       mpg_wp=0.5 * mA + 0.5 * mB, avail_p=0.5 * aA + 0.5 * aB, h1=hv[0], h2=hv[1], h3=hv[2])
        recs.append(rec)
    P = P.merge(pd.DataFrame(recs), on="player", how="left")
    P["age"] = P.age.fillna(21)
    P["amult"] = P.age.map(bin5).map(MIN_AGE)
    vet = P.mpg_wp.notna()
    P.loc[vet, "min_model"] = (P.mpg_wp * 82 * P.avail_p).clip(0, 3000)[vet]
    P["min_model_age"] = (P.min_model * P.amult).clip(upper=3000)
    def allocate(g):
        m = g.min_model_age.clip(upper=3000).copy(); excess = m.sum() - 19680
        if excess > 0:
            for i in m.sort_values().index:
                cut = min(excess, m[i]); m[i] -= cut; excess -= cut
                if excess <= 0: break
        else:
            for _ in range(10):
                room = m < 3000; m[room] *= (19680 - m[~room].sum()) / m[room].sum(); m = m.clip(upper=3000)
        return m
    P["min_proj"] = P.groupby("team", group_keys=False).apply(allocate)
    a = ps[ps.season == S].groupby("player").agg(gp=("gp", "sum"), mins=("min", "sum"), gs=("gs", "sum"))
    P = P.join(a, on="player"); P[["gp", "mins", "gs"]] = P[["gp", "mins", "gs"]].fillna(0)
    P["ng"] = ng[S]; P["season"] = S
    out.append(P); print(S, len(P), "absentees", len(ab), flush=True)
pd.concat(out, ignore_index=True).to_parquet(f"{D}/minutes_cal.parquet", index=False)
