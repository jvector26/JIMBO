"""In-season MINUTES backtest (daily-pipeline design, companion to inseason_bt.py).

At each checkpoint (league-median games 5..70) of seasons 2014-15..2025-26, ratings are updated with the chosen M4
method (box update dS=1, RAPM-toward-prior lam 6000, team offsets lam_t 80). Then rest-of-season player minutes are
PROJECTED (not actual) from information before the checkpoint, and every later game is predicted. Scored on game
margin RMSE and team rest-of-season average margin RMSE, against the oracle that uses each game's actual minutes.

Minutes candidates for each player on a team at the checkpoint (current team = team of his latest game; preseason
roster players who have not appeared stay with their preseason team):
  pre   preseason projected mpg per team game (pre_P min_proj / 82)
  std   season-to-date minutes per team game since joining the current team (DNPs count as 0)
  l10   minutes over the team's last 10 games (or fewer since joining)
  est   = blend: w*(r*l10 + (1-r)*std) + (1-w)*pre, w = n/(n+k), n = team games since joining
  out   if absent from the last j team games -> multiply by f (proxy for injury; the pipeline has real injury reports)
Team totals are rescaled to 240 minutes per game (5 on-court shares).
Stage 1 (ratings) is cached per season in data/ism_r_{S}.pkl.
"""
import os, sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
import inseason_bt as B
D = "/home/claude/data"
CHECK = [5, 10, 20, 30, 41, 55, 70]
LAM, DS, H0, LAM_T = 6000, 1, 2000, 80


def stage1(S):
    """M4 ratings + team offsets per checkpoint (mirrors inseason_bt.run_season for the chosen key)."""
    fn = f"{D}/ism_r_{S}.pkl"
    if os.path.exists(fn): return pickle.load(open(fn, "rb"))
    N = S - 1
    pre = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    g = pd.read_parquet(f"{D}/games_{S}.parquet")[["game", "date", "home_team", "away_team", "home_pts", "away_pts"]]
    g["margin"] = g.home_pts - g.away_pts
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg = pg[pg["min"] > 0]
    st = pd.read_parquet(f"{D}/stints_{S}.parquet")
    st_date = st.game.map(g.set_index("game").date); home_of = g.set_index("game").home_team
    O_h, D_h, wo_h, wd_h = B.history_box(N)
    allp = pd.Index(pg.player.unique()).union(pre.index)
    hb = pd.Series(B.box_of(pd.concat([O_h.div(wo_h, axis=0), D_h.div(wd_h, axis=0)], axis=1).fillna(0)), index=O_h.index)
    pre_r = pre.rating.reindex(allp).fillna(B.SLOPE * hb.reindex(allp)).fillna(B.SLOPE * B.UNRATED)
    tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "t"}),
                    g[["game", "date", "away_team"]].rename(columns={"away_team": "t"})]).sort_values(["date", "game"])
    dates = np.sort(g.date.unique())
    med = pd.Series({d: tg[tg.date < d].groupby("t").size().reindex(tg.t.unique()).fillna(0).median() for d in dates})
    pg2 = pg.merge(g[["game", "home_team"]], on="game"); pg2["side"] = np.where(pg2.team == pg2.home_team, 1.0, -1.0)
    out = {}
    for cp in CHECK:
        if not (med >= cp).any(): continue
        d = med.index[med >= cp][0]
        past_g = g[g.date < d]; fut = g[g.date >= d]
        if len(fut) < 30: continue
        pst = st[st_date < d]
        pa = sum(pst[f"fg2a{z}"] + pst[f"fg3a{z}"] - pst[f"oreb{z}"] + pst[f"tov{z}"] + 0.44 * pst[f"fta{z}"] for z in "AB")
        pace = pa.sum() / 2 / pst.game.nunique()
        ppg = pg[pg.game.isin(past_g.game)]
        fnow = B.partial_features(ppg, pst, None)
        A, Bm, y, w, hA, _ = B.stint_rows(pst, home_of, g.set_index("game").date)
        r = pre_r.copy(); ids = fnow.index
        Oh = O_h.reindex(ids).fillna(0); Dh = D_h.reindex(ids).fillna(0)
        woh = wo_h.reindex(ids).fillna(0); wdh = wd_h.reindex(ids).fillna(0)
        hist_box = pd.Series(B.box_of(pd.concat([Oh.div(woh.replace(0, np.nan), axis=0), Dh.div(wdh.replace(0, np.nan), axis=0)], axis=1).fillna(0)), index=ids)
        nohist = woh < 1
        hist_box[nohist] = (pre_r.reindex(ids) / B.SLOPE)[nohist]
        woh = woh.where(~nohist, H0); wdh = wdh.where(~nohist, H0)
        cur_box = pd.Series(B.box_of(fnow), index=ids)
        wc = DS * (fnow.poss_off + fnow.poss_def) / 2; wh = (woh + wdh) / 2
        r = r.add(B.SLOPE * ((hist_box * wh + cur_box * wc) / (wh + wc) - hist_box), fill_value=0)
        prior = r.reindex(np.unique(np.r_[A.ravel(), Bm.ravel()])).fillna(B.SLOPE * B.UNRATED)
        dlt, _ = B.rapm_delta(A, Bm, y, w, hA, prior, LAM)
        r = r.add(dlt, fill_value=0)
        x = pg2[pg2.game.isin(past_g.game)]
        tm = x.groupby(["game", "team"])["min"].transform("sum")
        s = (5 * x["min"] / tm * x.side * x.player.map(r).fillna(B.SLOPE * B.UNRATED)).groupby(x.game).sum()
        pp = past_g.assign(pred=past_g.game.map(s).fillna(0).to_numpy() * pace / 100 + B.HCA)
        off = B.team_offsets(pp, LAM_T)
        out[cp] = dict(d=d, r=r, off=off, pace=pace)
    pickle.dump(out, open(fn, "wb"))
    return out


def minutes_inputs(S, d, g, pg, pre, id2ab):
    """Per (team, player) at checkpoint date d: pre mpg, std mpg, l10 mpg, n games since joining, absent streak."""
    ab2id = {v: k for k, v in id2ab.items()}
    past = pg[pg.date < d]
    tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "team"}),
                    g[["game", "date", "away_team"]].rename(columns={"away_team": "team"})])
    tgp = tg[tg.date < d].sort_values(["date", "game"])
    tgp["k"] = tgp.groupby("team").cumcount(ascending=False)     # 0 = team's most recent game
    ngames = tgp.groupby("team").size()
    last = past.sort_values(["date", "game"]).drop_duplicates("player", keep="last").set_index("player").team
    rows = []
    # players who have appeared: current team = team of latest game
    x = past.merge(tgp[["game", "team", "k"]], on=["game", "team"])
    x = x[x.team == x.player.map(last)]
    first_k = x.groupby("player").k.max()                       # team-game index of first appearance for current team
    for p, t in last.items():
        kmax = int(first_k[p]); n = kmax + 1
        xp = x[x.player == p]
        m_by_k = xp.groupby("k")["min"].sum()
        std = m_by_k.sum() / n
        n10 = min(10, n); l10 = m_by_k[m_by_k.index < n10].sum() / n10
        played = sorted(m_by_k.index[m_by_k > 0]); streak = played[0] if played else n
        rows.append((t, p, std, l10, n, streak))
    df = pd.DataFrame(rows, columns=["team", "player", "std", "l10", "n", "streak"])
    # preseason-roster players who have not appeared yet
    pr = pre[~pre.index.isin(last.index)]
    extra = pd.DataFrame({"team": pr.team.map(ab2id).values, "player": pr.index.values, "std": 0.0, "l10": 0.0, "n": 0,
                          "streak": pr.team.map(ab2id).map(ngames).fillna(0).values})
    df = pd.concat([df, extra], ignore_index=True)
    # preseason mpg: own projection if on the preseason roster (any team), else missing
    df["pre"] = df.player.map(pre.min_proj / 82)
    df["on_pre"] = df.pre.notna()
    return df, ngames


def est_minutes(df, k, rw, j, f, pre_new=None):
    """Expected rest-of-season minutes per team game, rescaled to 240 per team."""
    n = df.n.to_numpy(float)
    cur = rw * df.l10 + (1 - rw) * df["std"]
    prior = df.pre.fillna(df["std"] if pre_new is None else pre_new).to_numpy(float)
    w = n / (n + k) if k > 0 else np.where(n > 0, 1.0, 0.0)
    m = w * cur + (1 - w) * prior
    if j is not None:
        m = np.where(df.streak >= j, m * f, m)
    m = pd.Series(np.asarray(m, float), index=df.index)
    tot = m.groupby(df.team).transform("sum")
    return 240 * m / tot.replace(0, np.nan)


def run(S, grid):
    R = stage1(S)
    g = pd.read_parquet(f"{D}/games_{S}.parquet")
    g["margin"] = g.home_pts - g.away_pts
    id2ab = dict(zip(g.home_team, g.htm))
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet")[["game", "player", "team", "min", "date"]]
    pg = pg[pg["min"] > 0]
    pre = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    pgh = pg.merge(g[["game", "home_team"]], on="game"); pgh["side"] = np.where(pgh.team == pgh.home_team, 1.0, -1.0)
    res = []
    for cp, Z in R.items():
        d, r, off, pace = Z["d"], Z["r"], Z["off"], Z["pace"]
        fut = g[g.date >= d]
        rv = lambda s: s.map(r).fillna(B.SLOPE * B.UNRATED)
        adj = fut.home_team.map(off).fillna(0).to_numpy() - fut.away_team.map(off).fillna(0).to_numpy()
        # oracle: actual minutes of each future game
        x = pgh[pgh.game.isin(fut.game)]
        tm = x.groupby(["game", "team"])["min"].transform("sum")
        s = (5 * x["min"] / tm * x.side * rv(x.player)).groupby(x.game).sum()
        pred_or = fut.game.map(s).fillna(0).to_numpy() * pace / 100 + B.HCA + adj
        # actual rest-of-season minutes per player-team (for player-level scoring)
        df, ngames = minutes_inputs(S, d, g, pg, pre, id2ab)
        tfut = pd.concat([fut.home_team, fut.away_team]).value_counts()
        act = x.groupby(["team", "player"])["min"].sum()
        df["act"] = [act.get((t, p), 0.0) / tfut.get(t, 1) for t, p in zip(df.team, df.player)]
        df["r"] = rv(df.player).to_numpy()

        def score(pred, tag, extra=None):
            e = fut.margin.to_numpy() - pred
            tt = pd.concat([pd.DataFrame({"t": fut.home_team, "a": fut.margin, "p": pred}),
                            pd.DataFrame({"t": fut.away_team, "a": -fut.margin, "p": -pred})]).groupby("t").agg(a=("a", "mean"), p=("p", "mean"), n=("a", "size"))
            row = dict(season=S, cp=cp, key=tag, n=len(fut), sse=float((e ** 2).sum()),
                       team_sse=float(((tt.a - tt.p) ** 2 * tt.n).sum()), team_n=int(tt.n.sum()))
            if extra is not None: row.update(extra)
            res.append(row)
        score(pred_or, "oracle")
        for key in grid:
            m = est_minutes(df, *key)
            share = m / 48                                             # on-court share (sums to 5 per team)
            tr = (share * df.r).groupby(df.team).sum()                 # team rating per 100 poss (5-man sum)
            pred = (fut.home_team.map(tr).fillna(0) - fut.away_team.map(tr).fillna(0)).to_numpy() * pace / 100 + B.HCA + adj
            me = (m - df.act)
            score(pred, key, dict(p_sse=float((me ** 2).sum()), p_n=len(df), p_wsse=float((me ** 2 * df.act).sum()), p_w=float(df.act.sum())))
    return res


def grid():
    if os.environ.get("FINE"):
        G = [(0, 0.0, None, 1.0), (1e9, 0.0, None, 1.0)]
        G += [(k, rw, j, f) for k in (3, 5, 7, 10, 15) for rw in (0.3, 0.5, 0.7, 1.0) for j, f in ((None, 1.0), (5, 0.5), (5, 0.25))]
        return G
    G = [(0, 0.0, None, 1.0)]                                          # k=0: season-to-date only
    G += [(k, rw, None, 1.0) for k in (1e9, 5, 10, 20, 40) for rw in (0.0, 0.5)]   # 1e9 = preseason only
    G += [(k, rw, j, f) for k in (10, 20) for rw in (0.0, 0.5) for j in (3, 5, 10) for f in (0.0, 0.25, 0.5)]
    return G


if __name__ == "__main__":
    seasons = list(map(int, sys.argv[1:])) or list(range(2014, 2026))
    for S in seasons:
        res = run(S, grid())
        pickle.dump(res, open(f"{D}/ism_{S}.pkl", "wb"))
        o = {x["cp"]: np.sqrt(x["team_sse"] / x["team_n"]) for x in res if x["key"] == "oracle"}
        print(S, "oracle team rmse", {k: round(v, 2) for k, v in o.items()}, flush=True)
