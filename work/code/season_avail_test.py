"""Season-minutes rule: split availability x minutes-when-playing vs the current per-team-game blend (2026-10-29, session 38).

User question (Giannis 37.5 min when playing in the app): the nightly blends minutes per TEAM game toward this season at
w = n/(n+5), but availability toward this season at n/(n+20). A star with low preseason availability who has played every
game gets his per-team-game minutes pushed up (because he was available) while availability barely moves, so
min_rem / (games x avail) explodes. Team strength uses the per-team-game number, so the test is which per-team-game
projection predicts the rest of the season better.

Same replay as code/inseason_minutes.py (checkpoints at league-median games 5..70, 2014-15..2025-26, M4 ratings cached in
data/ism_r_{S}.pkl, every later game predicted with projected minutes), no injury reports in either rule.
Candidates (per team game, then team rescaled to 240):
  cur              w (0.5 l10 + 0.5 std) + (1-w) pre, w = n/(n+5)              [production nightly]
  split(Ka,k2,rw)  avail = (Ka avp + gp) / (Ka + n)                              [production av, Ka = 20]
                   wp    = w2 obs_wp + (1-w2) pre/avp, w2 = gp/(gp+k2), obs_wp = rw l10_wp + (1-rw) std_wp
                   m     = avail x wp
avp = preseason availability proxy: 0.4 a1 + 0.25 a2 + 0.15 a3 + 0.2 x 0.62 from the prior three seasons' games played /
team games (project.py's recency rule B; corr .87 with production avail_pre for 2026, same mean). Players without NBA history: 0.62.
n = team games since joining the current team, gp = games he played in them (minutes > 0).
Scores: team rest-of-season margin RMSE, game RMSE, player minutes-per-team-game error (minutes-weighted and plain), and the
'fragile and healthy so far' subset (avp < 0.65, played every team game, n >= 4).
"""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
import inseason_minutes as IM, inseason_bt as B
D = "/home/claude/data"


def avail_proxy(S):
    av = {}
    for k, y in enumerate([S - 1, S - 2, S - 3]):
        fn = f"{D}/player_games_{y}.parquet"
        if not os.path.exists(fn): continue
        pg = pd.read_parquet(fn, columns=["player", "game", "min"]); pg = pg[pg["min"] > 0]
        ng = len(pd.read_parquet(f"{D}/games_{y}.parquet")) * 2 / 30
        av[k] = (pg.groupby("player").game.nunique() / ng).clip(upper=1)
    A = pd.DataFrame(av).reindex(columns=[0, 1, 2])
    a1 = A[0].fillna(A[1]).fillna(A[2]); a2 = A[1].fillna(a1); a3 = A[2].fillna(a2)
    return 0.4 * a1 + 0.25 * a2 + 0.15 * a3 + 0.2 * 0.62


def inputs(S, d, g, pg, pre, id2ab, avp):
    """IM.minutes_inputs plus games played since joining (gp) and last-10 games played (gp10)."""
    df, ngames = IM.minutes_inputs(S, d, g, pg, pre, id2ab)
    past = pg[pg.date < d]
    tg = pd.concat([g[["game", "date", "home_team"]].rename(columns={"home_team": "team"}),
                    g[["game", "date", "away_team"]].rename(columns={"away_team": "team"})])
    tgp = tg[tg.date < d].sort_values(["date", "game"])
    tgp["k"] = tgp.groupby("team").cumcount(ascending=False)
    last = past.sort_values(["date", "game"]).drop_duplicates("player", keep="last").set_index("player").team
    x = past.merge(tgp[["game", "team", "k"]], on=["game", "team"]); x = x[x.team == x.player.map(last)]
    gp = x.groupby("player").game.nunique()
    n_of = df.set_index("player").n
    x["n10"] = x.player.map(n_of).clip(upper=10)
    gp10 = x[x.k < x.n10].groupby("player").game.nunique()
    df["gp"] = df.player.map(gp).fillna(0).to_numpy()
    df["gp10"] = df.player.map(gp10).fillna(0).to_numpy()
    df["avp"] = df.player.map(avp).fillna(0.62).clip(0.3, 1).to_numpy()   # production clips avail at 0.3 for the prior cap
    return df


def est_split(df, Ka, k2, rw, asym=False, j=None, fo=1.0):
    n, gp = df.n.to_numpy(float), df.gp.to_numpy(float)
    avail = ((Ka * df.avp + gp) / (Ka + n)).to_numpy(float)
    if asym:   # missed games count fast (as the current per-team-game blend, k2), healthy games above the prior slowly (Ka)
        fast = ((k2 * df.avp + gp) / (k2 + n)).to_numpy(float)
        avail = np.where(fast < df.avp, fast, avail)
    std_wp = np.where(gp > 0, df["std"] * n / np.maximum(gp, 1), 0)
    n10 = np.minimum(n, 10)
    l10_wp = np.where(df.gp10 > 0, df.l10 * n10 / np.maximum(df.gp10, 1), std_wp)
    obs = rw * l10_wp + (1 - rw) * std_wp
    prior_wp = (df.pre / df.avp).to_numpy(float)
    prior_wp = np.where(np.isnan(prior_wp), obs, prior_wp)
    w2 = gp / (gp + k2)
    m = avail * (w2 * obs + (1 - w2) * prior_wp)
    if j is not None: m = np.where(df.streak >= j, m * fo, m)
    m = pd.Series(m, index=df.index)
    return 240 * m / m.groupby(df.team).transform("sum").replace(0, np.nan), avail


GRID = [("cur", None), ("cur_j3", "j3"), ("cur_j5h", "j5h")] + [(f"split_Ka{Ka}_k{k2}_rw{rw}", (Ka, k2, rw)) for Ka in (10, 20, 40) for k2 in (3, 5, 10) for rw in (0.0, 0.5)]
GRID += [(f"asym_Ka{Ka}_k{k2}_rw{rw}", (Ka, k2, rw, True)) for Ka in (10, 20, 40) for k2 in (3, 5) for rw in (0.5,)]
GRID += [("split_Ka20_k5_rw0.5_j3", (20, 5, 0.5, False, 3, 0.0)), ("split_Ka20_k5_rw0.5_j5h", (20, 5, 0.5, False, 5, 0.5)),
         ("asym_Ka20_k5_rw0.5_j3", (20, 5, 0.5, True, 3, 0.0)), ("asym_Ka20_k5_rw0.5_j5h", (20, 5, 0.5, True, 5, 0.5))]
CUR = {None: (None, 1.0), "j3": (3, 0.0), "j5h": (5, 0.5)}


def run(S):
    R = IM.stage1(S)
    g = pd.read_parquet(f"{D}/games_{S}.parquet"); g["margin"] = g.home_pts - g.away_pts
    id2ab = dict(zip(g.home_team, g.htm))
    pg = pd.read_parquet(f"{D}/player_games_{S}.parquet")[["game", "player", "team", "min", "date"]]; pg = pg[pg["min"] > 0]
    pre = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    avp = avail_proxy(S)
    pgh = pg.merge(g[["game", "home_team"]], on="game")
    res, sub = [], []
    for cp, Z in R.items():
        d, r, off, pace = Z["d"], Z["r"], Z["off"], Z["pace"]
        fut = g[g.date >= d]
        rv = lambda s: s.map(r).fillna(B.SLOPE * B.UNRATED)
        adj = fut.home_team.map(off).fillna(0).to_numpy() - fut.away_team.map(off).fillna(0).to_numpy()
        df = inputs(S, d, g, pg, pre, id2ab, avp)
        x = pgh[pgh.game.isin(fut.game)]
        tfut = pd.concat([fut.home_team, fut.away_team]).value_counts()
        act = x.groupby(["team", "player"])["min"].sum(); actg = x.groupby(["team", "player"]).game.nunique()
        df["act"] = [act.get((t, p), 0.0) / tfut.get(t, 1) for t, p in zip(df.team, df.player)]
        df["act_g"] = [actg.get((t, p), 0) for t, p in zip(df.team, df.player)]
        df["act_wp"] = [act.get((t, p), 0.0) / max(actg.get((t, p), 0), 1) for t, p in zip(df.team, df.player)]
        df["r"] = rv(df.player).to_numpy()
        frag = (df.avp < 0.65) & (df.gp == df.n) & (df.n >= 4) & (df.pre * 82 > 1000)
        for tag, par in GRID:
            if par is None or isinstance(par, str):
                m = IM.est_minutes(df, 5, 0.5, *CUR[par])
                av_disp = ((20 * df.avp + df.gp) / (20 + df.n)).to_numpy()   # production av (k 20) for the display check
            else:
                m, av_disp = est_split(df, *par)
            tr = (m / 48 * df.r).groupby(df.team).sum()
            pred = (fut.home_team.map(tr).fillna(0) - fut.away_team.map(tr).fillna(0)).to_numpy() * pace / 100 + B.HCA + adj
            e = fut.margin.to_numpy() - pred
            tt = pd.concat([pd.DataFrame({"t": fut.home_team, "a": fut.margin, "p": pred}),
                            pd.DataFrame({"t": fut.away_team, "a": -fut.margin, "p": -pred})]).groupby("t").agg(a=("a", "mean"), p=("p", "mean"), n=("a", "size"))
            me = m - df.act
            res.append(dict(season=S, cp=cp, key=tag, n=len(fut), sse=float((e ** 2).sum()),
                            team_sse=float(((tt.a - tt.p) ** 2 * tt.n).sum()), team_n=int(tt.n.sum()),
                            p_sse=float((me ** 2).sum()), p_n=len(df), p_wsse=float((me ** 2 * df.act).sum()), p_w=float(df.act.sum()),
                            f_sse=float((me[frag] ** 2).sum()), f_n=int(frag.sum()), f_bias=float(me[frag].sum())))
            if frag.any():
                f = df[frag]
                sub.append(pd.DataFrame(dict(season=S, cp=cp, key=tag, player=f.player, m=m[frag], act=f.act,
                                             disp_wp=(m[frag] / av_disp[frag.to_numpy()]), act_wp=f.act_wp, act_g=f.act_g)))
    return res, (pd.concat(sub) if sub else pd.DataFrame())


if __name__ == "__main__":
    seasons = list(map(int, sys.argv[1:])) or list(range(2014, 2026))
    for S in seasons:
        res, sub = run(S)
        pickle.dump((res, sub), open(f"{D}/savail_{S}.pkl", "wb"))
        t = {x["key"]: np.sqrt(x["team_sse"] / x["team_n"]) for x in res if x["cp"] == 20}
        print(S, "cp20 team rmse cur %.3f split20/5/.5 %.3f" % (t.get("cur", np.nan), t.get("split_Ka20_k5_rw0.5", np.nan)), flush=True)
