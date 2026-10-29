"""Monte Carlo season simulation for 2026-27.
Schedule: NBA formula (div 4 games; 6 conf non-div opps x4, 4 x3; other conf x2), 41 home / 41 away for every team.
The 3-game opponents are a random 4-regular graph per conference (re-drawn per batch), oriented with an Euler
circuit so each team hosts exactly 2 of its 4 three-game series (2 home games) -> exactly 41 home games.
Each simulated season draws a true team strength ~ N(projected margin, TAU) (projection error: injuries,
minutes, development, trades), then plays every game with p(home win) = Phi((m_h - m_a + HCA) / SIG).
"""
import numpy as np, pandas as pd
from scipy.stats import norm
D = "data"
DIV = {"Atlantic": "BOS BKN NYK PHI TOR", "Central": "CHI CLE DET IND MIL", "Southeast": "ATL CHA MIA ORL WAS",
       "Northwest": "DEN MIN OKC POR UTA", "Pacific": "GSW LAC LAL PHX SAC", "Southwest": "DAL HOU MEM NOP SAS"}
EAST = {"Atlantic", "Central", "Southeast"}
TEAMS = sorted(t for v in DIV.values() for t in v.split())
IDX = {t: i for i, t in enumerate(TEAMS)}
div_of = {t: d for d, v in DIV.items() for t in v.split()}
conf_of = {t: "E" if div_of[t] in EAST else "W" for t in TEAMS}
SIG, HCA = 12.8, 2.6     # per-game margin sd (as in project.py); home edge in points (~2.7/100 poss * pace)
import json
_B = json.load(open(f"{D}/blend.json"))  # fit in preseason backtest 2014-25 (see HANDOFF)
TAU_MODEL, TAU_BLEND = _B["tau_model"], _B["tau_blend"]  # sd of true margin around projection (pts/game)
TAU = TAU_MODEL
# Playoff strength (session 21, 2026-10-07; RESEARCH_LOG 'PLAYOFF MINUTES TEST'): play-in + series use margin + po_delta; series home court HCA_PO.
_PO = json.load(open(f"{D}/po.json")); PO_META = {k: v for k, v in _PO.items() if k != "note"}  # PO_META is what the page gets (meta.po)

def po_delta(team, rating, minutes, games, G, pace, po=_PO):
    """Playoff strength change per team (points/game): playoff-shaped minutes (stars play more, short rotation) minus the
    regular-season minutes. Same curve as gh/model/jimbo_sim.py po_delta and the page's poDelta()."""
    d = pd.DataFrame({"t": np.asarray(team), "r": np.asarray(rating, float), "m": np.asarray(minutes, float), "g": np.asarray(games, float)})
    d["G"] = d.t.map(G) if hasattr(G, "get") else float(G)
    d = d[d.t.isin(TEAMS) & (d.m > 0) & (d.G > 0)].copy()
    out = pd.Series(0.0, index=TEAMS)
    if not len(d): return out
    gl = d.g.clip(lower=1).clip(upper=d.G); mpg = (d.m / gl).clip(upper=48); av = (gl / d.G).to_numpy()
    c, k = np.array(po["coef"]), po["knots"]
    X = np.column_stack([av, av * mpg] + [av * np.maximum(mpg.to_numpy() - kk, 0) for kk in k])
    d["mB"] = np.clip(X @ c, 0, po["cap"])
    d["mB"] = d.mB * 240 / d.groupby("t").mB.transform("sum").replace(0, np.nan)
    d["mA"] = d.m * 240 / d.groupby("t").m.transform("sum")
    v = (d.r * (d.mB.fillna(d.mA) - d.mA)).groupby(d.t).sum() / 48 * pace / 100
    out[v.index] = v.values
    if "clip" in po: out[out != 0] = out[out != 0].clip(*po["clip"])
    return out * po.get("weight", 1.0)

def three_game_graph(rng, conf):
    """Random 4-regular graph on non-division conference pairs, via randomized greedy + restarts."""
    ts = [t for t in TEAMS if conf_of[t] == conf]
    while True:
        deg = {t: 0 for t in ts}; edges = []
        pairs = [(a, b) for i, a in enumerate(ts) for b in ts[i + 1:] if div_of[a] != div_of[b]]
        rng.shuffle(pairs)
        for a, b in pairs:
            if deg[a] < 4 and deg[b] < 4: edges.append((a, b)); deg[a] += 1; deg[b] += 1
        if all(v == 4 for v in deg.values()): return edges

def euler_orient(edges):
    """Orient edges of an even-degree graph so indegree == outdegree (Hierholzer)."""
    adj = {}
    for k, (a, b) in enumerate(edges): adj.setdefault(a, []).append((b, k)); adj.setdefault(b, []).append((a, k))
    used, out = set(), []
    for start in adj:
        stack, path = [(start, None)], []
        while stack:
            v, e = stack[-1]
            while adj[v] and adj[v][-1][1] in used: adj[v].pop()
            if adj[v]:
                w, k = adj[v].pop(); used.add(k); stack.append((w, (v, w)))
            else:
                stack.pop()
                if e: out.append(e)
    return out  # (u, v): u hosts 2 of the 3 games

def make_schedule(rng):
    g = []  # (home, away)
    for i, a in enumerate(TEAMS):
        for b in TEAMS[i + 1:]:
            if div_of[a] == div_of[b]: g += [(a, b)] * 2 + [(b, a)] * 2
            elif conf_of[a] != conf_of[b]: g += [(a, b), (b, a)]
    for conf in "EW":
        three = set(frozenset(e) for e in three_game_graph(rng, conf))
        ts = [t for t in TEAMS if conf_of[t] == conf]
        for i, a in enumerate(ts):
            for b in ts[i + 1:]:
                if div_of[a] != div_of[b] and frozenset((a, b)) not in three: g += [(a, b)] * 2 + [(b, a)] * 2
        for u, v in euler_orient(list(tuple(e) for e in three)): g += [(u, v)] * 2 + [(v, u)]
    s = np.array([[IDX[h], IDX[a]] for h, a in g])
    assert len(s) == 1230 and (np.bincount(s[:, 0], minlength=30) == 41).all()
    return s

def seed_conf(w, h2h_noise, conf_idx):
    """Rank a conference by wins; ties broken randomly (tiny jitter)."""
    key = w[:, conf_idx] + h2h_noise[:, conf_idx]
    return conf_idx[np.argsort(-key, axis=1)]

def bo7_prob(pa, pb):
    """P(team with home court wins best-of-7, 2-2-1-1-1). pa/pb = its win prob at home/away (arrays)."""
    home = [1, 1, 0, 0, 1, 0, 1]; dp = {(0, 0): np.ones_like(pa)}
    for _ in range(7):
        nd = {}
        for (x, y), pr in dp.items():
            if x == 4 or y == 4: nd[(x, y)] = nd.get((x, y), 0) + pr; continue
            p = pa if home[x + y] else pb
            nd[(x + 1, y)] = nd.get((x + 1, y), 0) + pr * p; nd[(x, y + 1)] = nd.get((x, y + 1), 0) + pr * (1 - p)
        dp = nd
    return sum(pr for (x, y), pr in dp.items() if x == 4)

def simulate(margins, n_sims=20000, n_sched=40, seed=7, tau=None):
    tau = TAU if tau is None else tau
    rng = np.random.default_rng(seed)
    m = np.array([margins[t] for t in TEAMS])
    per = n_sims // n_sched
    W = np.zeros((n_sims, 30), int)
    TRUE = np.zeros((n_sims, 30))
    for b in range(n_sched):
        s = make_schedule(rng)
        true = m + rng.normal(0, tau, (per, 30)); true -= true.mean(1, keepdims=True)
        p = norm.cdf((true[:, s[:, 0]] - true[:, s[:, 1]] + HCA) / SIG)
        hw = rng.random(p.shape) < p
        w = np.zeros((per, 30), int)
        for k in range(30):
            w[:, k] = hw[:, s[:, 0] == k].sum(1) + (~hw[:, s[:, 1] == k]).sum(1)
        W[b * per:(b + 1) * per] = w; TRUE[b * per:(b + 1) * per] = true
    return W, TRUE, rng

def postseason(W, TRUE, rng, pdelta=None, hca_po=None):
    """pdelta: per-team playoff strength change (po_delta) added in play-in and playoff games; hca_po: home court in playoff
    series (play-in games keep the regular-season HCA)."""
    n = len(W)
    if pdelta is not None: TRUE = TRUE + np.asarray(pdelta, float)[None, :]
    HP = HCA if hca_po is None else hca_po
    res = {k: np.zeros(30) for k in ["playoffs", "playin", "top6", "seed1", "r2", "cf", "finals", "title"]}
    jitter = rng.random(W.shape) * 0.5
    pwin = lambda a, b, home: norm.cdf((TRUE[np.arange(n), a] - TRUE[np.arange(n), b] + (HCA if home else -HCA)) / SIG)   # play-in
    pser = lambda a, b, home: norm.cdf((TRUE[np.arange(n), a] - TRUE[np.arange(n), b] + (HP if home else -HP)) / SIG)     # playoff series
    champs_by_conf = {}
    for conf in "EW":
        ci = np.array([IDX[t] for t in TEAMS if conf_of[t] == conf])
        order = seed_conf(W, jitter, ci)  # (n, 15) team indices by seed
        r = np.arange(n)
        for k in range(10): np.add.at(res["playin"], order[:, k], 1 if k >= 6 else 0)
        for k in range(6): np.add.at(res["top6"], order[:, k], 1)
        np.add.at(res["seed1"], order[:, 0], 1)
        s7, s8, s9, s10 = order[:, 6], order[:, 7], order[:, 8], order[:, 9]
        g1 = rng.random(n) < pwin(s7, s8, True); seed7 = np.where(g1, s7, s8); loser78 = np.where(g1, s8, s7)
        g2 = rng.random(n) < pwin(s9, s10, True); w910 = np.where(g2, s9, s10)
        g3 = rng.random(n) < pwin(loser78, w910, True); seed8 = np.where(g3, loser78, w910)
        seeds = np.column_stack([order[:, :6], seed7, seed8])
        for k in range(8): np.add.at(res["playoffs"], seeds[:, k], 1)
        def series(a, b):  # a has home court
            P = bo7_prob(pser(a, b, True), pser(a, b, False))
            aw = rng.random(n) < P
            return np.where(aw, a, b)
        def better(a, b):  # home court to better record (ties: lower seed index passed as a)
            wa, wb = W[r, a], W[r, b]
            swap = wb > wa
            return np.where(swap, b, a), np.where(swap, a, b)
        r1 = [series(*better(seeds[:, i], seeds[:, 7 - i])) for i in range(4)]  # 1v8, 2v7, 3v6, 4v5
        for t in r1: np.add.at(res["r2"], t, 1)
        r2 = [series(*better(r1[0], r1[3])), series(*better(r1[1], r1[2]))]
        for t in r2: np.add.at(res["cf"], t, 1)
        cc = series(*better(r2[0], r2[1])); np.add.at(res["finals"], cc, 1)
        champs_by_conf[conf] = cc
    a, b = champs_by_conf["E"], champs_by_conf["W"]
    wa, wb = W[np.arange(n), a], W[np.arange(n), b]
    hi, lo = np.where(wb > wa, b, a), np.where(wb > wa, a, b)
    P = bo7_prob(pser(hi, lo, True), pser(hi, lo, False))
    ch = np.where(rng.random(n) < P, hi, lo); np.add.at(res["title"], ch, 1)
    return {k: v / n for k, v in res.items()}

if __name__ == "__main__":
    T = pd.read_csv(f"{D}/teams_proj_2026.csv", index_col=0)
    L = pd.read_csv(f"{D}/win_totals_2026.csv").set_index("team")
    imp = lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100)
    po = L.over.map(imp) / (L.over.map(imp) + L.under.map(imp))
    L["mkt_mean"] = L.line + 7 * norm.ppf(po)
    outs = {}
    for mode in ("model", "blend"):
        if mode == "model":
            mg, tau = T.margin.to_dict(), TAU_MODEL
        else:
            # blend in win space, convert back to a per-game margin (inverse of 82*Phi(m/SIG)); pre-sim win formula
            mw = 82 * norm.cdf(T.margin / SIG)
            P = pd.read_csv(f"{D}/players_proj_2026.csv")
            age = (P.age.fillna(21) * P.min_proj).groupby(P.team).sum() / P.groupby("team").min_proj.sum()
            age_c = (age - age.mean()).reindex(T.index)
            bw = 41 + _B["w_model"] * (mw - 41) + _B["w_vegas"] * (L.mkt_mean.reindex(T.index) - 41) + _B["w_age"] * age_c
            T["roster_age_c"] = age_c
            bw = bw - (bw.mean() - 41)
            mg, tau = pd.Series(SIG * norm.ppf(bw / 82), index=T.index).to_dict(), TAU_BLEND
        W, TRUE, rng = simulate(mg, tau=tau)
        # py_sim baseline for test_dashboard.js: series home court only, NO po_delta (the test calls the page sim without opt.pd).
        # The page itself (and the nightly) add po_delta on top; sim_2026.csv titles are therefore the "regular-minutes" version.
        post = postseason(W, TRUE, rng, hca_po=_PO["hca_po"])
        o = pd.DataFrame({"team": TEAMS, "mean_w": W.mean(0), "p10": np.percentile(W, 10, 0), "p90": np.percentile(W, 90, 0),
                          **post, "p_over": [(W[:, IDX[t]] > L.line[t]).mean() for t in TEAMS]}).set_index("team")
        outs[mode] = o
        np.save(f"{D}/sim_wins_2026_{mode}.npy", W.astype(np.int8))
    M, B = outs["model"], outs["blend"]
    out = M.add_prefix("").join(B.add_prefix("blend_"))
    out.insert(0, "conf", [conf_of[t] for t in out.index])
    out = out.join(L[["line", "mkt_mean"]]).join(T[["roster_age_c"]])
    out["edge"] = out.mean_w - out.mkt_mean
    out = out.sort_values("blend_mean_w", ascending=False).reset_index()
    out.to_csv(f"{D}/sim_2026.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["team", "line", "mkt_mean", "mean_w", "edge", "roster_age_c", "blend_mean_w", "blend_p10", "blend_p90", "blend_playoffs", "blend_title", "blend_p_over", "p_over"]
    print(out[cols].round(3).to_string(index=False))
