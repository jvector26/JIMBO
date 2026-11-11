"""T8 refit: per window N, box game diffs (box decay .1/.4/1, as bc_boxdecay) and refit lineup residuals for
variant 'all' (rebuilt features, = production recipe) and 'ng' (box counts + on-court possessions without garbage stints).
Also saves the 'all' player ratings with components (used by T3). python3 gt_refit.py 2014 2024"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code"); import impact as I
from features import OFF_FEATS, DEF_FEATS
FD = "/home/claude/fund"
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
F0 = pd.read_parquet(f"{I.D}/features.parquet")
def variant(tag):
    x = pd.read_parquet(f"{FD}/features_{tag}.parquet")
    meta = F0[["season", "player", "name", "teams", "age"]]
    return x.merge(meta, on=["season", "player"], how="left")
FV = {"all": variant("all"), "ng": variant("ng")}
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
FC = ["f:" + x for x in OFF_FEATS + DEF_FEATS]
for N in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f"{FD}/gt_refit_{N}.parquet"
    if os.path.exists(fn): continue
    rows = I.load_rows([N-2, N-1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    nxt = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]].copy()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    for tag, F in FV.items():
        I._feats = F
        o = I.fit_window(N, Wo, Wd, lam=2000, rows=rows, box_decay=(.1, .4, 1.)).set_index("player")
        if tag == "all": o.reset_index().to_parquet(f"{FD}/ratings_all_{N}.parquet", index=False)
        for c in ["res_o", "res_d"] + FC:
            x = nxt.player.map(o[c]).fillna(0) * nxt["min"] / 48
            t = x.groupby([nxt.game, nxt.team]).sum()
            g[f"{c}|{tag}"] = (t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100
    g.to_parquet(fn, index=False); print("done", N, flush=True)
