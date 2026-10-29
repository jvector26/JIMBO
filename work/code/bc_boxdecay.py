"""Rewrite backtest_components f:<feature> game diffs (except f:o_3eff = shooting prior) for a box decay.
python3 code/bc_boxdecay.py a,b,c [--check]. Backup: data/pre_bd/backtest_components.parquet."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "code"); import impact as I
from features import OFF_FEATS, DEF_FEATS
dec = tuple(map(float, sys.argv[1].split(","))); check = "--check" in sys.argv
W = json.load(open(f"{I.D}/weights_bt.json")); Wa = {**{k: v * .5 for k, v in W["Wo"].items()}, **{k: v * .5 for k, v in W["Wd"].items()}}
FE = [x for x in OFF_FEATS + DEF_FEATS if x != "o_3eff"]
f = I.feats(); gm = pd.read_parquet(f"{I.D}/games_all.parquet"); out = []
for N in range(2014, 2025):
    fw = f[f.season.between(N - 2, N)].copy(); d = fw.season.map(dict(zip([N - 2, N - 1, N], dec)))
    V = {}
    for x in FE:
        wt = (fw.poss_off if x.startswith("o_") else fw.poss_def) * d
        V["f:" + x] = (fw[x].fillna(0) * Wa[x] * wt).groupby(fw.player).sum() / wt.groupby(fw.player).sum()
    V = pd.DataFrame(V).fillna(0)
    pace = (I.load_rows([N - 2, N - 1, N]).groupby("game").poss.sum() / 2).mean()
    pg = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = V.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame((t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100, columns=V.columns); dd["game"] = g.game.values
    out.append(dd)
New = pd.concat(out).set_index("game")
B = pd.read_parquet(f"{I.D}/backtest_components.parquet")
if check:
    print("max abs diff vs current:", (B.set_index("game")[New.columns] - New.reindex(B.game).values).abs().max().max()); sys.exit()
for c in New.columns: B[c] = B.game.map(New[c]).values
B.to_parquet(f"{I.D}/backtest_components.parquet", index=False); print("rewritten", len(B))
