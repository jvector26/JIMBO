"""Box-part decay grid (residual unaffected: fit_window's prior uses per-season features). Game diffs N+1 of the window
box sum (W-weighted, poss x decay averaged, o_3eff excluded) for many decays -> data/boxdecay_bt.parquet."""
import json, numpy as np, pandas as pd, impact as I
from features import OFF_FEATS, DEF_FEATS
W = json.load(open(f"{I.D}/weights_bt.json")); Wo, Wd = pd.Series(W["Wo"]) * .5, pd.Series(W["Wd"]) * .5
OF = [x for x in OFF_FEATS if x != "o_3eff"]
DEC = {"b00": (0, .2, 1), "b01": (.05, .3, 1), "b1": (.1, .4, 1), "b15": (.15, .45, 1), "b2": (.2, .5, 1), "b3": (.3, .6, 1),
       "b0": (0, 0, 1), "b1b": (.1, .5, 1), "b2b": (.2, .4, 1), "b05": (0, .4, 1)}
f = I.feats(); gm = pd.read_parquet(f"{I.D}/games_all.parquet"); out = []
f["vo"] = f[OF].fillna(0).to_numpy() @ Wo[OF].values; f["vd"] = f[DEF_FEATS].fillna(0).to_numpy() @ Wd[DEF_FEATS].values
for N in range(2014, 2025):
    fw = f[f.season.between(N - 2, N)]
    pace = (I.load_rows([N - 2, N - 1, N]).groupby("game").poss.sum() / 2).mean()
    V = {}
    for tag, dec in DEC.items():
        dm = dict(zip([N - 2, N - 1, N], dec)); d = fw.season.map(dm)
        wo, wd = fw.poss_off * d, fw.poss_def * d
        so = (fw.vo * wo).groupby(fw.player).sum() / wo.groupby(fw.player).sum()
        sd = (fw.vd * wd).groupby(fw.player).sum() / wd.groupby(fw.player).sum()
        V[f"{tag}:box_o"] = so; V[f"{tag}:box_d"] = sd
    V = pd.DataFrame(V).fillna(0)
    pg = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = V.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame((t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100, columns=V.columns); dd["game"] = g.game.values
    out.append(dd)
pd.concat(out).to_parquet(f"{I.D}/boxdecay_bt.parquet", index=False); print("ok")
