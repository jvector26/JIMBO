import numpy as np, pandas as pd, shift_bt as S
D=S.D; old=pd.read_parquet(f"{D}/shift_bt.parquet"); out=[]
for T in range(2015,2026):
    pg=pd.read_parquet(f"{D}/player_games_{T}.parquet")[["game","player","team","min"]]
    g=old[old.season==T][["game","home_team","away_team"]].copy()
    hk=pd.MultiIndex.from_arrays([g.game,g.home_team]); ak=pd.MultiIndex.from_arrays([g.game,g.away_team])
    R={f"sign_adj_l{l}":S.ridge(T,l,"sign") for l in (3000,10000,30000)}
    R.update({f"pm_adj_l{l}":S.ridge(T,l,"pm") for l in (30000,100000)})
    for c,r in R.items():
        t=(pg.player.map(r).fillna(0)*pg["min"]/48).groupby([pg.game,pg.team]).sum(); g[c]=t.reindex(hk).to_numpy()-t.reindex(ak).to_numpy()
    out.append(g); print(T,flush=True)
old.merge(pd.concat(out).drop(columns=["home_team","away_team"]),on="game").to_parquet(f"{D}/shift_bt.parquet",index=False)
