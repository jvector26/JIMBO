"""Robustness: same penalty with PREGAME-KNOWABLE shares (each player's mean minutes over his team's previous 5 games, absent players
included), so tonight's garbage-time minutes cannot drive it. Players with no minutes in the previous 5 team games are invisible here."""
import numpy as np, pandas as pd
D="/home/claude/data"; T=pd.read_parquet("thin_table.parquet"); rows=[]
for S in range(2015,2026):
    g=pd.read_parquet(f"{D}/games_{S}.parquet"); hm=g.set_index("game").home_team; gd=g.set_index("game").date
    pg=pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg=pg[pg["min"]>0]
    for t,dft in pg.groupby("team"):
        gl=pd.Series(dft.game.unique()); gl=gl.iloc[np.argsort(gl.map(gd).values,kind="stable")].to_numpy()
        mm=dft.pivot_table(index="game",columns="player",values="min",aggfunc="sum").reindex(gl).fillna(0)
        tr=mm.rolling(5,min_periods=3).mean().shift(1); cum=mm.cumsum().shift(1).fillna(0); app=(mm>0).cumsum().shift(1).fillna(0)
        w=tr.div(tr.sum(axis=1),axis=0)*5; w=w.stack(); w=w[w>0].rename("tw").reset_index(); w.columns=["game","player","tw"]
        w["team"]=t; w["tsw"]=w.tw*np.where(w.game.map(hm)==t,1.0,-1.0)
        w["cum_t"]=cum.stack().reindex(pd.MultiIndex.from_frame(w[["game","player"]])).to_numpy(); rows.append(w)
    print(S,flush=True)
W=pd.concat(rows,ignore_index=True)
# season-level attributes per player (group needs opening-roster / regular flags; bench vs fringe not used here)
a=T.drop_duplicates(["season","player"]).set_index(["season","player"]); W["season"]=W.game.map(T.drop_duplicates("game").set_index("game").season)
W["out"]=[a.grp.get((s,p))=="out" for s,p in zip(W.season,W.player)]
# season minutes before the game across all teams: take from T where he played tonight, else team-only cumulative
c=T.set_index(["game","player"]).cum; W["cum"]=c.reindex(pd.MultiIndex.from_frame(W[["game","player"]])).to_numpy(); W["cum"]=W.cum.fillna(W.cum_t)
W.to_parquet("thin_pregame.parquet",index=False); print(len(W))
