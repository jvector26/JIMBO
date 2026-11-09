"""Player-game table for the low-season-minutes penalty (session 47). One row per player-game 2015-25:
sw = signed minute share (5*min/team min, + home / - away), cum = player's season minutes before the game, tgp = team games before,
grp: out = not on an opening roster | reg = last season 20+ mpg and 30+ games (an absence is almost surely injury) |
     bench = other, has appeared in half or more of the team's games so far | fringe = other, has appeared in fewer than half
missed = consecutive team games missed right before this one (0 if he played the previous game or just joined)."""
import numpy as np, pandas as pd
D="/home/claude/data"; ps=pd.read_parquet(f"{D}/player_seasons.parquet"); out=[]
for S in range(2015,2026):
    g=pd.read_parquet(f"{D}/games_{S}.parquet"); hm=g.set_index("game").home_team; gd=g.set_index("game").date
    pg=pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg=pg[pg["min"]>0][["game","player","team","min"]].copy()
    pg["date"]=pg.game.map(gd); pg=pg.sort_values(["date","game"]).reset_index(drop=True)
    pre=pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    ly=ps[ps.season==S-1].set_index("player"); reg=set(ly.index[(ly.mpg>=20)&(ly.gp>=30)])
    pg["cum"]=pg.groupby("player")["min"].cumsum()-pg["min"]; pg["pgp"]=pg.groupby("player").cumcount()
    tg=pg[["team","game","date"]].drop_duplicates().sort_values(["date","game"]); tg["tgp"]=tg.groupby("team").cumcount()
    pg=pg.merge(tg[["team","game","tgp"]],on=["team","game"])
    pg=pg.sort_values(["player","team","date","game"]); prev=pg.groupby(["player","team"]).tgp.shift(1)
    pg["missed"]=(pg.tgp-prev-1).fillna(0).clip(lower=0)
    pg["sw"]=5*pg["min"]/pg.groupby(["game","team"])["min"].transform("sum")*np.where(pg.team==pg.game.map(hm),1.0,-1.0)
    inpre=pg.player.isin(pre.index); isreg=pg.player.isin(reg)
    pg["grp"]=np.where(~inpre,"out",np.where(isreg,"reg",np.where(pg.pgp>=0.5*pg.tgp,"bench","fringe")))
    pg["rook"]=pg.player.map(pre.kind).eq("rookie"); pg["season"]=S; out.append(pg)
T=pd.concat(out,ignore_index=True); T.to_parquet("thin_table.parquet",index=False)
lo=T[(T.tgp>=15)&(T.cum<300)]
print(len(T),"player-games; mid-season (team game 15+) with <300 season minutes:",len(lo),f"= {lo['min'].sum()/T['min'].sum():.1%} of all minutes")
print((lo.groupby("grp")["min"].sum()/T["min"].sum()).round(4).to_dict())
lay=T[(T.missed>=5)&(T.cum>=600)]; print("layoff returns (5+ team games missed, 600+ season minutes):",len(lay),f"{lay['min'].sum()/T['min'].sum():.2%} of minutes; of which last-season regulars {lay.grp.eq('reg').mean():.0%}")
