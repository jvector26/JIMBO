"""Team-game situation features for the gap map (session 47). All 'tr' features use the team's previous 5 games (pregame-knowable);
'absent'/'returning' use tonight's actual box (diagnostic only). Output /home/claude/lines/gap_features.parquet (one row per team-game)."""
import numpy as np, pandas as pd
D="/home/claude/data"; bd=pd.read_parquet(f"{D}/birthdates.parquet").set_index("player").birthdate
rows=[]
for S in range(2015,2026):
    g=pd.read_parquet(f"{D}/games_{S}.parquet"); g["dt"]=pd.to_datetime(g.date.astype(str)); g["margin"]=g.home_pts-g.away_pts
    pg=pd.read_parquet(f"{D}/player_games_{S}.parquet"); pg=pg[pg["min"]>0]
    pre=pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player").set_index("player")
    kind=pre.kind; rating=pre.rating
    gd=g.set_index("game").dt; gh=g.set_index("game").home_team; gm=g.set_index("game").margin
    for t,dft in pg.groupby("team"):
        gl=pd.Series(dft.game.unique()); gl=gl.iloc[np.argsort(gl.map(gd).values,kind="stable")].to_numpy()
        mm=dft.pivot_table(index="game",columns="player",values="min",aggfunc="sum").reindex(gl).fillna(0)
        tr=mm.rolling(5,min_periods=3).mean().shift(1)
        dts=pd.Series(gl).map(gd).to_numpy(); first=(mm>0).values.argmax(axis=0)  # index of first game with team, per player
        age_at=lambda d:((d-bd.reindex(mm.columns))/np.timedelta64(1,"D")/365.25).to_numpy()
        won=np.array([(gm[x]>0)==(gh[x]==t) for x in gl],float); cw=np.r_[0,np.cumsum(won)[:-1]]
        isnew=np.array([kind.get(p,"none")!="vet" for p in mm.columns])
        for i,gid in enumerate(gl):
            r={"season":S,"game":gid,"team":t,"home":gh[gid]==t,"gp":i,"winpct":cw[i]/i if i else np.nan,
               "rest":(dts[i]-dts[i-1])/np.timedelta64(1,"D") if i else 9.0,
               "g4":int(((dts[:i]>=dts[i]-np.timedelta64(3,"D"))).sum()) if i else 0}
            v=tr.iloc[i].to_numpy()
            if i>=3 and np.nansum(v)>0:
                w=v/v.sum(); a=age_at(dts[i]); ok=np.isfinite(a)
                r["age"]=(w[ok]*a[ok]).sum()/w[ok].sum(); r["young"]=w[ok&(a<=22.5)].sum()
                r["newp"]=w[isnew].sum()
                r["acq"]=w[(first>=max(i-10,1))&(first>0)].sum() if i>=10 else 0.0      # joined the team in its last 10 games (not opening night)
                played=mm.iloc[i].to_numpy()>0; r["absent"]=w[~played].sum()
                tm=mm.iloc[i].to_numpy(); r["returning"]=tm[v==0].sum()/tm.sum()
            rows.append(r)
    print(S,len(rows),flush=True)
F=pd.DataFrame(rows); F.to_parquet("gap_features.parquet",index=False); print(F.describe().T.round(3).to_string())
