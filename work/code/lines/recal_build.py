"""RE-CALIBRATION WITH CLEAN MINUTES (session 47). Rebuild the game-level calibration design (slope, unrated value, rookie mapping;
same design as code/slope_refit.py / rookie_refit.py) with three kinds of minute shares:
  full = all minutes (what the current values were fit on), ng = non-garbage stints, h1 = first half only.
Per-player window ratings rebuilt as in bc_boxdecay.py (box part, decay .1/.4/1) + shooting.talent_feature (o_3eff) + 0.6 res_o + 0.8 res_d
(bt_split_ratings, pre-FT-reweight residuals). Output /home/claude/lines/recal_design.parquet (one row per game x share kind)."""
import sys, json, os, numpy as np, pandas as pd
os.chdir("/home/claude"); sys.path.insert(0,"code"); import impact as I, shooting as SH
from features import OFF_FEATS, DEF_FEATS
D=I.D; W=json.load(open(f"{D}/weights_bt.json")); Wa={**{k:v*.5 for k,v in W["Wo"].items()},**{k:v*.5 for k,v in W["Wd"].items()}}
FE=[x for x in OFF_FEATS+DEF_FEATS if x!="o_3eff"]; f=I.feats(); gm=pd.read_parquet(f"{D}/games_all.parquet"); gm["margin"]=gm.home_pts-gm.away_pts
R=pd.read_parquet(f"{D}/bt_split_ratings.parquet"); ps=pd.read_parquet(f"{D}/player_seasons.parquet")[["season","player","age"]]
dr=pd.read_parquet(f"{D}/draft_r1.parquet")[["year","pick","player"]]; RV=json.load(open(f"{D}/rookie_value.json")); SC=pd.read_parquet(f"{D}/ncaa_scores.parquet")
CURVE=pd.read_csv(f"{D}/aging.csv").set_index("bin").adj.to_dict()
BINS=[(0,21.99,"<=21"),(22,23.99,"22-23"),(24,25.99,"24-25"),(26,27.99,"26-27"),(28,29.99,"28-29"),(30,31.99,"30-31"),(32,33.99,"32-33"),(34,35.99,"34-35"),(36,99,"36+")]
out=[]
for N in range(2014,2025):
    S=N+1; PRE=set(pd.read_parquet(f"{D}/pre_P_{S}.parquet").player); HP=pd.read_parquet(f"{D}/player_seasons.parquet").query(f"{N-2}<=season<={N}").groupby("player").on_poss.sum(); fw=f[f.season.between(N-2,N)].copy(); d=fw.season.map({N-2:.1,N-1:.4,N:1.0}); box=0
    for x in FE:
        wt=(fw.poss_off if x.startswith("o_") else fw.poss_def)*d
        box=box+((fw[x].fillna(0)*Wa[x]*wt).groupby(fw.player).sum()/wt.groupby(fw.player).sum()).fillna(0)
    t3=SH.talent_feature(N); t3=(t3 if isinstance(t3,pd.Series) else pd.Series(t3))*Wa["o_3eff"]
    r=R[R.season==N].set_index("player"); rating=(box.add(t3.reindex(box.index).fillna(0))).add(.6*r.res_o.reindex(box.index).fillna(0)).add(.8*r.res_d.reindex(box.index).fillna(0))
    rows=I.load_rows([N-2,N-1,N]); pace=(rows.groupby("game").poss.sum()/2).mean()
    st=pd.read_parquet(f"{D}/stints_{S}.parquet",columns=["game","t_start","t_end","margin_start","teamA","lineupA","teamB","lineupB"])
    st["dur"]=st.t_end-st.t_start; m_=st.margin_start.abs(); st["ng"]=~(((st.t_start>=2160)&(m_>=20))|((st.t_start>=2520)&(m_>=15))); st["h1"]=st.t_start<1440
    E=[]
    for side in "AB":
        e=st[["game","dur","ng","h1","team"+side,"lineup"+side]].rename(columns={"team"+side:"team","lineup"+side:"lu"}); e=e.assign(player=e.lu.str.split("-")).explode("player"); e=e[e.player.str.len()>0]; e["player"]=e.player.astype(np.int64)
        e["full"]=e.dur; e["ng"]=e.dur*e.ng; e["h1"]=e.dur*e.h1; E.append(e.groupby(["game","team","player"])[["full","ng","h1"]].sum().reset_index())
    x=pd.concat(E).groupby(["game","team","player"],as_index=False).sum().merge(ps[ps.season==S][["player","age"]],on="player",how="left")
    x["rated"]=x.player.isin(box.index); x["rt"]=x.player.map(rating).fillna(0.0)
    rk=x.player.map(dr[dr.year==S].set_index("player").pick); sc=SC[SC.year==S].drop_duplicates("player").set_index("player")
    cs=x.player.map(sc.cscore); col=x.player.map(sc.college).fillna(False).astype(bool); clip=(cs-RV["col_mean"]).clip(-RV["col_cap"],RV["col_cap"])
    pickval=RV["rk"]+RV["rk_lp"]*(np.log(rk)-RV["center_lp"])+RV["rk_age"]*(x.age.fillna(20)-RV["center_age"]); isrk=(~x.rated)&rk.notna(); cc=isrk&col&cs.notna()
    cols={"base":x.rt*x.rated,"rk_col":cc.astype(float),"rk_col_cs":cc.astype(float)*clip.fillna(0),"rk_non_val":(isrk&~cc).astype(float)*pickval.fillna(0),"other_unrated":((~x.rated)&rk.isna()).astype(float),
          "unr_open":((~x.rated)&rk.isna()&x.player.isin(PRE)).astype(float),"unr_late":((~x.rated)&rk.isna()&~x.player.isin(PRE)).astype(float),
          "thin_open":(x.rated&x.player.isin(PRE)&(x.player.map(HP).fillna(0)<2000)).astype(float),"rated_late":(x.rated&~x.player.isin(PRE)).astype(float),
          "age_off":x.rated*x.age.map(lambda a: next((CURVE[l] for lo,hi,l in BINS if lo<=a<=hi),0.0) if a==a else 0.0)}
    for lo,hi,l in BINS: cols["A:"+l]=(x.rated&x.age.between(lo,hi)).astype(float)
    g=gm[gm.season==S][["game","home_team","away_team","margin"]]; hk=pd.MultiIndex.from_arrays([g.game,g.home_team]); ak=pd.MultiIndex.from_arrays([g.game,g.away_team])
    for kind in ["full","ng","h1"]:
        w=5*x[kind]/x.groupby(["game","team"])[kind].transform("sum")*pace/100; o=pd.DataFrame({"game":g.game.values,"season":S,"kind":kind,"margin":g.margin.values})
        for c,v in cols.items():
            t=(w*v).groupby([x.game,x.team]).sum(); o[c]=t.reindex(hk).to_numpy()-t.reindex(ak).to_numpy()
        out.append(o)
    print(S,flush=True)
A=pd.concat(out,ignore_index=True); A.to_parquet("/home/claude/lines/recal_design.parquet",index=False)
B=pd.read_parquet(f"{D}/backtest_components.parquet").merge(pd.read_parquet(f"{D}/bt_split.parquet"),on=["game","home_team","away_team"]); B["x"]=B[[c for c in B.columns if c.startswith("f:")]].sum(axis=1)+.6*B.res_o+.8*B.res_d
c=A[A.kind=="full"].merge(B[["game","x"]].assign(game=B.game.astype(int)),on="game"); print("check full-minutes base vs backtest file: corr",np.corrcoef(c.base,c.x)[0,1].round(4),"sd",c.base.std().round(3),c.x.std().round(3),"mean abs diff",(c.base-c.x).abs().mean().round(3))
