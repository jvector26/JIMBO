"""Variant preseason ratings for the lines bar: pre_P rating shifted by SLOPE x (change in the lineup part), window N=S-1 residuals
(bt_split_ratings; same construction as code/sepposs/sep_pre.py). Written to /home/claude/lines/{tag}pre_P_{S}.parquet."""
import json, numpy as np, pandas as pd
D="/home/claude/data"; SLOPE=pd.read_csv(f"{D}/calib.csv",index_col=0).iloc[:,0]["slope"]
R=pd.read_parquet(f"{D}/bt_split_ratings.parquet"); ps=pd.read_parquet(f"{D}/player_seasons.parquet")
S=pd.read_parquet(f"{D}/sep_poss.parquet"); K=json.load(open(f"{D}/sep_k.json"))
S["co"]=(S.neff_o/S.nnaive_o).clip(0,1)/K["c_neff_o"]; S["cd"]=(S.neff_d/S.nnaive_d).clip(0,1)/K["c_neff_d"]
print("ratings seasons",R.season.min(),R.season.max(),"slope",SLOPE)
for tag in ["woff4_","shr_","neff_"]:
    mv=[]
    for Sn in range(2014,2026):
        N=Sn-1; r=R[R.season==N].set_index("player"); s=S[S.N==N].set_index("player")
        Pw=ps[ps.season.between(N-2,N)].groupby("player").on_poss.sum()
        d=pd.read_parquet(f"{D}/pre_P_{Sn}.parquet"); P=d.player.map(Pw).fillna(0)
        ro=d.player.map(r.res_o); rd=d.player.map(r.res_d)
        if tag=="woff4_": cut=0.2*ro
        else:
            fo=d.player.map(s.co).fillna(1.0) if tag=="neff_" else 1.0; fd=d.player.map(s.cd).fillna(1.0) if tag=="neff_" else 1.0
            cut=0.6*ro*(1/3)*np.clip(1-P*fo/10000,0,1)+0.8*rd*(1/3)*np.clip(1-P*fd/10000,0,1)
        cut=cut.fillna(0); has=d.player.isin(r.index)&d.kind.ne("newcomer")&d.kind.ne("rookie")
        d.loc[has,"rating"]=d.rating[has]-SLOPE*cut[has]; mv.append((SLOPE*cut[has]).std() if has.any() else 0.0)
        d.to_parquet(f"{tag}pre_P_{Sn}.parquet",index=False)
    print(tag,"sd of rating change",np.mean(mv).round(3))
