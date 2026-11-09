import sqlite3, json, sys, numpy as np, pandas as pd
W="/home/claude/"
G=pd.read_parquet(W+"data/games_all.parquet"); G=G[G.season>=2011].copy()
G["hm"]=G.home_pts-G.away_pts
c=sqlite3.connect("/tmp/odds/odds.sqlite")
tabs={}
for y in range(2007,2023): tabs[y]=f"odds_{y}-{str(y+1)[2:]}_new"
tabs[2023]="2023-24"; tabs[2024]="2024-25"; tabs[2025]="odds_2025-26"
K=[]
for y,t in tabs.items():
    d=pd.read_sql(f'select * from "{t}"',c); d["season"]=y; K.append(d)
K=pd.concat(K); K["date"]=K.Date.str.replace("-","").astype(int)
full={'Atlanta Hawks':'ATL','Boston Celtics':'BOS','Brooklyn Nets':'BKN','New Jersey Nets':'NJN','Charlotte Hornets':'CHA','Charlotte Bobcats':'CHA','Chicago Bulls':'CHI','Cleveland Cavaliers':'CLE','Dallas Mavericks':'DAL','Denver Nuggets':'DEN','Detroit Pistons':'DET','Golden State Warriors':'GSW','Houston Rockets':'HOU','Indiana Pacers':'IND','Los Angeles Clippers':'LAC','LA Clippers':'LAC','Los Angeles Lakers':'LAL','Memphis Grizzlies':'MEM','Miami Heat':'MIA','Milwaukee Bucks':'MIL','Minnesota Timberwolves':'MIN','New Orleans Pelicans':'NOP','New Orleans Hornets':'NOP','New York Knicks':'NYK','Oklahoma City Thunder':'OKC','Orlando Magic':'ORL','Philadelphia 76ers':'PHI','Phoenix Suns':'PHX','Portland Trail Blazers':'POR','Sacramento Kings':'SAC','San Antonio Spurs':'SAS','Toronto Raptors':'TOR','Utah Jazz':'UTA','Washington Wizards':'WAS','Seattle SuperSonics':'SEA'}
K["h"]=K.Home.map(full); K["a"]=K.Away.map(full)
print("kyle unmapped:",set(K.Home[K.h.isna()])|set(K.Away[K.a.isna()]))
for q in ["Spread","OU","ML_Home","ML_Away","Points","Win_Margin"]: K[q]=pd.to_numeric(K[q],errors="coerce")
K=K[["season","date","h","a","Spread","OU","ML_Home","ML_Away","Points","Win_Margin"]]
F=pd.DataFrame(json.load(open("/tmp/odds/nba.json")))
F["date"]=F.date.astype(int)
nick={x.split()[-1]:v for x,v in full.items()}; nick.update({'Blazers':'POR','Trailblazers':'POR','Seventysixers':'PHI','NewJersey':'BKN','Golden State':'GSW','LA Clippers':'LAC','Oklahoma City':'OKC','Warriors':'GSW','Clippers':'LAC','Thunder':'OKC'})
F["home_team"]=F.home_team.astype(str);F["away_team"]=F.away_team.astype(str);print(set(F.home_team)-set(nick),set(F.away_team)-set(nick))
F["h"]=F.home_team.map(nick); F["a"]=F.away_team.map(nick)
for q in ["home_open_spread","home_close_spread","home_close_ml","away_close_ml","home_final","away_final","close_over_under"]: F[q]=pd.to_numeric(F[q],errors="coerce")
F=F[["date","h","a","home_open_spread","home_close_spread","home_close_ml","away_close_ml","home_final","away_final","close_over_under"]]
tri=lambda s:s.replace({"NOH":"NOP","NOK":"NOP","NJN":"BKN","CHH":"CHA"})
G["h"]=tri(G.htm); G["a"]=tri(G.vtm); K["h"]=tri(K.h); K["a"]=tri(K.a); F["h"]=tri(F.h); F["a"]=tri(F.a)
print(sorted(set(G.h)))
def join(G,X,nm):
    # exact date, then +-1 day (UTC/local)
    out=G.merge(X,on=["date","h","a"],how="left",indicator=True)
    print(nm,"exact match",(out._merge=="both").mean().round(4))
    return out.drop(columns="_merge")
M=join(G,K.drop(columns="season"),"kyle"); M=join(M,F,"finned")
M.to_pickle("joined_raw.pkl")
print(M.groupby("season")[["Spread","home_close_spread","home_open_spread"]].agg(lambda s:s.notna().sum()).assign(n=M.groupby("season").size()))
# sign checks
for col in ["Spread","home_close_spread"]:
    m=M[col].notna(); print(col,"corr with home margin",np.corrcoef(M.loc[m,col],M.loc[m,"hm"])[0,1].round(3))
m=M.Win_Margin.notna(); print("kyle Win_Margin==hm",(M.Win_Margin[m]==M.hm[m]).mean().round(4), "abs eq",(M.Win_Margin[m].abs()==M.hm[m].abs()).mean().round(4))
F2=M[M.home_final.notna()]; print("finned score ok",((F2.home_final.astype(float)==F2.home_pts)&(F2.away_final.astype(float)==F2.away_pts)).mean().round(4))
print(M.groupby("season").apply(lambda d: pd.Series({"k_corr":d.Spread.corr(d.hm),"k_ml_corr":d.Spread.corr(-np.sign(d.ML_Home)*1.0),"k_absmean":d.Spread.abs().mean(),"f_corr":d.home_close_spread.corr(d.hm)})).round(3))
