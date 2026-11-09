import numpy as np, pandas as pd
D=pd.read_pickle("m4_with_lines.pkl"); D["off"]=D.J-D.Jbase
T=[]
for S in range(2014,2026):
    t=pd.read_parquet(f"m4tr_{S}.parquet"); T.append(t)
T=pd.concat(T); gd=D.set_index("game").date
T=T[T.game.isin(gd.index)]; T["gdate"]=T.game.map(gd); T=T[T.cp_date<=T.gdate].sort_values("cp_date").groupby("game").tail(1)
D=D.merge(T[["game","pred","pred_tr"]],on="game"); D=D[D.pred_tr.notna()].copy()
for c,o in [("pred","Ja"),("pred_tr","Jt")]:
    D[o]=np.nan
    for s in D.season.unique():
        tr=D.season!=s; b=np.polyfit(D[c][tr],D.margin[tr],1); D.loc[~tr,o]=np.polyval(b,D[c][~tr])+D.off[~tr]
    print(o,"slope",np.polyfit(D[c],D.margin,1).round(3))
rm=lambda e:np.sqrt((e**2).mean())
def clus(x,yv,g):
    b=(x*yv).sum()/(x*x).sum(); r=yv-b*x; se=np.sqrt(sum(((x*r)[g==s].sum())**2 for s in np.unique(g)))/(x*x).sum(); return b,se
def rep(d,col,name,J):
    d=d[d[col].notna()]; g=d.season.values; ej=(d.margin-d[J]).values; el=(d.margin-d[col]).values; gap=(d[J]-d[col]).values
    b,se=clus(gap,el,g); m=np.abs(gap)>=3; w=(np.sign(gap[m])==np.sign(el[m])).sum(); l=(np.sign(gap[m])==-np.sign(el[m])).sum()
    print(f"{name:34s} n {len(d):5d} | RMSE J {rm(ej):.3f} line {rm(el):.3f} | weight on J {b:+.3f} se {se:.3f} t {b/se:+.1f} | sd gap {gap.std():.2f} | ATS gap3+ {w}-{l} ({w/(w+l):.3f})")
for J,nm in [("Ja","ACTUAL minutes"),("Jt","TRAILING 5-game minutes")]:
    print("==",nm)
    rep(D,"line","best line, all 2014-25",J); rep(D[D.season<=2019],"line","  2014-19",J); rep(D[D.season>=2020],"line","  2020-25",J)
    rep(D[D.season<=2021],"close","true close 2014-21",J); rep(D[D.season<=2021],"open","open 2014-21",J); rep(D[D.season>=2022],"line","SBR line 2022-25",J)
# three-way: margin-line on (Jt-line) and (Ja-Jt)
d=D[D.line.notna()]; X=np.c_[d.Jt-d.line,d.Ja-d.Jt]; yv=(d.margin-d.line).values; b=np.linalg.lstsq(X,yv,rcond=None)[0]; print("joint: (Jt-line)",b[0].round(3),"(Ja-Jt, the who-played part)",b[1].round(3), "sd Ja-Jt",(d.Ja-d.Jt).std().round(2))
d=D[D.close.notna()&(D.season<=2021)]; X=np.c_[d.Jt-d.close,d.Ja-d.Jt]; b=np.linalg.lstsq(X,(d.margin-d.close).values,rcond=None)[0]; print("true close 2014-21 joint:",b.round(3))
# does the line move toward JIMBO? close-open on Jt-open
d=D[D.close.notna()&D.open.notna()&(D.season<=2021)]; x=(d.Jt-d.open).values; yv=(d.close-d.open).values; print("line move (close-open) on (Jt-open): slope",clus(x,yv,d.season.values), " on (Ja-open):",clus((d.Ja-d.open).values,yv,d.season.values))
print("== robustness (trailing minutes) ==")
d=D[(D.season<=2021)&D.close.notna()&D.open.notna()]
print("|gap|>10 share",(np.abs(d.Jt-d.close)>10).mean().round(4),"|close-open|>4 share",((d.close-d.open).abs()>4).mean().round(4))
rep(d[(d.close-d.open).abs()<=3],"close","close, |close-open|<=3","Jt")
rep(d[(d.Jt-d.close).abs()<=8],"close","close, |gap|<=8","Jt")
rep(D[(D.Jt-D.line).abs()<=8],"line","best line, |gap|<=8","Jt")
rep(D[D.gp>=10],"line","best line, team games 10+","Jt")
rep(D[D.gp<25],"line","best line, team games <25","Jt"); rep(D[D.gp>=25],"line","best line, team games 25+","Jt")
# season-level slopes -> t across seasons
sl=[]
for s,dd in D[D.line.notna()].groupby("season"):
    x=(dd.Jt-dd.line).values; yv=(dd.margin-dd.line).values; sl.append((x*yv).sum()/(x*x).sum())
sl=np.array(sl); print("per-season weights",sl.round(2),"mean",sl.mean().round(3),"t",(sl.mean()/(sl.std(ddof=1)/np.sqrt(len(sl)))).round(1),"positive",(sl>0).sum(),"of",len(sl))
