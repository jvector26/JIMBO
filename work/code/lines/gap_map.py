"""GAP MAP (session 47): where does standalone JIMBO lose to the line? Lines are a yardstick only.
J = standalone in-season JIMBO (M4 + offsets, LOSO-calibrated), trailing-minutes version (pregame) and actual-minutes version.
Part A: excess squared error vs the line by situation (share of the total gap).
Part B: which pregame factors predict JIMBO's misses (y - J), and how much of the gap each closes (LOSO) -- no line used in the fit."""
import numpy as np, pandas as pd, lines_bar as LB
from lines_bar import G,y,seas,SEAS,loso
b=LB.load(""); line=G.line.to_numpy()
F=pd.read_parquet("gap_features.parquet")
H=F[F.home].set_index("game"); A=F[~F.home].set_index("game")
for c in ["gp","winpct","rest","g4","age","young","newp","acq","absent","returning"]:
    G["h_"+c]=G.game.map(H[c]).to_numpy(); G["a_"+c]=G.game.map(A[c]).to_numpy()
m=np.isfinite(line)&np.isfinite(b["tr"])&G.h_age.notna().to_numpy()&G.a_age.notna().to_numpy()
Jt=loso(np.c_[b["tr"],b["off"]],m); Ja=loso(np.c_[b["act"],b["off"]],m); Lc=loso(line,m)
et=(y-Jt)**2; ea=(y-Ja)**2; el=(y-line)**2
r=lambda e,mm:np.sqrt(e[mm].mean())
print(f"games {m.sum()} | line {r(el,m):.3f} | JIMBO pregame minutes {r(et,m):.3f} (gap {r(et,m)-r(el,m):+.3f}) | JIMBO actual minutes {r(ea,m):.3f} (gap {r(ea,m)-r(el,m):+.3f})")
TOT=(et-el)[m].sum(); TOTA=(ea-el)[m].sum()
def seg(title,labels,masks):
    print(f"\n-- {title}"); print(f"{'':30s}{'games':>7s}{'share':>7s}{'line':>8s}{'JIMBO':>8s}{'gap':>7s}{'%ofgap':>8s} | {'gap(act)':>8s}{'%ofgap':>8s} | bias J-line")
    for lb,mk in zip(labels,masks):
        mk=mk&m; n=mk.sum()
        if n<50: continue
        print(f"{lb:30s}{n:7d}{n/m.sum():7.1%}{r(el,mk):8.2f}{r(et,mk):8.2f}{r(et,mk)-r(el,mk):+7.2f}{(et-el)[mk].sum()/TOT:8.1%} | {r(ea,mk)-r(el,mk):+8.2f}{(ea-el)[mk].sum()/TOTA:8.1%} | {(Jt-line)[mk].mean():+.2f}")
gp=(G.h_gp+G.a_gp).to_numpy()/2
seg("season stage (team games played)",["3-10","10-25","25-50","50-70","70+"],[(gp<10),(gp>=10)&(gp<25),(gp>=25)&(gp<50),(gp>=50)&(gp<70),gp>=70])
hb=(G.h_rest==1).to_numpy(); ab=(G.a_rest==1).to_numpy()
seg("back-to-backs (no rest term in this backtest; production has -2.0)",["neither","home only on B2B","away only on B2B","both"],[~hb&~ab,hb&~ab,~hb&ab,hb&ab])
ab_=np.maximum(G.h_absent,G.a_absent).to_numpy()
seg("regulars missing tonight (max of two teams, share of recent minutes)",["<10%","10-20%","20-30%","30%+"],[ab_<.1,(ab_>=.1)&(ab_<.2),(ab_>=.2)&(ab_<.3),ab_>=.3])
rt=np.maximum(G.h_returning,G.a_returning).to_numpy()
seg("players back tonight after 5+ games away (share of tonight's minutes)",["none","<10%","10%+"],[rt==0,(rt>0)&(rt<.1),rt>=.1])
aq=np.maximum(G.h_acq,G.a_acq).to_numpy()
seg("new to the team in its last 10 games (share of recent minutes)",["none","<10%","10%+"],[aq==0,(aq>0)&(aq<.1),aq>=.1])
tank=lambda p:((G[p+"_gp"]>=50)&(G[p+"_winpct"]<.35)).to_numpy(); good=lambda p:((G[p+"_gp"]>=50)&(G[p+"_winpct"]>=.6)).to_numpy()
ht,at=tank("h"),tank("a"); late=gp>=50
seg("late season (50+ games): bad teams (<35% wins)",["no bad team","bad team vs other","both bad"],[late&~ht&~at,late&(ht^at),late&ht&at])
last=gp>=74; seg("final 8 games",["final 8 games","rest of season"],[last,~last])
ad=(G.h_age-G.a_age).to_numpy(); seg("age gap (home minus away, minutes-weighted)",["home 2+ yrs younger","within 2 yrs","home 2+ yrs older"],[ad<=-2,(ad>-2)&(ad<2),ad>=2])
yg=np.maximum(G.h_young,G.a_young).to_numpy(); seg("young rotation (share of minutes aged 22 or less, max of two teams)",["<20%","20-40%","40%+"],[yg<.2,(yg>=.2)&(yg<.4),yg>=.4])
nw=np.maximum(G.h_newp,G.a_newp).to_numpy(); seg("rookies / unrated newcomers (share of recent minutes, max)",["<10%","10-25%","25%+"],[nw<.1,(nw>=.1)&(nw<.25),nw>=.25])
al=np.abs(line); seg("size of favourite (line)",["0-3","3.5-7","7.5-12","12.5+"],[al<=3,(al>3)&(al<=7),(al>7)&(al<=12),al>12])
alt=G.htm.isin(["DEN","UTA"]).to_numpy(); seg("altitude home (DEN, UTA)",["DEN/UTA home","other"],[alt,~alt])
seg("by season",[str(s) for s in SEAS],[seas==s for s in SEAS])
# team-season persistence: how much of each forecast's error is a team-level miss that lasts all season
T=pd.concat([pd.DataFrame({"s":seas[m],"t":G.home_team[m],"ej":(y-Jt)[m],"el":(y-line)[m],"ea":(y-Ja)[m]}),pd.DataFrame({"s":seas[m],"t":G.away_team[m],"ej":-(y-Jt)[m],"el":-(y-line)[m],"ea":-(y-Ja)[m]})])
ts=T.groupby(["s","t"]).agg(n=("ej","size"),ej=("ej","mean"),el=("el","mean"),ea=("ea","mean"))
print("\n-- team-season average miss (points per game, sd across 330 team-seasons; noise alone ~",round(12.9/np.sqrt(ts.n.mean()),2),")")
print("   line",ts.el.std().round(2),"| JIMBO pregame",ts.ej.std().round(2),"| JIMBO actual minutes",ts.ea.std().round(2))
# Part B
def cl_t(x,e,g):
    x=x-x.mean(); bb=(x*e).sum()/(x*x).sum(); rr=e-bb*x; return bb,bb/(np.sqrt(sum(((x*rr)[g==s].sum())**2 for s in np.unique(g)))/(x*x).sum())
prog=np.clip(gp/82,0,1)
feats={"back-to-back (home - away)":hb.astype(float)-ab.astype(float),
 "3rd game in 4 nights (h - a)":((G.h_g4>=2)&(G.h_rest>1)).to_numpy(float)-((G.a_g4>=2)&(G.a_rest>1)).to_numpy(float),
 "long rest 3+ days (h - a)":(G.h_rest>=4).to_numpy(float)-(G.a_rest>=4).to_numpy(float),
 "regulars missing tonight (rating-weighted, actual box)":b["act"]-b["tr"],
 "new-to-team share (h - a)":(G.h_acq-G.a_acq).to_numpy(),
 "age (h - a), years":ad, "age x season progress":ad*prog,
 "young-minutes share (h - a)":(G.h_young-G.a_young).to_numpy(), "young share x season progress":(G.h_young-G.a_young).to_numpy()*prog,
 "rookie/newcomer share (h - a)":(G.h_newp-G.a_newp).to_numpy(),
 "late-season bad team (h - a)":(ht.astype(float)-at.astype(float)),
 "late-season good team 60%+ (h - a)":(good("h").astype(float)-good("a").astype(float)),
 "altitude home":alt.astype(float),
}
gap=r(et,m)-r(el,m); base_X=np.c_[b["tr"],b["off"]]
print(f"\n-- PART B: factors that predict JIMBO's pregame misses (final - JIMBO). Gap to line = {gap:.3f}")
print(f"{'factor':52s}{'coef':>8s}{'t':>7s}{'dRMSE':>9s}{'t':>6s}{'seasons':>9s}{'% gap closed':>14s}")
res={}
for nm,x in feats.items():
    x=np.where(np.isfinite(x),x,0.0); c,t=cl_t(x[m],(y-Jt)[m],seas[m]); pv=loso(np.c_[base_X,x],m); a=LB.pair(pv,Jt,m); res[nm]=x
    print(f"{nm:52s}{c:+8.3f}{t:+7.1f}{a[0]:+9.4f}{a[1]:+6.1f}{a[2]:>6d}/{a[3]}{-a[0]/gap:13.1%}")
# league home edge known before the season: last season's mean home margin (pregame-knowable)
hm_prev=LB.GA.groupby("season").margin.mean().shift(1); hca=G.season.map(hm_prev).to_numpy()
c,t=cl_t(hca[m],(y-Jt)[m],seas[m]); pv=loso(np.c_[base_X,hca],m); a=LB.pair(pv,Jt,m)
print(f"{'last season league home edge (pts)':52s}{c:+8.3f}{t:+7.1f}{a[0]:+9.4f}{a[1]:+6.1f}{a[2]:>6d}/{a[3]}{-a[0]/gap:13.1%}")
res["hca"]=hca
def step(nm,cols,minutes="tr",ref=None):
    pv=loso(np.column_stack([b[minutes],b["off"]]+[res[k] for k in cols]),m); g_=r((y-pv)**2,m)-r(el,m)
    extra="" if ref is None else "  step t %+.1f, %d/%d seasons, halves %+.4f/%+.4f"%(LB.pair(pv,ref,m)[1:2]+LB.pair(pv,ref,m)[2:4]+LB.pair(pv,ref,m)[4:6])
    print(f"  {nm:58s} error {r((y-pv)**2,m):.3f}  gap to line {g_:+.3f}{extra}"); return pv
print("\n-- STAIRCASE: gap to the line (line = %.3f)"%r(el,m))
p0=step("backtest JIMBO, pregame minutes",[])
p1=step("+ back-to-back (already in production)",["back-to-back (home - away)"],ref=p0)
p2=step("+ last season's home edge (production uses 2.0)",["back-to-back (home - away)","hca"],ref=p1)
ext=["age (h - a), years","young-minutes share (h - a)","late-season bad team (h - a)","late-season good team 60%+ (h - a)","altitude home","new-to-team share (h - a)","rookie/newcomer share (h - a)"]
p3=step("+ age, young share, late-season, altitude, new players",["back-to-back (home - away)","hca"]+ext,ref=p2)
p4=step("+ knowing who plays tonight (actual minutes)",["back-to-back (home - away)","hca"],minutes="act",ref=p2)
# JIMBO's own favourite size (no selection on the line)
aj=np.abs(Jt); fav=np.sign(Jt)
print("\n-- by JIMBO's own favourite size: favourite-signed averages")
for lb,mk in [("0-3",aj<=3),("3-7",(aj>3)&(aj<=7)),("7-12",(aj>7)&(aj<=12)),("12+",aj>12)]:
    mk=mk&m; print(f"  JIMBO fav by {lb:5s} n {mk.sum():5d} | JIMBO {(fav*Jt)[mk].mean():5.2f} line {(fav*line)[mk].mean():5.2f} actual {(fav*y)[mk].mean():5.2f} | gap {r(et,mk)-r(el,mk):+.2f}")
# stage x learning: early games
print("\n-- early season detail (pregame minutes): gap by team games played")
for lo,hi in [(3,6),(6,10),(10,15),(15,20),(20,25),(25,35)]:
    mk=m&(gp>=lo)&(gp<hi); print(f"  games {lo}-{hi}: n {mk.sum()} gap {r(et,mk)-r(el,mk):+.2f} (actual minutes {r(ea,mk)-r(el,mk):+.2f})")
print("\n-- remaining gap WITH actual minutes + back-to-back (p4), by how well JIMBO knows the players")
e4=(y-p4)**2; T4=(e4-el)[m].sum()
def seg4(lb,mk):
    mk=mk&m; d=(e4-el)[mk]; se=d.std()/np.sqrt(mk.sum())/(2*12.9)
    print(f"  {lb:44s} n {mk.sum():5d} ({mk.sum()/m.sum():5.1%}) gap {r(e4,mk)-r(el,mk):+.2f} (se {se:.2f})  share of remaining gap {d.sum()/T4:6.1%}")
lowk=(nw>=.25)|(aq>=.10)
seg4("rookies/newcomers 25%+ of minutes",nw>=.25); seg4("new-to-team 10%+",aq>=.10); seg4("either (thin history)",lowk); seg4("neither",~lowk)
seg4("final 12 games (70+)",gp>=70); seg4("games 10-25",(gp>=10)&(gp<25)); seg4("games 25-70",(gp>=25)&(gp<70)); seg4("games 3-10",gp<10)
seg4("thin history AND games 70+",lowk&(gp>=70)); seg4("thin history, games <70",lowk&(gp<70)); seg4("well-known rosters, games 70+",~lowk&(gp>=70)); seg4("well-known rosters, games <70",~lowk&(gp<70))
seg4("late bad team vs other",late&(ht^at))
