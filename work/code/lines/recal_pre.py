"""Arbiter: preseason team backtest (projected minutes, so no score-driven-minutes problem). Apply the clean-minutes calibration
to the preseason player ratings (pre_P) piece by piece and score with code/age_eval.py."""
import shutil, subprocess, numpy as np, pandas as pd
D="/home/claude/data"; ps=pd.read_parquet(f"{D}/player_seasons.parquet"); SL=1.0501
def run(tag,slope=1.0,newc=None,thin=0.0,rook=1.0):
    for S in range(2014,2026):
        d=pd.read_parquet(f"{D}/pre_P_{S}.parquet"); hp=ps[ps.season.between(S-3,S-1)].groupby("player").on_poss.sum(); P=d.player.map(hp).fillna(0)
        v=d.kind.eq("vet"); d.loc[v,"rating"]=d.rating[v]*slope
        if thin: d.loc[v&(P<2000),"rating"]+=thin*SL*slope
        if newc is not None: d.loc[d.kind.eq("newcomer"),"rating"]=newc*SL*slope
        d.loc[d.kind.eq("rookie"),"rating"]=d.rating[d.kind.eq("rookie")]*rook*slope
        d.to_parquet(f"{D}/{tag}pre_P_{S}.parquet",index=False); shutil.copy(f"{D}/pre_T_{S}.parquet",f"{D}/{tag}pre_T_{S}.parquet")
    return subprocess.run(["python3","-W","ignore","/home/claude/code/age_eval.py",tag],capture_output=True,text=True).stdout.strip()
print(run("rc0_"))
print("slope x1.075        ",run("rc1_",slope=1.075))
print("newcomers -2.21     ",run("rc2_",newc=-2.21))
print("thin vets -1.46     ",run("rc3_",thin=-1.46))
print("rookies x0.83       ",run("rc4_",rook=0.83))
print("newc + thin         ",run("rc5_",newc=-2.21,thin=-1.46))
print("newc + thin + rook  ",run("rc6_",newc=-2.21,thin=-1.46,rook=0.83))
print("all four            ",run("rc7_",slope=1.075,newc=-2.21,thin=-1.46,rook=0.83))
print("half-size newc+thin ",run("rc8_",newc=-3.26,thin=-0.73))
