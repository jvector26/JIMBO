import shutil, subprocess, numpy as np, pandas as pd
D="/home/claude/data"; ps=pd.read_parquet(f"{D}/player_seasons.parquet"); SL=1.0501
def run(tag,thin=0.0,cut=2000,smooth=False):
    for S in range(2014,2026):
        d=pd.read_parquet(f"{D}/pre_P_{S}.parquet"); hp=ps[ps.season.between(S-3,S-1)].groupby("player").on_poss.sum(); P=d.player.map(hp).fillna(0); v=d.kind.eq("vet")
        w=np.clip(1-P/cut,0,1) if smooth else (P<cut).astype(float); d.loc[v,"rating"]+=thin*SL*w[v]
        d.to_parquet(f"{D}/{tag}pre_P_{S}.parquet",index=False); shutil.copy(f"{D}/pre_T_{S}.parquet",f"{D}/{tag}pre_T_{S}.parquet")
    subprocess.run(["python3","-W","ignore","/home/claude/lines/age_eval_dump.py",tag],capture_output=True,text=True)
    return pd.read_csv(f"ae_{tag}.csv")
b=run("rd0_"); e0=(b.y-b["mod"])**2
print(f"current: model {np.sqrt(e0.mean()):.3f} wins ({len(b)} team-seasons)")
def rep(lab,a):
    e=(a.y-a["mod"])**2; d=(e-e0).groupby(a.season).mean(); h=lambda m:np.sqrt(e[m].mean())-np.sqrt(e0[m].mean())
    print(f"{lab:44s} model {np.sqrt(e.mean()):.3f} ({np.sqrt(e.mean())-np.sqrt(e0.mean()):+.3f}) | seasons better {(d<0).sum()}/{len(d)} | paired t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.1f} | halves {h(a.season<=2019):+.3f}/{h(a.season>=2020):+.3f}")
for th in [-0.75,-1.46,-2.2,-3.0]: rep(f"thin vets (<2000 poss) {th:+.2f}",run("rd1_",thin=th))
for cut in [1000,3000,4000]: rep(f"thin vets (<{cut} poss) -1.46",run("rd2_",thin=-1.46,cut=cut))
for cut,th in [(4000,-2.0),(6000,-2.0),(4000,-3.0)]: rep(f"smooth: {th:+.1f} x (1 - poss/{cut})",run("rd3_",thin=th,cut=cut,smooth=True))
