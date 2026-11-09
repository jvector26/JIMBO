import numpy as np, pandas as pd, sys
W="/home/claude/"
B=pd.read_parquet(W+"data/backtest_components.parquet").merge(pd.read_parquet(W+"data/bt_split.parquet"),on=["game","home_team","away_team"])
F=[c for c in B.columns if c.startswith("f:")]
B["x"]=B[F].sum(axis=1)+B.unrated+.6*B.res_o+.8*B.res_d
# LOSO-calibrated JIMBO margin (same as research bar)
B["J"]=np.nan
for s in B.season.unique():
    tr=B.season!=s; b=np.polyfit(B.x[tr],B.margin[tr],1); B.loc[~tr,"J"]=np.polyval(b,B.x[~tr])
print("LOSO all games",round(((B.margin-B.J)**2).mean()**.5,4))
L=pd.read_csv("hist_game_lines.csv"); L["game"]=L.game.astype(int); B["game"]=B.game.astype(int)
D=B.merge(L[["game","date","hm","open","close","line","src"]],on="game",how="left")
assert (D.hm==D.margin).all()
D.to_pickle("bt_with_lines.pkl")
def clus(x,y,g):
    # through-origin slope with season-clustered se
    b=(x*y).sum()/(x*x).sum(); r=y-b*x
    se=np.sqrt(sum(((x*r)[g==s].sum())**2 for s in np.unique(g)))/(x*x).sum()
    return b,se
def rep(D,col,name):
    d=D[D[col].notna()]; g=d.season.values
    ej=d.margin-d.J; el=d.margin-d[col]; gap=(d.J-d[col]).values
    b,se=clus(gap,el.values,g)
    dm=(ej**2-el**2).groupby(d.season).mean()
    # encompassing
    X=np.c_[np.ones(len(d)),d.J,d[col]]; co=np.linalg.lstsq(X,d.margin,rcond=None)[0]
    # ATS record siding with JIMBO at |gap|>=3
    m=np.abs(gap)>=3; w=(np.sign(gap[m])==np.sign(el.values[m])).sum(); l=(np.sign(gap[m])==-np.sign(el.values[m])).sum()
    print(f"{name:26s} n {len(d):5d} | RMSE JIMBO {np.sqrt((ej**2).mean()):.3f} line {np.sqrt((el**2).mean()):.3f} | dMSE t {dm.mean()/(dm.std()/np.sqrt(len(dm))):.1f} | slope {b:+.3f} se {se:.3f} t {b/se:+.1f} | free coefs J {co[1]:.2f} line {co[2]:.2f} | sd gap {gap.std():.2f} | ATS gap3+ {w}-{l} ({w/(w+l):.3f})")
    return b,se
print("\n== line = close where known, else open (2015-21); SBR single line 2022+ ==")
rep(D,"line","all 2015-25")
rep(D[D.season<=2019],"line","2015-19"); rep(D[D.season>=2020],"line","2020-25")
for s in sorted(D.season.unique()): rep(D[D.season==s],"line",str(s))
print("\n== opening line only (2015-21) ==")
rep(D,"open","open 2015-21"); 
print("== closing line only (2015-21, finned) =="); rep(D,"close","close 2015-25 (incl sbr)"); rep(D[D.season<=2021],"close","close 2015-21")
