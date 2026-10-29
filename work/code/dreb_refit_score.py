import numpy as np, pandas as pd
exec(open("defsub_score.py").read().split("H1, H2")[0])
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
R = pd.concat([pd.read_parquet(f"{D}/dreb_refit_{N}.parquet") for N in range(2014, 2025)])
B = B.merge(R, on=["game", "home_team", "away_team"])
chk = np.abs(B["res_d_1.0"] - B.res_d).max(); print("reproduce res_d max diff", round(chk, 4))
boxnd = (box - B["f:d_dreb"]).to_numpy(); dr = B["f:d_dreb"].to_numpy()
def pred(m, wo=.6, wd=.8): return boxnd + m*dr + wo*B[f"res_o_{m}"].to_numpy() + wd*B[f"res_d_{m}"].to_numpy()
for m in [0.25, 0.5, 1.0]:
    p = pred(m); print(f"refit x{m}: LOSO {loso(p[:,None]):.4f}   halves", end=" ")
    for msk in (H1, H2):
        # in-half LOSO
        yy, ss = y[msk], seas[msk]; X = np.c_[np.ones(msk.sum()), p[msk]]; e=[]
        for s in np.unique(ss):
            b = np.linalg.lstsq(X[ss!=s], yy[ss!=s], rcond=None)[0]; e.append(yy[ss==s]-X[ss==s]@b)
        print(f"{np.sqrt((np.concatenate(e)**2).mean()):.4f}", end=" ")
    print()
# also old (1.0) in halves for reference computed above; paired test: game-level squared-error difference by season
p1, p5 = pred(1.0), pred(0.5)
def err(p):
    X = np.c_[np.ones(len(y)), p]; e = np.empty(len(y))
    for s in np.unique(seas):
        b = np.linalg.lstsq(X[seas!=s], y[seas!=s], rcond=None)[0]; e[seas==s] = y[seas==s]-X[seas==s]@b
    return e**2
for m in [0.25, 0.5]:
    d = pd.Series(err(pred(m)) - err(p1)).groupby(seas).mean()
    print(f"x{m} vs x1 MSE diff by season:", d.round(2).to_dict(), " t", round(d.mean()/d.std()*np.sqrt(len(d)), 1))
# does w_def/w_off stay right under x0.5?
for wo, wd in [(.6,.8),(.6,1.0),(.4,.8),(.6,.6)]:
    print(f"x0.5 wo {wo} wd {wd}: {loso(pred(0.5,wo,wd)[:,None]):.4f}")
