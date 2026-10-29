import numpy as np, pandas as pd
exec(open("defsub_score.py").read().split("H1, H2")[0].replace('pd.read_parquet(f"{D}/defsub_{N}.parquet")', 'pd.read_parquet(f"{D}/defsub2_{N}.parquet")'))
H1, H2, ALL = seas <= 2019, seas >= 2020, np.ones(len(y), bool)
print(f"base LOSO {loso(base[:, None]):.4f}")
for name in ["r3", "rrim", "rmid", "rtov", "orb", "rfta", "prim"]:
    for ld in [4000, 16000, 64000]:
        c = f"ds_{name}_{ld}"; X = np.c_[base, B[c]]
        b, t = coefs(X, ALL); b1, t1 = coefs(X, H1); b2, t2 = coefs(X, H2)
        print(f"{c:15s} {loso(X):.4f}  t {t[1]:+.1f}  halves t {t1[1]:+.1f} / {t2[1]:+.1f}")
cs = [f"ds_{n}_16000" for n in ["r3", "rrim", "rmid", "rtov", "orb", "rfta", "prim"]]
print("all @16000", round(loso(np.c_[base, B[cs]]), 4))
box2 = box - B["f:d_opprim"] - B["f:d_oppxq"]; base2 = (box2 + .6 * B.res_o + .8 * B.res_d).to_numpy()
for ld in [4000, 16000, 64000]:
    cs = [f"ds_{n}_{ld}" for n in ["r3", "rrim", "rmid", "rtov", "prim"]]
    print(f"replace raw opp feats w/ adjusted @{ld}: {loso(np.c_[base2, B[cs]]):.4f}")
