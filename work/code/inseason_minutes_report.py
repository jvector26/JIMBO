import pickle, numpy as np, pandas as pd
D = "/home/claude/data"
R = pd.DataFrame([x for S in range(2014, 2026) for x in pickle.load(open(f"{D}/ism_{S}.pkl", "rb"))])
R = R[R.season != 2019]; R["key"] = R.key.astype(str)
def tab(df, col, n):
    a = df.groupby(["key", "cp"])[[col, n]].sum(); return np.sqrt(a[col] / a[n]).unstack()
T = tab(R, "team_sse", "team_n"); G = tab(R, "sse", "n"); P = tab(R[R.key != "oracle"], "p_wsse", "p_w")
pd.set_option("display.width", 200)
T["avg"] = T.mean(1); G["avg"] = G.mean(1); P["avg"] = P.mean(1)
print("TEAM rest-of-season margin RMSE (pts/game)"); print(T.sort_values("avg").round(3).head(14).to_string())
print(T.loc[["oracle", "(1000000000.0, 0.0, None, 1.0)", "(0, 0.0, None, 1.0)"]].round(3).to_string())
print("\nGAME RMSE"); print(G.sort_values("avg").round(3).head(6).to_string()); print(G.loc[["oracle"]].round(3).to_string())
print("\nPLAYER minutes/game RMSE (weighted by actual)"); print(P.sort_values("avg").round(2).head(8).to_string())
print(P.loc[["(1000000000.0, 0.0, None, 1.0)", "(0, 0.0, None, 1.0)"]].round(2).to_string())
