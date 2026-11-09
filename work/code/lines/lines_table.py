import numpy as np, pandas as pd
M=pd.read_pickle("joined_raw.pkl")
sp=M.Spread.where(M.Spread.abs()<=30)
sg=np.where(M.ML_Home<M.ML_Away,1,-1)
k=np.where(M.season<=2021,sg*sp.abs(),sp); k=pd.Series(k,index=M.index).where(M.ML_Home.notna()|(M.season>2021))
fsg=np.where(M.home_close_ml<M.away_close_ml,1,-1)
fc=(fsg*M.home_close_spread.abs().where(M.home_close_spread.abs()<=30)).where(M.home_close_ml.notna())
fo=(fsg*M.home_open_spread.abs().where(M.home_open_spread.abs()<=30)).where(M.home_close_ml.notna())
L=M[["season","game","date","h","a","hm"]].copy()
L["open"]=np.where(M.season<=2021,fo.fillna(k),np.nan)      # home favored by (points), + = home
L["close"]=np.where(M.season<=2021,fc,k)                    # 2022+: single SBR line (timing unknown, treated as close)
L["line"]=L.close.fillna(L.open)
L["src"]=np.where(M.season>2021,"sbr_single",np.where(fc.notna(),"close","open_fallback")); L.loc[L.line.isna(),"src"]="none"
L.to_csv("hist_game_lines.csv",index=False)
print(pd.crosstab(L.season,L.src))
print(L.groupby("season").apply(lambda d:pd.Series({"n":d.line.notna().sum(),"rmse":((d.hm-d.line)**2).mean()**.5,"bias":(d.hm-d.line).mean(),"corr":d.line.corr(d.hm)})).round(3))
