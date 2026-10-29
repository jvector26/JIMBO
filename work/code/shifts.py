# Build player "shifts" (check-in -> check-out) from stints; shift result = sign of team margin during the shift.
import numpy as np, pandas as pd, sys
D="/home/claude/data"
def build(y):
    st=pd.read_parquet(f"{D}/stints_{y}.parquet").sort_values(["game","period","t_start"]).reset_index(drop=True)
    st["idx"]=st.groupby("game").cumcount()
    st["sec"]=(st.t_end-st.t_start).clip(lower=0)
    out=[]
    for me,op in (("A","B"),("B","A")):
        poss=st[f"fg2a{me}"]+st[f"fg3a{me}"]-st[f"oreb{me}"]+st[f"tov{me}"]+0.44*st[f"fta{me}"]
        x=pd.DataFrame({"game":st.game,"idx":st.idx,"team":st[f"team{me}"],"lineup":st[f"lineup{me}"],
                        "m":st[f"pts{me}"]-st[f"pts{op}"],"sec":st.sec,"poss":poss,
                        "garb":st.margin_start.abs()*(1 if True else 1)})
        out.append(x)
    x=pd.concat(out,ignore_index=True)
    x["player"]=x.lineup.str.split("-"); x=x.explode("player").drop(columns="lineup")
    x["player"]=x.player.astype(np.int64)
    x=x.sort_values(["game","team","player","idx"])
    new=(x.game.ne(x.game.shift())|x.player.ne(x.player.shift())|x.idx.ne(x.idx.shift()+1))
    x["shift"]=new.cumsum()
    sh=x.groupby("shift").agg(game=("game","first"),team=("team","first"),player=("player","first"),
                              m=("m","sum"),sec=("sec","sum"),poss=("poss","sum"),start_idx=("idx","first")).reset_index(drop=True)
    sh["sign"]=np.sign(sh.m).astype(np.int8); sh["season"]=y
    return sh
if __name__=="__main__":
    for y in range(int(sys.argv[1]),int(sys.argv[2])+1):
        sh=build(y); sh.to_parquet(f"{D}/shifts_{y}.parquet",index=False)
        print(y,len(sh),"shifts; per team-game",round(len(sh)/sh.groupby(['game','team']).ngroups,1),
              "tie%",round((sh.sign==0).mean()*100,1),"med sec",sh.sec.median(),flush=True)
