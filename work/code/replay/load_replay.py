"""Load replay outputs (out/S_cp/latest.json) + actual results -> one row per season/checkpoint/team."""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, "model"); import jimbo_sim as SIM
CP = [0, 5, 10, 20, 30, 41, 55, 70]
def load():
    rows, sched = [], {}
    for S in range(2014, 2026):
        g = pd.read_parquet(f"/home/claude/data/games_{S}.parquet")
        fw = pd.Series(0, index=SIM.TEAMS)
        for h, a, hp, ap in zip(g.htm, g.vtm, g.home_pts, g.away_pts):
            fw[h if hp > ap else a] += 1
        ng = pd.concat([g.htm, g.vtm]).value_counts()
        for cp in CP:
            L = json.load(open(f"out/{S}_{cp}/latest.json"))
            asof = int(L["asof"].replace("-", ""))
            rg = g[g.date >= asof]
            sched[(S, cp)] = np.array([[SIM.IDX[h], SIM.IDX[a]] for h, a in zip(rg.htm, rg.vtm)], int).reshape(-1, 2)
            # actual rest-of-season margin sum per team
            rm = pd.Series(0.0, index=SIM.TEAMS)
            for h, a, hp, ap in zip(rg.htm, rg.vtm, rg.home_pts, rg.away_pts):
                rm[h] += hp - ap; rm[a] += ap - hp
            for t in L["teams"]:
                rows.append(dict(S=S, cp=cp, t=t["t"], w=t["w"], gp=t["gp"], g_rem=int(ng[t["t"]] - t["gp"]), model=t["model_margin"],
                                 gap=t["gap"], proj_w=t["proj_w"], final_w=int(fw[t["t"]]), rem_margin=rm[t["t"]], line=t["line"]))
    return pd.DataFrame(rows), sched
