"""T8: features.py recipe recomputed from the extra pass (fund/xg_{y}), with or without garbage-time stints, seasons 2012-25.
python3 gt_features.py -> fund/features_all.parquet (check: must equal data/features.parquet) and fund/features_ng.parquet."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
from features import OFF_FEATS, DEF_FEATS
D = "/home/claude/data"; FD = "/home/claude/fund"; S = range(2012, 2026)
CNT = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "astd_fgm"]

def on_court(season, drop):
    st = pd.read_parquet(f"{D}/stints_{season}.parquet")
    if drop:
        m = st.margin_start.abs(); st = st[~(((st.t_start >= 2160) & (m >= 20)) | ((st.t_start >= 2520) & (m >= 15)))]
    out = []
    for me, op in (("A", "B"), ("B", "A")):
        x = pd.DataFrame({"lineup": st[f"lineup{me}"],
            "poss_off": st[f"fg2a{me}"] + st[f"fg3a{me}"] - st[f"oreb{me}"] + st[f"tov{me}"] + 0.44 * st[f"fta{me}"],
            "poss_def": st[f"fg2a{op}"] + st[f"fg3a{op}"] - st[f"oreb{op}"] + st[f"tov{op}"] + 0.44 * st[f"fta{op}"],
            "opp_rim_a": st[f"za1{op}"], "opp_rim_m": st[f"rim_m{op}"], "opp_mid_a": st[f"za2{op}"] + st[f"za3{op}"],
            "opp_3_a": st[f"za4{op}"] + st[f"za5{op}"] + st[f"za6{op}"]})
        x["player"] = x.pop("lineup").str.split("-"); x = x.explode("player"); x = x[x.player != ""]; x["player"] = x.player.astype(np.int64)
        out.append(x.groupby("player").sum())
    o = pd.concat(out).groupby(level=0).sum(); o["season"] = season
    return o.reset_index()

def build(drop):
    ps = []
    for s in S:
        x = pd.read_parquet(f"{FD}/xg_{s}.parquet")
        if drop: x = x[x.gar == 0]
        ps.append(x.groupby("player")[CNT + ["sec"]].sum().reset_index().assign(season=s))
    ps = pd.concat(ps, ignore_index=True)
    oc = pd.concat([on_court(s, drop) for s in S], ignore_index=True)
    f = ps.merge(oc, on=["season", "player"], how="inner"); f = f[(f.poss_off > 0) & (f.poss_def > 0)].reset_index(drop=True)
    lg = f.groupby("season")[["fg3m", "fg3a", "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "ftm", "fta"]].sum()
    lg["p3"], lg["prim"], lg["pmid"], lg["pft"] = lg.fg3m / lg.fg3a, lg.rim_fgm / lg.rim_fga, lg.mid_fgm / lg.mid_fga, lg.ftm / lg.fta
    L = lg.loc[f.season].reset_index(drop=True)
    sh = lambda m, a, p, k: (m + k * p) / (a + k)
    po, pd_ = f.poss_off / 100, f.poss_def / 100
    f["o_3vol"] = f.fg3a / po; f["o_3eff"] = f.o_3vol * 3 * (sh(f.fg3m, f.fg3a, L.p3, 250) - L.p3)
    f["o_rimvol"] = f.rim_fga / po; f["o_rimeff"] = f.o_rimvol * 2 * (sh(f.rim_fgm, f.rim_fga, L.prim, 100) - L.prim)
    f["o_midvol"] = f.mid_fga / po; f["o_mideff"] = f.o_midvol * 2 * (sh(f.mid_fgm, f.mid_fga, L.pmid, 150) - L.pmid)
    f["o_ftvol"] = f.fta / po; f["o_fteff"] = f.o_ftvol * (sh(f.ftm, f.fta, L.pft, 75) - L.pft)
    f["o_unast"] = (f.fgm - f.astd_fgm) / po; f["o_ast"] = f.ast / po; f["o_tov"] = f.tov / po; f["o_oreb"] = f.oreb / po
    f["d_dreb"] = f.dreb / pd_; f["d_blk"] = f.blk / pd_; f["d_stl"] = f.stl / pd_; f["d_pf"] = f.pf / pd_
    f["d_opprim"] = (f.opp_rim_a / pd_) * 2 * (sh(f.opp_rim_m, f.opp_rim_a, L.prim, 400) - L.prim)
    opp_fga = f.opp_rim_a + f.opp_mid_a + f.opp_3_a
    xq = (f.opp_rim_a * 2 * L.prim + f.opp_mid_a * 2 * L.pmid + f.opp_3_a * 3 * L.p3) / opp_fga
    f["d_oppxq"] = xq * opp_fga / pd_
    for c in OFF_FEATS + DEF_FEATS:
        w = f.poss_off if c.startswith("o_") else f.poss_def
        mean = (f[c] * w).groupby(f.season).transform("sum") / w.groupby(f.season).transform("sum")
        f[c] = ((f[c] - mean) * (w / (w + 300))).fillna(0.0)
    f["min"] = f.sec / 60
    return f[["season", "player", "min", "poss_off", "poss_def"] + OFF_FEATS + DEF_FEATS]

if __name__ == "__main__":
    a = build(False); n = build(True)
    a.to_parquet(f"{FD}/features_all.parquet", index=False); n.to_parquet(f"{FD}/features_ng.parquet", index=False)
    F = pd.read_parquet(f"{D}/features.parquet"); F = F[F.season >= 2012]
    m = F.merge(a, on=["season", "player"], how="outer", suffixes=("", "_x"), indicator=True)
    print("rows prod / rebuilt / matched:", len(F), len(a), (m._merge == "both").sum())
    print("max abs diff vs data/features.parquet by feature:")
    print(pd.Series({c: (m[c] - m[c + "_x"]).abs().max() for c in OFF_FEATS + DEF_FEATS + ["poss_off"]}).round(5).to_string())
    k = a.merge(n, on=["season", "player"], suffixes=("", "_n")); big = k[k.poss_off > 2000]
    print("\nwithout garbage: possessions kept", round(n.poss_off.sum() / a.poss_off.sum(), 4), "| players lost", len(a) - len(n))
    print("corr / sd of change (players 2000+ poss):")
    print(pd.DataFrame({c: [np.corrcoef(big[c], big[c + "_n"])[0, 1], (big[c + "_n"] - big[c]).std(), big[c].std()] for c in OFF_FEATS + DEF_FEATS}, index=["corr", "sd_change", "sd"]).T.round(3).to_string())
