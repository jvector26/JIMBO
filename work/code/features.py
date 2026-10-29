"""Player-season feature table for the component prior.
All rates are per 100 on-court possessions, centered on the season's possession-weighted league mean
(so an average player is ~0 in every era), and shrunk toward 0 for small samples."""
import numpy as np, pandas as pd

D = "/home/claude/data"
S = range(1996, 2026)

# component -> features (sign of each weight is learned from lineup outcomes)
COMPONENTS = {
    "three_pt":       ["o_3eff", "o_3vol"],
    "rim_finishing":  ["o_rimeff", "o_rimvol"],
    "pullup_mid":     ["o_mideff", "o_midvol"],
    "foul_drawing":   ["o_fteff", "o_ftvol"],
    "self_creation":  ["o_unast"],
    "playmaking":     ["o_ast"],
    "ball_security":  ["o_tov"],
    "off_rebounding": ["o_oreb"],
    "def_rebounding": ["d_dreb"],
    "rim_protection": ["d_blk", "d_opprim"],
    "perimeter_def":  ["d_stl", "d_pf", "d_oppxq"],
}
OFF_FEATS = [f for c, fs in COMPONENTS.items() for f in fs if f.startswith("o_")]
DEF_FEATS = [f for c, fs in COMPONENTS.items() for f in fs if f.startswith("d_")]


def on_court(season):
    """Per player-season: offensive/defensive possessions on court and opponent shot profile while on defense."""
    st = pd.read_parquet(f"{D}/stints_{season}.parquet")
    out = []
    for me, op in (("A", "B"), ("B", "A")):
        x = pd.DataFrame({
            "lineup": st[f"lineup{me}"],
            "poss_off": st[f"fg2a{me}"] + st[f"fg3a{me}"] - st[f"oreb{me}"] + st[f"tov{me}"] + 0.44 * st[f"fta{me}"],
            "poss_def": st[f"fg2a{op}"] + st[f"fg3a{op}"] - st[f"oreb{op}"] + st[f"tov{op}"] + 0.44 * st[f"fta{op}"],
            "opp_rim_a": st[f"za1{op}"], "opp_rim_m": st[f"rim_m{op}"],
            "opp_mid_a": st[f"za2{op}"] + st[f"za3{op}"],
            "opp_3_a": st[f"za4{op}"] + st[f"za5{op}"] + st[f"za6{op}"],
        })
        x["player"] = x.pop("lineup").str.split("-")
        x = x.explode("player")
        x = x[x.player != ""]
        x["player"] = x.player.astype(np.int64)
        out.append(x.groupby("player").sum())
    o = pd.concat(out).groupby(level=0).sum()
    o["season"] = season
    return o.reset_index()


def build():
    ps = pd.read_parquet(f"{D}/player_seasons.parquet")
    oc = pd.concat([on_court(s) for s in S], ignore_index=True)
    f = ps.merge(oc, on=["season", "player"], how="inner")
    f = f[(f.poss_off > 0) & (f.poss_def > 0)].copy()

    # league rates by season
    lg = f.groupby("season")[["fg3m", "fg3a", "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "ftm", "fta"]].sum()
    lg["p3"], lg["prim"], lg["pmid"], lg["pft"] = lg.fg3m / lg.fg3a, lg.rim_fgm / lg.rim_fga, lg.mid_fgm / lg.mid_fga, lg.ftm / lg.fta
    L = lg.loc[f.season].reset_index(drop=True)
    f = f.reset_index(drop=True)

    def shrunk(m, a, p, k):
        return (m + k * p) / (a + k)

    po, pd_ = f.poss_off / 100, f.poss_def / 100
    f["o_3vol"] = f.fg3a / po
    f["o_3eff"] = f.o_3vol * 3 * (shrunk(f.fg3m, f.fg3a, L.p3, 250) - L.p3)
    f["o_rimvol"] = f.rim_fga / po
    f["o_rimeff"] = f.o_rimvol * 2 * (shrunk(f.rim_fgm, f.rim_fga, L.prim, 100) - L.prim)
    f["o_midvol"] = f.mid_fga / po
    f["o_mideff"] = f.o_midvol * 2 * (shrunk(f.mid_fgm, f.mid_fga, L.pmid, 150) - L.pmid)
    f["o_ftvol"] = f.fta / po
    f["o_fteff"] = f.o_ftvol * (shrunk(f.ftm, f.fta, L.pft, 75) - L.pft)
    f["o_unast"] = (f.fgm - f.astd_fgm) / po
    f["o_ast"] = f.ast / po
    f["o_tov"] = f.tov / po
    f["o_oreb"] = f.oreb / po
    f["d_dreb"] = f.dreb / pd_
    f["d_blk"] = f.blk / pd_
    f["d_stl"] = f.stl / pd_
    f["d_pf"] = f.pf / pd_
    # opponent rim finishing while on court (points per 100 def poss above league), heavily shrunk
    f["d_opprim"] = (f.opp_rim_a / pd_) * 2 * (shrunk(f.opp_rim_m, f.opp_rim_a, L.prim, 400) - L.prim)
    # opponent shot quality allowed: location-expected points per FGA vs league, scaled per 100 possessions
    opp_fga = f.opp_rim_a + f.opp_mid_a + f.opp_3_a
    xq = (f.opp_rim_a * 2 * L.prim + f.opp_mid_a * 2 * L.pmid + f.opp_3_a * 3 * L.p3) / opp_fga
    f["d_oppxq"] = (xq * opp_fga / pd_)  # expected pts per 100 from shot profile alone

    # center per season (possession-weighted) and shrink small samples toward 0
    for c in OFF_FEATS + DEF_FEATS:
        w = f.poss_off if c.startswith("o_") else f.poss_def
        mean = (f[c] * w).groupby(f.season).transform("sum") / w.groupby(f.season).transform("sum")
        f[c] = ((f[c] - mean) * (w / (w + 300))).fillna(0.0)
    keep = ["season", "player", "name", "teams", "age", "min", "poss_off", "poss_def"] + OFF_FEATS + DEF_FEATS
    f[keep].to_parquet(f"{D}/features.parquet", index=False)
    return f[keep]


if __name__ == "__main__":
    f = build()
    print(f.shape)
    print(f[OFF_FEATS + DEF_FEATS].describe().T[["mean", "std", "min", "max"]].round(2))
