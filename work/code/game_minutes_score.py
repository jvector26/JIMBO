"""Score game-level minutes rules on data/game_minutes.parquet: player-game minutes RMSE (players who played; team scaled
to 240) and game margin RMSE using preseason ratings x predicted minutes (oracle 'who played', as a perfect injury report)."""
import numpy as np, pandas as pd, sys
D = "/home/claude/data"
G = pd.read_parquet(f"{D}/game_minutes.parquet")
UNR = -4.313 * 1.0501
G["r"] = G.rating.fillna(UNR)
pace = {S: (lambda T: (T.margin / T.net).mean() * 100)(pd.read_parquet(f"{D}/pre_T_{S}.parquet")) for S in range(2014, 2026)}
gm = pd.concat([pd.read_parquet(f"{D}/games_{S}.parquet") for S in range(2014, 2026)])[["game", "home_team", "away_team", "home_pts", "away_pts", "season"]]
def cur(G, k=5, rw=.5):
    w = G.n_tm / (G.n_tm + k); pre = G.pre_tg.fillna(6.0)
    rec = (rw * G.l10_tg + (1 - rw) * G.std_tg).fillna(pre)
    return w * rec + (1 - w) * pre
def new(G, k=5, a=.5, pre_col="pre_wp"):
    w = G.np / (G.np + k); pre = G[pre_col].fillna(10.0)
    rec = (a * G.l5_p + (1 - a) * G.std_p).fillna(pre)
    return w * rec + (1 - w) * pre
def evaluate(m, lab, cap=None, verbose=True):
    x = G[["game", "team", "min", "r", "season", "started", "pre_wp", "np"]].copy(); x["m"] = m.clip(lower=0).values
    x["m"] = x.m * 240 / x.groupby(["game", "team"]).m.transform("sum")
    if cap:   # cap and re-spread the excess to uncapped teammates (few iterations)
        for _ in range(5):
            over = (x.m - cap).clip(lower=0); x["m"] = x.m.clip(upper=cap)
            ex = over.groupby([x.game, x.team]).transform("sum"); room = (x.m < cap)
            x["m"] = x.m + np.where(room, ex * x.m / x.m.where(room, 0).groupby([x.game, x.team]).transform("sum"), 0)
    e = x.m - x["min"]
    s = (x.r * x.m).groupby([x.game, x.team]).sum() / 48; so = (x.r * x["min"]).groupby([x.game, x.team]).sum() / 48
    g = gm.copy()
    hs = s.reindex(pd.MultiIndex.from_arrays([g.game, g.home_team])).values; as_ = s.reindex(pd.MultiIndex.from_arrays([g.game, g.away_team])).values
    g["pred"] = (hs - as_) * g.season.map(pace) / 100 + 2.6; g["marg"] = g.home_pts - g.away_pts
    g = g.dropna(subset=["pred"])
    top = x.pre_wp >= 32
    res = dict(lab=lab, rmse=np.sqrt((e ** 2).mean()), rmse_st=np.sqrt((e[x.started == 1] ** 2).mean()),
               bias_top=e[top].mean(), m_top=x.m[top].mean(), a_top=x["min"][top].mean(),
               bias_st=e[x.started == 1].mean(), game=np.sqrt(((g.pred - g.marg) ** 2).mean()),
               game_h1=np.sqrt(((g.pred - g.marg)[g.season <= 2019] ** 2).mean()), game_h2=np.sqrt(((g.pred - g.marg)[g.season >= 2020] ** 2).mean()))
    if verbose: print(f"{lab:34s} min RMSE {res['rmse']:.2f} (starters {res['rmse_st']:.2f}, bias {res['bias_st']:+.2f}) | pre 32+ mpg: pred {res['m_top']:.1f} act {res['a_top']:.1f} | game RMSE {res['game']:.4f} ({res['game_h1']:.4f}/{res['game_h2']:.4f})")
    return res, g.set_index("game").pred
if __name__ == "__main__":
    evaluate(G["min"], "oracle (actual minutes)")
    evaluate(cur(G), "current nightly rule")
    for k in (2, 5, 10):
        for a in (0, .5, 1):
            evaluate(new(G, k, a), f"played-only k={k} a={a}")
