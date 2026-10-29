"""3PT efficiency talent: career-to-date 3P% (shrunk k=400 to league) + FT% prior (stronger for low-volume shooters),
times the decayed window 3PA rate. Replaces the window-average o_3eff feature. Validated in shooting_bt.py
(game LOSO 13.5026 -> 13.4939 at 1.5x, better in both halves). Used by project.py and preseason_backtest.py."""
import numpy as np, pandas as pd
D = "/home/claude/data"
MULT = 1.5
COEF = dict(c=0.003, car=0.868, ft=0.011, ft_lo=0.183)   # shooting_prior.py direct regression (next-season 3P% rel. league)
DEC = {0: 1.0, 1: 0.6, 2: 0.3}
_cache = {}
def _base():
    if "p" not in _cache:
        ps = pd.read_parquet(f"{D}/player_seasons.parquet"); f = pd.read_parquet(f"{D}/features.parquet")
        p = ps.groupby(["player", "season"])[["fg3m", "fg3a", "ftm", "fta"]].sum().reset_index().sort_values(["player", "season"])
        for c in ["fg3m", "fg3a", "ftm", "fta"]: p["c_" + c] = p.groupby("player")[c].cumsum()
        p = p.merge(f[["season", "player", "poss_off"]], on=["season", "player"], how="left")
        lg = p.groupby("season")[["fg3m", "fg3a"]].sum(); _cache["lg3"] = lg.fg3m / lg.fg3a; _cache["p"] = p
    return _cache["p"], _cache["lg3"]

def talent_feature(N):
    """Per player: new o_3eff in feature units (3PA/100 poss * 3 * (talent - league) * sample shrink)."""
    p, lg3 = _base()
    w = p[p.season.between(N-2, N)].copy(); w["dw"] = (N - w.season).map(DEC) * w.poss_off.fillna(0)
    v = (w.fg3a / (w.poss_off/100).clip(lower=0.01) * w.dw).groupby(w.player).sum() / w.groupby("player").dw.sum().clip(lower=1)
    poss = w.groupby("player").poss_off.sum()
    last = p[p.season <= N].groupby("player").last()
    car = (last.c_fg3m + 400*lg3[N]) / (last.c_fg3a + 400) - lg3[N]
    ft = (last.c_ftm + 0.75*50) / (last.c_fta + 50) - 0.75
    rel = COEF["c"] + COEF["car"]*car + COEF["ft"]*ft + COEF["ft_lo"]*ft*400/(last.c_fg3a + 400)
    return (v * 3 * rel.reindex(v.index) * poss/(poss + 300)).fillna(0)

def apply(r, N, W3, index_col=None):
    """r: ratings frame (fit_window output) with f:o_3eff, three_pt, off, total. W3 = weight used for o_3eff in r."""
    r = r.copy(); ids = r[index_col] if index_col else r.index.to_series()
    new = MULT * W3 * ids.map(talent_feature(N)).fillna(0).to_numpy()
    delta = new - r["f:o_3eff"].to_numpy()
    r["f:o_3eff"] = new
    for c in ["three_pt", "off", "total"]:
        if c in r: r[c] = r[c] + delta
    return r
