"""Team-halo test (Ajay Mitchell question, 2026-10-06). For each player's window ending N, halo = minutes-weighted mean
lineup residual of his season-N teammates (main team, excluding himself). If great teams inflate their players' residuals,
the halo part should NOT travel with movers: subtracting lam*halo from movers' residuals should improve game LOSO."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
from lineup_port_bt import G, B, y, seas, diffs, loso, fixed, gm, R
D = "/home/claude/data"
rows = []
for N in range(2014, 2025):
    r = R[R.season == N].set_index("player")
    m = gm[gm.season == N].groupby(["player", "team"])["min"].sum().reset_index()
    m = m[m.player.isin(r.index)]
    m["ro"] = m.player.map(r.res_o); m["rd"] = m.player.map(r.res_d)
    t = m.groupby("team").agg(W=("min", "sum"), So=("ro", lambda s: (s * m.loc[s.index, "min"]).sum()),
                              Sd=("rd", lambda s: (s * m.loc[s.index, "min"]).sum()))
    main = m.sort_values("min").groupby("player").last()
    tt = t.loc[main.team]
    W = tt.W.values - main["min"].values
    ho = (tt.So.values - main.ro.values * main["min"].values) / W
    hd = (tt.Sd.values - main.rd.values * main["min"].values) / W
    rows.append(pd.DataFrame({"N": N, "player": main.index, "halo_o": ho, "halo_d": hd, "minN": main["min"].values}))
H = pd.concat(rows)
G2 = G.merge(H, on=["N", "player"], how="left").fillna({"halo_o": 0, "halo_d": 0})
assert len(G2) == len(G)
G[["halo_o", "halo_d"]] = G2[["halo_o", "halo_d"]].values
print("halo sd (player-season):", H.halo_o.std().round(2), H.halo_d.std().round(2), "| corr res vs halo:",
      np.corrcoef(G.res_o, G.halo_o)[0, 1].round(2), np.corrcoef(G.res_d, G.halo_d)[0, 1].round(2))
mv = G.moved.values == 1
print("mover share of player-game minutes", np.average(mv, weights=G["min"]).round(3))
def hdiff(mask):
    v = G.sgn * G.w48 * mask
    return (pd.DataFrame({"o": v * G.halo_o, "d": v * G.halo_d, "game": G.game}).groupby("game")[["o", "d"]].sum()
            .reindex(B.index).fillna(0))
base = diffs(np.ones(len(G))); b0, s0 = fixed(base)
print(f"baseline LOSO {b0:.4f}")
hm, hs = hdiff(mv.astype(float)), hdiff((~mv).astype(float))
lin = 0.6 * base.o + 0.8 * base.d
for lab, msk in [("all", np.ones(len(y), bool)), ("2015-19", seas <= 2019), ("2020-25", seas >= 2020)]:
    X = np.c_[np.ones(msk.sum()), B.box.values[msk] + lin.values[msk], (0.6*hm.o + 0.8*hm.d).values[msk], (0.6*hs.o + 0.8*hs.d).values[msk]]
    b = np.linalg.lstsq(X, y[msk], rcond=None)[0]; r = y[msk] - X @ b; ss = seas[msk]; XtXi = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[ss == s].T @ r[ss == s], X[ss == s].T @ r[ss == s]) for s in np.unique(ss)); se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    print(f"{lab:8s} model {b[1]:.3f} | movers' halo {b[2]/b[1]:+.2f} (t {b[2]/se[2]:+.1f}) | stayers' halo {b[3]/b[1]:+.2f} (t {b[3]/se[3]:+.1f})"
          "   [coef relative to model; -1 = halo fully spurious]")
print("\nsubtract lam*halo from residual (movers only / everyone):")
for lam in [.25, .5, .75, 1.0]:
    for who, msk in [("movers", mv), ("all", np.ones(len(G), bool))]:
        hh = hdiff(msk.astype(float)); dd = base - lam * hh
        l, s = fixed(dd)
        h1 = np.sqrt(np.mean(s[:5]**2)) - np.sqrt(np.mean(s0[:5]**2)); h2 = np.sqrt(np.mean(s[5:]**2)) - np.sqrt(np.mean(s0[5:]**2))
        print(f"  lam {lam:.2f} {who:6s}: LOSO {l:.4f} (d {l-b0:+.4f}) seasons better {int((s<s0).sum())}/11 halves {h1:+.4f} {h2:+.4f}")
H.to_parquet(f"{D}/halo_players.parquet", index=False)
