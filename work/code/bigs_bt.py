"""Bigs investigation step 2: game-level diffs in season N+1 (actual minutes) of window-N player values x modifiers.
-> data/bigs_bt.parquet (one row per game, columns g_<name>, home minus away, pts per game scale like bt_split)."""
import json, numpy as np, pandas as pd, impact as I, shooting as SH
W3 = json.load(open(f"{I.D}/weights_bt.json"))["Wo"]["o_3eff"] * .5
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
OFFB = ["f:o_3vol", "f:o_3eff", "f:o_rimvol", "f:o_rimeff", "f:o_oreb", "f:o_ftvol", "f:o_ast", "f:o_unast", "f:o_tov"]
DEFB = ["f:d_dreb", "f:d_blk", "f:d_opprim", "f:d_stl", "f:d_pf", "f:d_oppxq"]
out = []
for N in range(2014, 2025):
    o = pd.read_parquet(f"{I.D}/bigs_win_{N}.parquet").set_index("player")
    rows = I.load_rows([N - 2, N - 1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    o["f:o_3eff"] = SH.MULT * W3 * o.index.to_series().map(SH.talent_feature(N)).fillna(0)
    p = o.pos.fillna(3.0)
    B = (p >= 3.9).astype(float); G = (p < 2.5).astype(float)
    reg = o.dposs > 1500
    ctx = (o.ctx - o.ctx[reg].mean()).fillna(0).clip(-.5, .5)
    thin = o.thin
    age1 = o.age_N + 1
    young = (age1 <= 23).astype(float).where(o.age_N.notna(), 0)
    boxo = o[OFFB + ["f:o_mideff", "f:o_midvol", "f:o_fteff"]].sum(axis=1); boxd = o[DEFB].sum(axis=1)
    V = {"res_o": o.res_o, "res_d": o.res_d,
         "res_o_B": o.res_o * B, "res_d_B": o.res_d * B, "res_o_G": o.res_o * G, "res_d_G": o.res_d * G,
         "res_o_thin": o.res_o * thin, "res_d_thin": o.res_d * thin,
         "res_o_ctx": o.res_o * ctx, "res_d_ctx": o.res_d * ctx, "ctx": ctx, "ctx_B": ctx * B,
         "boxo_B": boxo * B, "boxd_B": boxd * B, "B": B, "young": young, "young_B": young * B,
         "res_o_Bthin": o.res_o * B * thin, "res_d_Bthin": o.res_d * B * thin}
    for c in OFFB + DEFB:
        V[c.replace("f:", "") + "_B"] = o[c] * B
    tot = boxo + boxd + .6 * o.res_o + .8 * o.res_d
    hi = (tot > 2).astype(float)
    for k in (0, 1, 2, 3, 4):
        V[f"kink{k}"] = (tot - k).clip(lower=0)
    V.update({"reso_hi": o.res_o * hi, "resd_hi": o.res_d * hi, "boxo_hi": boxo * hi, "boxd_hi": boxd * hi,
              "thin_hi": (tot - 2).clip(lower=0) * thin, "kink2_B": (tot - 2).clip(lower=0) * B})
    V["tot_all"] = tot
    V.update({"tot_B": tot * B, "totHi": tot * hi, "totHi_B": tot * hi * B, "hi_B": hi * B, "hi": hi})
    V = pd.DataFrame(V)
    pg = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = V.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame((t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100, columns=["g_" + c for c in V.columns])
    T = t["tot_all"] * pace / 100
    th, ta = T.reindex(hk).to_numpy(), T.reindex(ak).to_numpy()
    dd["g_team_sq"] = np.sign(th) * th ** 2 - np.sign(ta) * ta ** 2
    for k in (0, 2, 4): dd[f"g_team_kink{k}"] = np.clip(th - k, 0, None) - np.clip(ta - k, 0, None)
    dd["g_team_sq2"] = th ** 2 - ta ** 2
    dd["game"] = g.game.values; out.append(dd); print("done", N + 1, flush=True)
pd.concat(out).to_parquet(f"{I.D}/bigs_bt.parquet", index=False)
