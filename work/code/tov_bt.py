"""Ball-security checks: does the TOV penalty over-penalize high-usage creators? Window-N player values (bigs_win, weights_bt)
x modifiers -> game diffs in N+1 with actual minutes -> data/tov_bt.parquet (g_* columns, home minus away, pts/game)."""
import json, numpy as np, pandas as pd, impact as I
Wb = json.load(open(f"{I.D}/weights_bt.json")); Wo = {k: .5 * v for k, v in Wb["Wo"].items()}
gm = pd.read_parquet(f"{I.D}/games_all.parquet")
out = []
for N in range(2014, 2025):
    o = pd.read_parquet(f"{I.D}/bigs_win_{N}.parquet").set_index("player")
    rows = I.load_rows([N - 2, N - 1, N]); pace = (rows.groupby("game").poss.sum() / 2).mean()
    r = {k: o[f"f:o_{k}"] / Wo[f"o_{k}"] for k in ["3vol", "rimvol", "midvol", "ftvol", "tov", "ast", "unast"]}
    usg = r["3vol"] + r["rimvol"] + r["midvol"] + .44 * r["ftvol"] + r["tov"]  # centered usage per 100 poss
    reg = o.dposs > 1500
    q85 = usg[reg].quantile(.85); q70 = usg[reg].quantile(.70)
    hi = (usg > q85).astype(float); mid = ((usg > q70) & (usg <= q85)).astype(float)
    uz = ((usg - usg[reg].mean()) / usg[reg].std()).clip(-3, 3)
    az = ((r["ast"] - r["ast"][reg].mean()) / r["ast"][reg].std()).clip(-3, 3)
    tov = o["f:o_tov"]
    V = {"tov": tov, "tov_hi": tov * hi, "tov_mid": tov * mid, "tov_uz": tov * uz, "tov_az": tov * az,
         "hi": hi, "crea": o["f:o_ast"] + o["f:o_unast"] + o["f:o_ftvol"] + o["f:o_fteff"],
         "crea_hi": (o["f:o_ast"] + o["f:o_unast"] + o["f:o_ftvol"] + o["f:o_fteff"]) * hi,
         "res_o_hi": o.res_o * hi, "ast_az": o["f:o_ast"] * az, "unast_az": o["f:o_unast"] * az, "az": az, "ast_uz": o["f:o_ast"] * uz, "boxo_az": o[[c for c in o.columns if c.startswith("f:o_")]].sum(axis=1) * az}
    # TOV relative to usage: residual of tov rate on usage among regulars (clumsy vs busy)
    b = np.polyfit(usg[reg], r["tov"][reg], 1)
    tov_x = r["tov"] - b[0] * usg; tov_x = tov_x - tov_x[reg].mean()
    V["tovx"] = Wo["o_tov"] * tov_x          # 'excess' turnovers at the same weight
    V["tovu"] = Wo["o_tov"] * (b[0] * usg)   # usage-explained part
    V = pd.DataFrame(V).fillna(0)
    pg = pd.read_parquet(f"{I.D}/player_games_{N+1}.parquet")[["game", "player", "team", "min"]]
    g = gm[gm.season == N + 1][["game", "home_team", "away_team"]]
    t = V.reindex(pg.player).fillna(0).mul(pg["min"].values / 48, axis=0)
    t.index = pd.MultiIndex.from_arrays([pg.game, pg.team]); t = t.groupby(level=[0, 1]).sum()
    hk = pd.MultiIndex.from_arrays([g.game, g.home_team]); ak = pd.MultiIndex.from_arrays([g.game, g.away_team])
    dd = pd.DataFrame((t.reindex(hk).to_numpy() - t.reindex(ak).to_numpy()) * pace / 100, columns=["g_" + c for c in V.columns])
    dd["game"] = g.game.values; out.append(dd); print(N + 1, "tov-usage slope", round(b[0], 3), flush=True)
pd.concat(out).to_parquet(f"{I.D}/tov_bt.parquet", index=False)
