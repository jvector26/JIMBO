"""Package JIMBO's preseason model into state/{S}/ for the nightly pipeline. Runs in Claude's sandbox (needs the full
model data in /home/claude/data), not on GitHub Actions.
  python pipeline/build_state.py 2026          production state (players_proj_2026 + FA pool + dashboard overrides)
  python pipeline/build_state.py 2025 --replay replay state from the 2025-26 preseason backtest (pipeline testing)
Files: players.parquet, hist_box.parquet, teams_pre.csv, config.json"""
import json, os, sys, numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = "/home/claude/data"
sys.path.insert(0, os.path.join(ROOT, "model"))
from jimbo_update import OFF_FEATS, DEF_FEATS, box_of
S = int(sys.argv[1]); REPLAY = "--replay" in sys.argv
N = S - 1
out = os.path.join(ROOT, "state", str(S) + ("_replay" if REPLAY else "")); os.makedirs(out, exist_ok=True)
cal = pd.read_csv(f"{D}/calib.csv", index_col=0).iloc[:, 0]
Wp = json.load(open(f"{D}/weights_prod.json")); B = json.load(open(f"{D}/blend.json"))
SIG = 12.8
cfg = {"season": S, "slope": float(cal["slope"]), "unrated": float(cal["unrated"]),
       "Wo": {k: 0.5 * v for k, v in Wp["Wo"].items()}, "Wd": {k: 0.5 * v for k, v in Wp["Wd"].items()},
       "lam": 6000.0, "dS": 1.0, "H0": 2000.0, "lam_t": 80.0, "min_games": 15, "hca": 2.6, "sig": SIG,
       "pace": float(json.load(open(f"{D}/pace_2025.json"))["pace"]),
       "tau_pre": B["tau_blend"], "tau_mid": 2.0, "tau_late": 1.8,
       "gap_k": 20.0, "mpg_k": 10.0, "avail_k": 20.0, "out_default_games": 10,
       "notes": "M4 update (inseason_bt.py). tau_* and gap_k / mpg_k / avail_k are PROVISIONAL (in-season minutes and "
                "uncertainty backtest not yet run)."}
# history window (box part of ratings), for the box update
F = pd.read_parquet(f"{D}/features.parquet")
fw = F[F.season.between(N - 2, N)].copy(); dec = fw.season.map({N - 2: .3, N - 1: .6, N: 1.0})
fw["wo"], fw["wd"] = fw.poss_off * dec, fw.poss_def * dec
H = pd.concat([fw[OFF_FEATS].mul(fw.wo, axis=0).groupby(fw.player).sum(), fw[DEF_FEATS].mul(fw.wd, axis=0).groupby(fw.player).sum(),
               fw.groupby("player")[["wo", "wd"]].sum()], axis=1)
H.to_parquet(os.path.join(out, "hist_box.parquet"))
hb = pd.Series(box_of(pd.concat([H[OFF_FEATS].div(H.wo, axis=0), H[DEF_FEATS].div(H.wd, axis=0)], axis=1).fillna(0),
                      pd.Series(cfg["Wo"]), pd.Series(cfg["Wd"])), index=H.index)
nm = F.sort_values("season").drop_duplicates("player", keep="last").set_index("player").name
if not REPLAY:
    P = pd.concat([pd.read_csv(f"{D}/players_proj_2026.csv"), pd.read_csv(f"{D}/players_fa_2026.csv")], ignore_index=True)
    pos = pd.read_csv(f"{D}/pos_2026.csv")
    P = P.merge(pos[["name", "team", "pos_est"]], on=["name", "team"], how="left")
    P["key"] = [f"p{int(p)}" if pd.notna(p) else f"n:{n}|{t}" for p, n, t in zip(P.player, P["name"], P.team)]
    P["avail_pre"] = (P.exp_games / 82).fillna(70 / 82).clip(0, 1)
    P["mpg_pre"] = P.min_model / (82 * P.avail_pre)   # so avail x mpg x 82 = project.py's min_model (pre-allocation)
    P = P.rename(columns={"rating": "rating_pre", "off": "off_pre", "def": "def_pre"})
    P["team"] = P.team.fillna("FA")
    keep = ["key", "player", "name", "team", "role", "rating_pre", "off_pre", "def_pre", "mpg_pre", "avail_pre", "min_proj",
            "age", "pos_est", "hustle", "note"]
    P = P[keep]
    T = pd.read_csv(f"{D}/teams_proj_2026.csv", index_col=0)
    L = pd.read_csv(f"{D}/win_totals_2026.csv").set_index("team")
    imp = lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100)
    po = L.over.map(imp) / (L.over.map(imp) + L.under.map(imp)); L["mkt_mean"] = L.line + 7 * norm.ppf(po)
    mw = 82 * norm.cdf(T.margin / SIG)
    age = (P.age.fillna(21) * P.min_proj).groupby(P.team).sum() / P.groupby("team").min_proj.sum()
    age_c = (age - age.reindex(T.index).mean()).reindex(T.index)
else:
    pp = pd.read_parquet(f"{D}/pre_P_{S}.parquet").drop_duplicates("player")
    P = pd.DataFrame({"key": "p" + pp.player.astype(int).astype(str), "player": pp.player, "name": pp.player.map(nm),
                      "team": pp.team, "role": "", "rating_pre": pp.rating, "off_pre": pp.rating / 2, "def_pre": pp.rating / 2,
                      "mpg_pre": pp.min_proj / 70, "avail_pre": 70 / 82, "min_proj": pp.min_proj, "age": np.nan,
                      "pos_est": 3.0, "hustle": 0.0, "note": ""})
    T = pd.read_parquet(f"{D}/pre_T_{S}.parquet").set_index("team")
    L = pd.read_csv(f"{D}/hist_lines.csv"); L = L[L.season == S].set_index("team"); L["mkt_mean"] = L.line
    mw = 82 * norm.cdf(T.margin / SIG)
    dfe = pd.read_parquet(f"{D}/direction_features.parquet"); dfe = dfe[dfe.season == S].set_index("team")
    age_c = (dfe.mw_age - dfe.mw_age.mean()).reindex(T.index)
# rated players not in the table (possible mid-season signings): box-only history rating, as the backtest
extra = hb.index.difference(P.player.dropna())
E = pd.DataFrame({"key": "p" + pd.Series(extra).astype(int).astype(str).values, "player": extra, "name": nm.reindex(extra).values,
                  "team": "FA", "role": "", "rating_pre": cfg["slope"] * hb.reindex(extra).values, "mpg_pre": 12.0,
                  "avail_pre": 0.6, "min_proj": 0.0, "pos_est": 3.0, "hustle": 0.0, "note": "not on an opening roster"})
E["off_pre"] = E.rating_pre / 2; E["def_pre"] = E.rating_pre / 2
P = pd.concat([P, E], ignore_index=True)
P.to_parquet(os.path.join(out, "players.parquet"), index=False)
bw = 41 + B["w_model"] * (mw - 41) + B["w_vegas"] * (L.mkt_mean.reindex(T.index) - 41) + B["w_age"] * age_c.fillna(0)
bw = bw - (bw.mean() - 41)
TP = pd.DataFrame({"model_margin_pre": T.margin, "blend_margin_pre": SIG * norm.ppf(bw / 82), "line": L.line.reindex(T.index),
                   "mkt_mean": L.mkt_mean.reindex(T.index), "model_w_pre": mw, "blend_w_pre": bw})
TP.index.name = "team"; TP.to_csv(os.path.join(out, "teams_pre.csv"))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
print(f"state -> {out}: {len(P)} players ({(P.team != 'FA').sum()} rostered), {len(H)} with history")
