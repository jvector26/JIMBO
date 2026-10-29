"""Contract-year test.
Free-agent classes 2012-2023 (sumitrodatta/contract-prediction-2023: Spotrac/capology) give who hit free agency each summer
and the length of the deal they signed. A FA summer Y is SCHEDULED if the player's previous signing (class Y-n, n years)
ends exactly at Y, or it's an RFA (rookie scale ending). Scheduled expiries are known before the season (no opt-out or
waiver selection), so season Y-1 (start-year label) is a clean 'contract year'. Unscheduled FAs (opt-outs, waivers,
declined options) are reported separately because they are chosen AFTER seeing the season.
Outcome: season box rating (production weights) and on/off, vs expectation from prior 3 seasons + age."""
import json, re, unicodedata
import numpy as np, pandas as pd
D = "/home/claude/data"
norm = lambda s: re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
                        .replace(" jr", "").replace(" iii", "").replace(" ii", "").replace(" sr", ""))

F = pd.read_parquet(f"{D}/features.parquet")
W = json.load(open(f"{D}/weights_prod.json")); Wall = {**W["Wo"], **W["Wd"]}
F["box"] = sum(F[f] * w for f, w in Wall.items())
ps = pd.read_parquet(f"{D}/player_seasons.parquet")[["season", "player", "on_off_diff", "gp"]]
F = F.merge(ps, on=["season", "player"], how="left")

# ---- free-agent classes -> player ids
fa = pd.read_csv(f"{D}/contracts/past_free_agents.csv", encoding="utf-8-sig")
f23 = pd.read_csv(f"{D}/contracts/free_agents_2023.csv")
f23 = f23.assign(contract_yrs=np.nan)[["season", "player", "type", "contract_yrs"]]
fa = pd.concat([fa[["season", "player", "type", "contract_yrs"]], f23], ignore_index=True)
ids = F[F.season >= 2005].groupby("player").name.last().reset_index(); ids["k"] = ids.name.map(norm)
name2id = ids.drop_duplicates("k", keep=False).set_index("k").player
fa["pid"] = fa.player.map(norm).map(name2id)
print(f"FA rows matched: {fa.pid.notna().mean():.3f}; unmatched e.g. {fa[fa.pid.isna()].player.head(6).tolist()}")
fa = fa.dropna(subset=["pid"]); fa["pid"] = fa.pid.astype(int)

# scheduled? previous signing ends exactly this summer
fa = fa.sort_values(["pid", "season"])
fa["prev_season"] = fa.groupby("pid").season.shift(); fa["prev_yrs"] = fa.groupby("pid").contract_yrs.shift()
fa["kind"] = "unknown"
fa.loc[fa.type == "RFA", "kind"] = "scheduled"
ok = fa.prev_season.notna() & (fa.prev_yrs > 0)
fa.loc[ok & (fa.prev_season + fa.prev_yrs == fa.season), "kind"] = "scheduled"
fa.loc[ok & (fa.prev_season + fa.prev_yrs > fa.season), "kind"] = "early"          # opted out / waived / option declined
fa.loc[(fa.season == 2023) & fa.type.isin(["UFA", "RFA"]), "kind"] = "scheduled"   # 2023 list is pre-decision
fa.loc[(fa.season == 2023) & fa.type.isin(["Player", "Club"]), "kind"] = "option"
print(fa.groupby("season").kind.value_counts().unstack(fill_value=0))

fa["cy_season"] = fa.season - 1                                                     # summer 2019 FA -> 2018-19 season
cy = fa.drop_duplicates(["pid", "cy_season"]).set_index(["pid", "cy_season"]).kind
signed = fa[fa.contract_yrs > 0].set_index(["pid", "season"]).contract_yrs          # first season of a new deal

# ---- expectation model
F = F.sort_values(["player", "season"])
g = F.groupby("player")
for k in (1, 2, 3):
    for c in ("box", "min", "on_off_diff"):
        F[f"{c}_l{k}"] = g[c].shift(k).where(g.season.shift(k) == F.season - k)
F["kind"] = [cy.get((p, s), "none") for p, s in zip(F.player, F.season)]
F["post"] = [(p, s) in signed.index for p, s in zip(F.player, F.season)]
A = F[(F.season.between(2011, 2022)) & (F.min_l1 >= 500) & (F["min"] >= 250)].copy()
A = A[A.kind.isin(["none", "scheduled", "early", "option"])]
for k in (1, 2, 3):
    A[f"w{k}"] = A[f"min_l{k}"].fillna(0)
    for c in ("box", "on_off_diff"): A[f"{c}_l{k}"] = A[f"{c}_l{k}"].fillna(0)
def design(c):
    X = pd.DataFrame({"const": 1.0}, index=A.index)
    for k in (1, 2, 3):
        X[f"{c}{k}"] = A[f"{c}_l{k}"] * A[f"w{k}"] / 1000; X[f"w{k}"] = A[f"w{k}"] / 1000
    for lo, hi in [(0, 22), (22, 24), (24, 26), (26, 28), (28, 30), (30, 32), (32, 34)]:
        X[f"age{lo}"] = ((A.age >= lo) & (A.age < hi)).astype(float)
    X["logmin1"] = np.log(A.min_l1)
    for yr in range(2012, 2023): X[f"s{yr}"] = (A.season == yr).astype(float)
    X["cy_sched"] = (A.kind == "scheduled").astype(float)
    X["cy_early"] = (A.kind == "early").astype(float)
    X["cy_option"] = (A.kind == "option").astype(float)
    X["post_deal"] = A.post.astype(float)
    return X
def fit(y, X, w, cl):
    Xw = X.values * np.sqrt(w)[:, None]; yw = y.values * np.sqrt(w)
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None); e = yw - Xw @ b
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    meat = sum(np.outer(s, s) for s in (pd.DataFrame(Xw * e[:, None]).groupby(cl.values).sum().values))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=X.columns)
print(f"\nsample: {len(A)} player-seasons 2011-12..2022-23; groups:", A.kind.value_counts().to_dict(), "post-deal", int(A.post.sum()))
for y, lab in [("box", "BOX RATING (pts/100)"), ("on_off_diff", "ON/OFF (pts/100, noisy)")]:
    X = design("box" if y == "box" else "on_off_diff")
    r = fit(A[y], X, A["min"].values, A.player)
    print(f"\n== {lab}, minutes-weighted, player-clustered SE")
    print(r.loc[["cy_sched", "cy_early", "cy_option", "post_deal"]].round(3).to_string())
# minutes: do contract-year players play more than expected?
X = design("box"); r = fit(np.log(A["min"]), X, np.ones(len(A)), A.player)
print("\n== log MINUTES (unweighted)\n", r.loc[["cy_sched", "cy_early", "cy_option", "post_deal"]].round(3).to_string())
A.to_parquet(f"{D}/contract_year_sample.parquet")

# ---- splits: rookie-scale RFAs vs veteran UFAs; and by salary tier proxy (prior minutes)
typ = fa.drop_duplicates(["pid", "cy_season"]).set_index(["pid", "cy_season"]).type
A["rfa"] = [typ.get((p, s)) == "RFA" for p, s in zip(A.player, A.season)]
def design2(c):
    X = design(c); X["cy_sched_rfa"] = X.cy_sched * A.rfa; X["cy_sched"] = X.cy_sched * ~A.rfa
    X["cy_sched_vet_big"] = X.cy_sched * (A.min_l1 >= 1800); return X
print("\nscheduled CY split: RFA n=%d, UFA n=%d (of which 1800+ min prior n=%d)" % (
    ((A.kind == "scheduled") & A.rfa).sum(), ((A.kind == "scheduled") & ~A.rfa).sum(), ((A.kind == "scheduled") & ~A.rfa & (A.min_l1 >= 1800)).sum()))
for y, c, w, lab in [("box", "box", A["min"].values, "box"), ("on_off_diff", "on_off_diff", A["min"].values, "on/off"),
                     (np.log(A["min"]), "box", np.ones(len(A)), "log minutes")]:
    yy = A[y] if isinstance(y, str) else y
    r = fit(yy, design2(c), w, A.player)
    print(f"{lab:12s}", "  ".join(f"{k} {r.coef[k]:+.3f} (t {r.t[k]:+.1f})" for k in ["cy_sched", "cy_sched_vet_big", "cy_sched_rfa", "post_deal"]))
