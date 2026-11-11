"""Extra play-by-play pass for the 2026-11-11 fundamentals tests (T6 transition, T8 garbage time).
Same lineup engine as parse_pbp.process_game, but the box is keyed by (player, garbage flag) and transition tallies are added.
Garbage flag = the stint rule of impact.load_rows (stint starts in Q4 with |margin| >= 20, or in the last 6 min with >= 15).
Transition play = FGA / turnover / FT trip by a team within W seconds of gaining the ball on a player defensive rebound or a steal.
Output /home/claude/fund/xg_{season}.parquet: one row per game x player x gar with the box counts, seconds and
  tr_fga tr_pts tr_plays (own) | on_tro_pts on_tro_plays on_trd_pts on_trd_plays on_och on_dch (team tallies while on court).
Usage: python3 parse_x.py 2012 2013 ...   (needs raw/{nbastats|cdnnba}_{y}.tar.xz + shotdetail_{y}.tar.xz, see fund_rebuild.sh)"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code")
import parse_pbp as P
OUT = "/home/claude/fund"; W = 7.0
STAT = P.STAT_KEYS
XK = ["tr_fga", "tr_pts", "tr_plays", "on_tro_pts", "on_tro_plays", "on_trd_pts", "on_trd_plays", "on_och", "on_dch"]


def process(g, rows):
    p2t, teams = {}, []
    for r in rows:
        tm, pl, k = r[3], r[4], r[2]
        if tm and tm > 1e9 and tm not in teams: teams.append(tm)
        if pl and tm and pl < 1e9 and k != "SEEN": p2t.setdefault(pl, tm)
    if len(teams) != 2: return None
    A, B = teams; oth = {A: B, B: A}
    box = {}; gar = [False]

    def bx(pl, tm):
        key = (pl, gar[0])
        if key not in box:
            box[key] = dict.fromkeys(STAT + XK, 0); box[key].update(team=tm, sec=0.0)
        return box[key]

    periods = sorted({r[0] for r in rows}); by_period = {p: [r for r in rows if r[0] == p] for p in periods}
    prev_end = {A: [], B: []}; last_miss_team = None; score = {A: 0, B: 0}
    for p in periods:
        prow = by_period[p]; first = {}
        for r in prow:
            pl = r[4]
            if pl and pl < 1e9 and pl not in first: first[pl] = r[2]
        on = {A: [], B: []}
        for pl, k in first.items():
            tm = p2t.get(pl)
            if tm in on and k != "SUBIN": on[tm].append(pl)
        for tm in (A, B):
            if len(on[tm]) < 5:
                for pl in prev_end[tm]:
                    if pl not in on[tm] and first.get(pl) is None and len(on[tm]) < 5: on[tm].append(pl)
            on[tm] = on[tm][:5]
        tcur = P.period_start(p)

        def setgar(t0):
            m = abs(score[A] - score[B]); gar[0] = (t0 >= 2160 and m >= 20) or (t0 >= 2520 and m >= 15)
        setgar(tcur)
        pos = [None, "dead", -99.0]; trft = [None, -1.0]

        def live(tm, t): return pos[0] == tm and pos[1] in ("dreb", "stl") and t - pos[2] <= W

        def tally(tm, pts, plays):
            for q in on[tm]:
                b = bx(q, tm); b["on_tro_pts"] += pts; b["on_tro_plays"] += plays
            for q in on[oth[tm]]:
                b = bx(q, oth[tm]); b["on_trd_pts"] += pts; b["on_trd_plays"] += plays

        def chance(tm):
            for q in on[tm]: bx(q, tm)["on_och"] += 1
            for q in on[oth[tm]]: bx(q, oth[tm])["on_dch"] += 1

        for (_, t, k, tm, pl, val, zone, astd) in prow:
            t = max(t, tcur)
            if t > tcur:
                dt = t - tcur
                for side in (A, B):
                    for q in on[side]: bx(q, side)["sec"] += dt
                tcur = t
            if k == "SEEN": continue
            if tm not in (A, B):
                tm = p2t.get(pl, tm)
                if tm not in (A, B): continue
            b = bx(pl, tm) if pl and pl < 1e9 else None
            if k in ("FGM", "FGX"):
                made = k == "FGM"; three = val == 3; z = int(zone) if zone == zone else 0
                tr = live(tm, t)
                if tr:
                    tally(tm, val if made else 0, 1); trft[0], trft[1] = tm, t
                    if b:
                        b["tr_fga"] += 1; b["tr_plays"] += 1
                        if made: b["tr_pts"] += val
                if made:
                    score[tm] += val; pos[0], pos[1], pos[2] = oth[tm], "dead", t
                else:
                    last_miss_team = tm
                if b:
                    b["fga"] += 1; b["fg3a"] += three
                    b["rim_fga"] += z == 1; b["mid_fga"] += z in (2, 3); b["c3_fga"] += z == 4
                    if made:
                        b["fgm"] += 1; b["pts"] += val; b["fg3m"] += three
                        b["rim_fgm"] += z == 1; b["mid_fgm"] += z in (2, 3); b["astd_fgm"] += astd
            elif k in ("FTM", "FTX"):
                tr = trft[0] == tm and trft[1] == t
                if b: b["fta"] += 1
                if tr:
                    tally(tm, 1 if k == "FTM" else 0, 0.44)
                    if b: b["tr_plays"] += 0.44
                if k == "FTM":
                    score[tm] += 1
                    if b:
                        b["ftm"] += 1; b["pts"] += 1
                        if tr: b["tr_pts"] += 1
                    pos[0], pos[1], pos[2] = oth[tm], "dead", t
                else:
                    last_miss_team = tm
            elif k == "REB":
                off = val if val >= 0 else int(tm == last_miss_team)
                if b: b["oreb" if off else "dreb"] += 1
                if off: pos[0], pos[1], pos[2] = tm, "oreb", t
                elif b is not None:
                    pos[0], pos[1], pos[2] = tm, "dreb", t; chance(tm)
                else: pos[0], pos[1], pos[2] = tm, "dead", t
            elif k == "TOV":
                if b: b["tov"] += 1
                if live(tm, t):
                    tally(tm, 0, 1)
                    if b: b["tr_plays"] += 1
                pos[0], pos[1], pos[2] = oth[tm], "dead", t
            elif k == "STL":
                if b: b["stl"] += 1
                pos[0], pos[1], pos[2] = tm, "stl", t; chance(tm)
            elif k == "PF":
                if b: b["pf"] += 1
                if live(oth[tm], t):  # defensive foul on a transition play: its FTs count, later shots do not
                    trft[0], trft[1] = oth[tm], t; pos[1] = "dead"
            elif k in ("AST", "BLK"):
                if b: b[k.lower()] += 1
            elif k == "SUBOUT" and pl in on[tm]:
                on[tm] = [q for q in on[tm] if q != pl]; setgar(tcur)
            elif k == "SUBIN" and pl not in on[tm]:
                on[tm] = on[tm] + [pl]; bx(pl, tm)
        tend = P.period_start(p) + P.period_len(p)
        if tend > tcur:
            for side in (A, B):
                for q in on[side]: bx(q, side)["sec"] += tend - tcur
        prev_end = {A: list(on[A]), B: list(on[B])}
    return box


def run(season):
    ev, names, abbr, dates, finals = P.norm_cdn(season) if season >= 2025 else P.norm_v2(season)
    ev["team"] = ev["team"].fillna(0).astype(np.int64); ev["player"] = ev["player"].fillna(0).astype(np.int64)
    sdz = pd.read_csv(P.read_csv_from_tar(f"shotdetail_{season}"), usecols=["GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC"])
    sdz = sdz.drop_duplicates(["GAME_ID", "GAME_EVENT_ID"]).rename(columns={"GAME_ID": "game", "GAME_EVENT_ID": "evn"})
    sdz["zone"] = sdz["SHOT_ZONE_BASIC"].map(P.ZONE)
    ev = ev.merge(sdz[["game", "evn", "zone"]], on=["game", "evn"], how="left")
    ev.loc[~ev["kind"].isin(["FGM", "FGX"]), "zone"] = 0; ev["zone"] = ev["zone"].fillna(0)
    cols = ["period", "t", "kind", "team", "player", "val", "zone", "astd"]; out = []
    for g, gdf in ev.groupby("game", sort=False):
        box = process(g, list(gdf[cols].itertuples(index=False, name=None)))
        if box is None: continue
        for (pl, gr), d in box.items(): out.append(dict(game=g, player=pl, gar=int(gr), **d))
    x = pd.DataFrame(out); x["season"] = season
    os.makedirs(OUT, exist_ok=True); x.to_parquet(f"{OUT}/xg_{season}.parquet", index=False)
    # checks: totals reproduce player_games; league transition share
    pg = pd.read_parquet(f"/home/claude/data/player_games_{season}.parquet")
    chk = ["pts", "fga", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "rim_fga", "astd_fgm"]
    d = (x.groupby(["game", "player"])[chk].sum() - pg.set_index(["game", "player"])[chk]).abs().sum().sum()
    ds = abs(x.sec.sum() - pg.sec.sum())
    poss = x.fga.sum() - x.oreb.sum() + x.tov.sum() + .44 * x.fta.sum()
    trp = x.on_tro_plays.sum() / 5; trpts = x.on_tro_pts.sum() / 5
    print(season, "rows", len(x), "| box diff vs player_games", int(d), "sec diff", round(ds, 1), "| garbage share of sec", round(x.sec[x.gar == 1].sum() / x.sec.sum(), 4),
          "| transition plays/poss", round(trp / poss, 3), "pts/play", round(trpts / trp, 3), "own plays share", round(x.tr_plays.sum() / trp, 3), flush=True)
    for f in os.listdir(P.RAW):
        if f.endswith(f"_{season}.csv"): os.remove(f"{P.RAW}/{f}")


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]): run(s)
