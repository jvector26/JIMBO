"""
Parse one season of NBA play-by-play into:
  player_games_{season}.parquet : player-game box score + minutes + plus-minus
  stints_{season}.parquet       : lineup stints (10 players on floor) with points/possession inputs (RAPM-ready)
  games_{season}.parquet        : game metadata (date, home/away, final score, lineup-quality flags)

Sources (github.com/shufinskiy/nba_data mirror of NBA.com):
  seasons 1996-2024 -> nbastats (stats.nba.com playbyplayv2 format, PLAYER1/2/3 ids)
  season  2025      -> cdnnba   (cdn.nba.com live format, explicit sub in/out, off/def rebound tags)
Game dates and home/away come from the shotdetail files.
"""
import re, sys, tarfile, os
import numpy as np, pandas as pd

RAW = os.environ.get("JIMBO_RAW", "/home/claude/raw")
OUT = os.environ.get("JIMBO_DATA", "/home/claude/data")

TECH_RE = re.compile(r"T\.FOUL|T\.Foul|Technical|Taunt|Non-Unsport|Def\. 3 Sec|DOUBLE TECH|Delay Tech", re.I)


def period_start(p):
    return (p - 1) * 720 if p <= 4 else 2880 + (p - 5) * 300


def period_len(p):
    return 720 if p <= 4 else 300


def read_csv_from_tar(name):
    csv = f"{RAW}/{name}.csv"
    if not os.path.exists(csv):
        with tarfile.open(f"{RAW}/{name}.tar.xz") as t:
            t.extract(f"{name}.csv", RAW)
    return csv


# ---------------------------------------------------------------- normalizers
# unified event columns: game, ord, period, t (elapsed game sec), kind, team, player, val
# kinds: FGM(val=pts) FGX(val=2|3 missed) FTM FTX AST STL BLK REB(val: 1=off,0=def,-1=unknown) TOV PF SUBOUT SUBIN SEEN
def norm_v2(season):
    df = pd.read_csv(read_csv_from_tar(f"nbastats_{season}"), low_memory=False)
    df = df.drop_duplicates(["GAME_ID", "EVENTNUM"]).reset_index(drop=True)
    mmss = df["PCTIMESTRING"].astype(str).str.split(":", expand=True)
    rem = mmss[0].astype(float) * 60 + mmss[1].astype(float)
    p = df["PERIOD"].astype(int)
    df["t"] = np.where(p <= 4, (p - 1) * 720 + 720 - rem, 2880 + (p - 5) * 300 + 300 - rem)
    # EVENTNUM is sometimes out of chronological order; sort by game clock, EVENTNUM as tiebreak
    df = df.sort_values(["GAME_ID", "PERIOD", "t", "EVENTNUM"], kind="stable").reset_index(drop=True)
    df["ord"] = np.arange(len(df))
    desc = (df["HOMEDESCRIPTION"].fillna("") + " | " + df["NEUTRALDESCRIPTION"].fillna("") + " | " +
            df["VISITORDESCRIPTION"].fillna(""))
    et, at = df["EVENTMSGTYPE"], df["EVENTMSGACTIONTYPE"]
    p1, p2, p3 = df["PLAYER1_ID"].fillna(0).astype(np.int64), df["PLAYER2_ID"].fillna(0).astype(np.int64), df["PLAYER3_ID"].fillna(0).astype(np.int64)
    t1, t2, t3 = df["PLAYER1_TEAM_ID"].fillna(0).astype(np.int64), df["PLAYER2_TEAM_ID"].fillna(0).astype(np.int64), df["PLAYER3_TEAM_ID"].fillna(0).astype(np.int64)
    pt1 = df["PERSON1TYPE"].fillna(0)
    is_player1 = pt1.isin([4, 5]) & (p1 > 0)
    is3 = desc.str.contains("3PT")
    df["evn"] = df["EVENTNUM"]
    df["astd"] = ((df["EVENTMSGTYPE"] == 1) & (p2 > 0) & df["PERSON2TYPE"].isin([4, 5])).astype(int)
    base = df[["GAME_ID", "ord", "PERIOD", "t", "evn", "astd"]].rename(columns={"GAME_ID": "game", "PERIOD": "period"})
    parts = []

    def add(mask, kind, team, player, val=0):
        b = base[mask].copy()
        b["kind"], b["team"], b["player"] = kind, team[mask].values, player[mask].values
        b["val"] = val[mask].values if isinstance(val, pd.Series) else val
        parts.append(b)

    add(et == 1, "FGM", t1, p1, pd.Series(np.where(is3, 3, 2), index=df.index))
    add((et == 1) & (p2 > 0) & df["PERSON2TYPE"].isin([4, 5]), "AST", t2, p2)
    add(et == 2, "FGX", t1, p1, pd.Series(np.where(is3, 3, 2), index=df.index))
    add((et == 2) & (p3 > 0) & desc.str.contains("BLOCK"), "BLK", t3, p3)
    ftmiss = df["SCORE"].isna()  # a made FT always updates SCORE; text-based MISS detection misses some rows
    add((et == 3) & ~ftmiss, "FTM", t1, p1, 1)
    add((et == 3) & ftmiss, "FTX", t1, p1)
    # rebounds: player rebound (team = player's team) or team rebound (PLAYER1_ID is team id)
    team_reb_team = pd.Series(np.where(is_player1, t1, p1), index=df.index)
    add((et == 4) & is_player1, "REB", t1, p1, -1)
    add((et == 4) & ~is_player1 & (p1 > 0), "REB", team_reb_team, pd.Series(0, index=df.index), -1)
    not_noTO = ~desc.str.contains("No Turnover", case=False)
    add((et == 5) & is_player1 & not_noTO, "TOV", t1, p1)
    add((et == 5) & ~is_player1 & (p1 > 0) & not_noTO, "TOV", team_reb_team, pd.Series(0, index=df.index))
    add((et == 5) & (p2 > 0) & desc.str.contains("STEAL") & df["PERSON2TYPE"].isin([4, 5]), "STL", t2, p2)
    tech = desc.str.contains(TECH_RE)
    add((et == 6) & is_player1 & ~tech, "PF", t1, p1)
    add((et == 6) & (p2 > 0) & df["PERSON2TYPE"].isin([4, 5]) & ~tech, "SEEN", t2, p2)  # fouled player
    add((et == 8), "SUBOUT", t1, p1)
    add((et == 8), "SUBIN", t1, p2)
    add((et == 7) & is_player1, "SEEN", t1, p1)
    for pid, tid, ptype in [(p1, t1, "PERSON1TYPE"), (p2, t2, "PERSON2TYPE"), (p3, t3, "PERSON3TYPE")]:
        add((et == 10) & (pid > 0) & df[ptype].isin([4, 5]), "SEEN", tid, pid)
    ev = pd.concat(parts, ignore_index=True)
    ev["val"] = ev["val"].astype(int)
    order = {"SUBOUT": 8, "SUBIN": 9}
    ev["sub_ord"] = ev["kind"].map(order).fillna(0)
    ev = ev.sort_values(["game", "ord", "sub_ord"]).drop(columns="sub_ord")

    names = pd.concat([df[["PLAYER1_ID", "PLAYER1_NAME"]].set_axis(["pid", "name"], axis=1),
                       df[["PLAYER2_ID", "PLAYER2_NAME"]].set_axis(["pid", "name"], axis=1)]).dropna()
    abbr = df[["GAME_ID", "PLAYER1_TEAM_ID", "PLAYER1_TEAM_ABBREVIATION"]].dropna().drop_duplicates()
    abbr.columns = ["game", "team", "abbr"]
    sc = df.dropna(subset=["SCORE"])
    sp = sc["SCORE"].str.split(" - ", expand=True).astype(int)
    f = sp.groupby(sc["GAME_ID"]).max()  # scores are monotonic, so max = final even if rows are out of order
    finals = pd.DataFrame({"game": f.index, "off_home": f[1].values, "off_away": f[0].values})
    return ev, names, abbr, None, finals


def norm_cdn(season, df=None):
    if df is None:
        df = pd.read_csv(read_csv_from_tar(f"cdnnba_{season}"), low_memory=False)
    df = df.drop_duplicates(["gameId", "actionNumber"]).sort_values(["gameId", "orderNumber"]).reset_index(drop=True)
    df["ord"] = np.arange(len(df))
    m = df["clock"].astype(str).str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
    rem = m[0] * 60 + m[1]
    p = df["period"].astype(int)
    df["t"] = np.where(p <= 4, (p - 1) * 720 + 720 - rem, 2880 + (p - 5) * 300 + 300 - rem)
    at, st = df["actionType"], df["subType"]
    pid = df["personId"].fillna(0).astype(np.int64)
    tid = df["teamId"].fillna(0).astype(np.int64)
    df["evn"] = df["actionNumber"]
    df["astd"] = (df["assistPersonId"].fillna(0) > 0).astype(int)
    base = df[["gameId", "ord", "period", "t", "evn", "astd"]].rename(columns={"gameId": "game"})
    parts = []

    def add(mask, kind, team, player, val=0):
        b = base[mask].copy()
        b["kind"], b["team"], b["player"] = kind, team[mask].values, player[mask].values
        b["val"] = val[mask].values if isinstance(val, pd.Series) else val
        parts.append(b)

    shot = at.isin(["2pt", "3pt"])
    made = df["shotResult"] == "Made"
    shotval = pd.Series(np.where(at == "3pt", 3, 2), index=df.index)
    add(shot & made, "FGM", tid, pid, shotval)
    add(shot & ~made, "FGX", tid, pid, shotval)
    ast = df["assistPersonId"].fillna(0).astype(np.int64)
    add(shot & made & (ast > 0), "AST", tid, ast)
    add((at == "freethrow") & made, "FTM", tid, pid, 1)
    add((at == "freethrow") & ~made, "FTX", tid, pid)
    add(at == "rebound", "REB", tid, pid, pd.Series(np.where(st == "offensive", 1, 0), index=df.index))
    add(at == "turnover", "TOV", tid, pid)
    add(at == "steal", "STL", tid, pid)
    add(at == "block", "BLK", tid, pid)
    add((at == "foul") & st.isin(["personal", "offensive"]) & (pid > 0), "PF", tid, pid)
    add((at == "substitution") & (st == "out"), "SUBOUT", tid, pid)
    add((at == "substitution") & (st == "in"), "SUBIN", tid, pid)
    add((at == "violation") & (pid > 0), "SEEN", tid, pid)
    fd = df["foulDrawnPersonId"].fillna(0).astype(np.int64)
    add((at == "foul") & (fd > 0) & (st != "technical"), "SEEN", pd.Series(0, index=df.index), fd)
    for c in ["jumpBallWonPersonId", "jumpBallLostPersonId", "jumpBallRecoverdPersonId"]:
        j = df[c].fillna(0).astype(np.int64)
        add((at == "jumpball") & (j > 0), "SEEN", pd.Series(0, index=df.index), j)
    ev = pd.concat(parts, ignore_index=True)
    ev["val"] = ev["val"].astype(int)
    ev["sub_ord"] = ev["kind"].map({"SUBOUT": 8, "SUBIN": 9}).fillna(0)
    ev = ev.sort_values(["game", "ord", "sub_ord"]).drop(columns="sub_ord")
    names = df[["personId", "playerNameI"]].dropna().set_axis(["pid", "name"], axis=1)
    abbr = df[["gameId", "teamId", "teamTricode"]].dropna().drop_duplicates()
    abbr.columns = ["game", "team", "abbr"]
    abbr["team"] = abbr["team"].astype(np.int64)
    dates = df.groupby("gameId")["timeActual"].min()
    f = df.dropna(subset=["scoreHome"]).groupby("gameId")[["scoreHome", "scoreAway"]].last().astype(int)
    finals = pd.DataFrame({"game": f.index, "off_home": f["scoreHome"].values, "off_away": f["scoreAway"].values})
    return ev, names, abbr, dates, finals


# ---------------------------------------------------------------- per-game lineup engine
STAT_KEYS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
             "rim_fgm", "rim_fga", "mid_fgm", "mid_fga", "c3_fga", "astd_fgm"]
# per-side stint stats. zones: 1 restricted area, 2 paint non-RA, 3 midrange, 4 corner 3, 5 above-break 3, 6 backcourt
SIDE_KEYS = ["pts", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "x3", "xft", "rim_m",
             "za1", "za2", "za3", "za4", "za5", "za6", "za0"]
NS = len(SIDE_KEYS)
IX = {k: i for i, k in enumerate(SIDE_KEYS)}


def process_game(g, rows, xpct, lg3, lgft):
    """rows: tuples (period, t, kind, team, player, val, zone, astd) in event order."""
    p2t, teams = {}, []
    for r in rows:
        tm, pl, k = r[3], r[4], r[2]
        if tm and tm > 1e9 and tm not in teams:
            teams.append(tm)
        if pl and tm and pl < 1e9 and k != "SEEN":
            p2t.setdefault(pl, tm)
    if len(teams) != 2:
        return None
    A, B = teams

    box = {}
    def bx(pl, tm):
        if pl not in box:
            box[pl] = dict.fromkeys(STAT_KEYS, 0); box[pl].update(team=tm, sec=0.0, pm=0, started=0)
        return box[pl]

    stints, flags = [], 0
    periods = sorted({r[0] for r in rows})
    by_period = {p: [r for r in rows if r[0] == p] for p in periods}
    prev_end = {A: [], B: []}
    last_miss_team = None
    score = {A: 0, B: 0}
    for p in periods:
        prow = by_period[p]
        first = {}
        for r in prow:
            pl = r[4]
            if pl and pl < 1e9 and pl not in first:
                first[pl] = r[2]
        on = {A: [], B: []}
        for pl, k in first.items():
            tm = p2t.get(pl)
            if tm in on and k != "SUBIN":
                on[tm].append(pl)
        for tm in (A, B):
            if len(on[tm]) < 5:
                for pl in prev_end[tm]:
                    if pl not in on[tm] and first.get(pl) is None and len(on[tm]) < 5:
                        on[tm].append(pl)
            on[tm] = on[tm][:5]
            if len(on[tm]) != 5:
                flags += 1
        if p == 1:
            for tm in (A, B):
                for pl in on[tm]:
                    bx(pl, tm)["started"] = 1
        tcur = period_start(p)
        st = {"start": tcur, "m0": score[A] - score[B], A: [0.0] * NS, B: [0.0] * NS}

        def close(tend):
            if tend > st["start"] or any(st[A]) or any(st[B]):
                stints.append((g, p, st["start"], tend, st["m0"], A, tuple(sorted(on[A])), B, tuple(sorted(on[B])),
                               *st[A], *st[B]))

        for (_, t, k, tm, pl, val, zone, astd) in prow:
            t = max(t, tcur)
            if t > tcur:
                dt = t - tcur
                for side in (A, B):
                    for q in on[side]:
                        bx(q, side)["sec"] += dt
                tcur = t
            if k == "SEEN":
                continue
            if tm not in (A, B):
                tm = p2t.get(pl, tm)
                if tm not in (A, B):
                    continue
            S = st[tm]
            b = bx(pl, tm) if pl and pl < 1e9 else None
            if k in ("FGM", "FGX"):
                made = k == "FGM"
                three = val == 3
                z = int(zone) if zone == zone else 0
                S[IX["za%d" % z]] += 1
                if three:
                    S[IX["fg3a"]] += 1; S[IX["x3"]] += xpct.get(pl, (lg3, lgft))[0]
                    if made: S[IX["fg3m"]] += 1
                else:
                    S[IX["fg2a"]] += 1
                    if made: S[IX["fg2m"]] += 1
                if z == 1 and made: S[IX["rim_m"]] += 1
                if made:
                    S[IX["pts"]] += val
                    score[tm] += val
                    for side in (A, B):
                        for q in on[side]:
                            bx(q, side)["pm"] += val if side == tm else -val
                else:
                    last_miss_team = tm
                if b:
                    b["fga"] += 1; b["fg3a"] += three
                    b["rim_fga"] += z == 1; b["mid_fga"] += z in (2, 3); b["c3_fga"] += z == 4
                    if made:
                        b["fgm"] += 1; b["pts"] += val; b["fg3m"] += three
                        b["rim_fgm"] += z == 1; b["mid_fgm"] += z in (2, 3); b["astd_fgm"] += astd
            elif k in ("FTM", "FTX"):
                S[IX["fta"]] += 1; S[IX["xft"]] += xpct.get(pl, (lg3, lgft))[1]
                if b: b["fta"] += 1
                if k == "FTM":
                    S[IX["ftm"]] += 1; S[IX["pts"]] += 1; score[tm] += 1
                    if b: b["ftm"] += 1; b["pts"] += 1
                    for side in (A, B):
                        for q in on[side]:
                            bx(q, side)["pm"] += 1 if side == tm else -1
                else:
                    last_miss_team = tm
            elif k == "REB":
                off = val if val >= 0 else int(tm == last_miss_team)
                S[IX["oreb" if off else "dreb"]] += 1
                if b: b["oreb" if off else "dreb"] += 1
            elif k == "TOV":
                S[IX["tov"]] += 1
                if b: b["tov"] += 1
            elif k in ("AST", "STL", "BLK", "PF"):
                if b: b[k.lower()] += 1
            elif k == "SUBOUT" and pl in on[tm]:
                close(tcur); on[tm] = [q for q in on[tm] if q != pl]
                st["start"] = tcur; st["m0"] = score[A] - score[B]; st[A] = [0.0] * NS; st[B] = [0.0] * NS
            elif k == "SUBIN" and pl not in on[tm]:
                on[tm] = on[tm] + [pl]; bx(pl, tm)
        tend = period_start(p) + period_len(p)
        if tend > tcur:
            for side in (A, B):
                for q in on[side]:
                    bx(q, side)["sec"] += tend - tcur
        close(tend)
        prev_end = {A: list(on[A]), B: list(on[B])}
    return box, stints, flags, (A, B)


ZONE = {"Restricted Area": 1, "In The Paint (Non-RA)": 2, "Mid-Range": 3, "Left Corner 3": 4, "Right Corner 3": 4,
        "Above the Break 3": 5, "Backcourt": 6}


def shooter_pcts(season, out=None):
    """Shrunk season 3P% and FT% per shooter, used to replace made/missed 3s and FTs with expected values in stints."""
    f = f"{out or OUT}/player_games_{season}.parquet"
    if os.path.exists(f):
        t = pd.read_parquet(f).groupby("player")[["fg3m", "fg3a", "ftm", "fta"]].sum()
    elif not os.path.exists(f"{OUT}/player_seasons.parquet"):  # live pipeline, first pass: league-average shooters
        return {}, 0.36, 0.78
    else:  # fresh rebuild: same season totals from player_seasons (no second parse pass needed)
        ps = pd.read_parquet(f"{OUT}/player_seasons.parquet")
        t = ps[ps.season == season].groupby("player")[["fg3m", "fg3a", "ftm", "fta"]].sum()
    lg3, lgft = t.fg3m.sum() / t.fg3a.sum(), t.ftm.sum() / t.fta.sum()
    x3 = (t.fg3m + 250 * lg3) / (t.fg3a + 250)
    xft = (t.ftm + 75 * lgft) / (t.fta + 75)
    return {p: (a, b) for p, a, b in zip(t.index, x3, xft)}, lg3, lgft


def run(season, cdn_df=None, meta=None, xy_zones=False, out=None, quiet=False):
    """cdn_df: cdn action rows (live JSON flattened, see pbp_live.py) instead of the mirror CSV.
    meta: DataFrame game,date,htm,vtm replacing shotdetail (live pipeline). xy_zones: zones from coordinates."""
    out = out or OUT
    if cdn_df is not None or season >= 2025:
        ev, names, abbr, dates, finals = norm_cdn(season, cdn_df)
        if xy_zones:
            import cdn_zone
            src = cdn_df if cdn_df is not None else pd.read_csv(read_csv_from_tar(f"cdnnba_{season}"), low_memory=False)
            sh = src[src["actionType"].isin(["2pt", "3pt"])].drop_duplicates(["gameId", "actionNumber"])
            xyz = pd.DataFrame({"game": sh["gameId"].values, "evn": sh["actionNumber"].values,
                                "zone_xy": cdn_zone.zone_from_xy(sh["xLegacy"], sh["yLegacy"], sh["actionType"] == "3pt")})
    else:
        ev, names, abbr, dates, finals = norm_v2(season)
    ev["team"] = ev["team"].fillna(0).astype(np.int64)
    ev["player"] = ev["player"].fillna(0).astype(np.int64)
    if season <= 2001:  # original Charlotte Hornets appear under both franchise ids in the feed; unify
        ev["team"] = ev["team"].replace(1610612766, 1610612740)
        abbr["team"] = abbr["team"].replace(1610612766, 1610612740)
    if xy_zones:
        ev = ev.merge(xyz.rename(columns={"zone_xy": "zone"}), on=["game", "evn"], how="left")
    else:
        sdz = pd.read_csv(read_csv_from_tar(f"shotdetail_{season}"), usecols=["GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC"])
        sdz = sdz.drop_duplicates(["GAME_ID", "GAME_EVENT_ID"]).rename(columns={"GAME_ID": "game", "GAME_EVENT_ID": "evn"})
        sdz["zone"] = sdz["SHOT_ZONE_BASIC"].map(ZONE)
        ev = ev.merge(sdz[["game", "evn", "zone"]], on=["game", "evn"], how="left")
    ev.loc[~ev["kind"].isin(["FGM", "FGX"]), "zone"] = 0
    ev["zone"] = ev["zone"].fillna(0)
    xpct, lg3, lgft = shooter_pcts(season, out)
    pg_rows, stint_rows, game_rows = [], [], []
    cols = ["period", "t", "kind", "team", "player", "val", "zone", "astd"]
    for g, gdf in ev.groupby("game", sort=False):
        res = process_game(g, list(gdf[cols].itertuples(index=False, name=None)), xpct, lg3, lgft)
        if res is None:
            continue
        box, stints, flags, (A, B) = res
        for pl, d in box.items():
            pg_rows.append(dict(game=g, player=pl, **d))
        stint_rows.extend(stints)
        game_rows.append(dict(game=g, teamA=A, teamB=B, lineup_flags=flags))

    pg = pd.DataFrame(pg_rows)
    pg["min"] = pg["sec"] / 60
    st = pd.DataFrame(stint_rows, columns=["game", "period", "t_start", "t_end", "margin_start", "teamA", "lineupA", "teamB", "lineupB"]
                      + [k + "A" for k in SIDE_KEYS] + [k + "B" for k in SIDE_KEYS])
    st[[k + s_ for s_ in "AB" for k in SIDE_KEYS]] = st[[k + s_ for s_ in "AB" for k in SIDE_KEYS]].astype("float32")
    st["lineupA"] = st["lineupA"].map(lambda x: "-".join(map(str, x)))
    st["lineupB"] = st["lineupB"].map(lambda x: "-".join(map(str, x)))
    gm = pd.DataFrame(game_rows)

    # dates + home/away from shotdetail
    if meta is not None:
        sd = meta[["game", "date", "htm", "vtm"]].drop_duplicates("game")
    else:
        sd = pd.read_csv(read_csv_from_tar(f"shotdetail_{season}"), usecols=["GAME_ID", "GAME_DATE", "HTM", "VTM"]).drop_duplicates("GAME_ID")
        sd.columns = ["game", "date", "htm", "vtm"]
    gm = gm.merge(sd, on="game", how="left")
    ab = abbr.drop_duplicates(["game", "team"])
    gm = gm.merge(ab.rename(columns={"team": "teamA", "abbr": "abbrA"}), on=["game", "teamA"], how="left")
    gm = gm.merge(ab.rename(columns={"team": "teamB", "abbr": "abbrB"}), on=["game", "teamB"], how="left")
    gm["home_team"] = np.where(gm["abbrA"] == gm["htm"], gm["teamA"], np.where(gm["abbrB"] == gm["htm"], gm["teamB"], 0))
    gm["away_team"] = np.where(gm["home_team"] == gm["teamA"], gm["teamB"], np.where(gm["home_team"] == gm["teamB"], gm["teamA"], 0))
    if dates is not None:
        d2 = pd.to_datetime(dates, utc=True, format="ISO8601").dt.tz_convert("America/New_York").dt.strftime("%Y%m%d").astype(int)
        gm["date"] = gm["date"].fillna(gm["game"].map(d2))
    pts = pg.groupby(["game", "team"])["pts"].sum().rename("pts").reset_index()
    gm = gm.merge(pts.rename(columns={"team": "home_team", "pts": "home_pts"}), on=["game", "home_team"], how="left")
    gm = gm.merge(pts.rename(columns={"team": "away_team", "pts": "away_pts"}), on=["game", "away_team"], how="left")
    gm["season"] = season
    gm = gm.merge(finals, on="game", how="left")
    gm["score_ok"] = (gm.home_pts == gm.off_home) & (gm.away_pts == gm.off_away)
    nper = st.groupby("game")["period"].max()
    tmin = pg.groupby(["game", "team"])["min"].sum().groupby("game").min()
    gm["min_gap"] = gm["game"].map(nper).map(lambda p: 240 + 25 * max(0, p - 4)) - gm["game"].map(tmin)
    gm = gm[["season", "game", "date", "home_team", "away_team", "htm", "vtm", "home_pts", "away_pts",
             "off_home", "off_away", "score_ok", "lineup_flags", "min_gap"]]

    pg["season"] = season
    pg = pg.merge(gm[["game", "date"]], on="game", how="left")
    st["season"] = season
    names["pid"] = names["pid"].astype(np.int64)
    nm = names.drop_duplicates("pid", keep="last")
    nm["season"] = season

    os.makedirs(out, exist_ok=True)
    pg.to_parquet(f"{out}/player_games_{season}.parquet", index=False)
    st.to_parquet(f"{out}/stints_{season}.parquet", index=False)
    gm.to_parquet(f"{out}/games_{season}.parquet", index=False)
    nm.to_parquet(f"{out}/names_{season}.parquet", index=False)
    if cdn_df is None and os.path.isdir(RAW):
        for f in os.listdir(RAW):
            if f.endswith(f"_{season}.csv"):
                os.remove(f"{RAW}/{f}")
    print(season, "games", len(gm), "player-games", len(pg), "stints", len(st),
          "| score mismatches", int((~gm.score_ok).sum()), "| games w/ minute gap>1", int((gm.min_gap > 1).sum()),
          "| periods w/ bad starters", int(gm.lineup_flags.sum()), flush=True)


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
