"""Fetch NBA data on GitHub Actions runners (the Claude sandbox cannot reach NBA.com).

Sources (verified from runners 2026-09-19):
  cdn.nba.com liveData/staticData  -> needs curl_cffi impersonate='chrome' (plain requests get Akamai 403)
  S3 mirror of liveData            -> plain requests work (backup for pbp/box)
  ESPN site.api injuries/scoreboard -> Chrome impersonation; site.web.api.espn.com works plainly

Usage:
  python pipeline/nba_fetch.py schedule --season 2026
  python pipeline/nba_fetch.py games --season 2026 [--since 2026-10-20] [--until ...] [--limit N]
  python pipeline/nba_fetch.py injuries
  python pipeline/nba_fetch.py odds [--date 20261021]
Outputs under live/{season}/ (compact, committed to the repo).
"""
import argparse, datetime as dt, gzip, io, json, os, sys, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDN = "https://cdn.nba.com/static/json"
S3 = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Referer": "https://www.nba.com/",
       "Origin": "https://www.nba.com"}

# columns of the cdn action feed used by parse_pbp.norm_cdn / cdn_zone (same names as the shufinskiy mirror CSV)
ACTION_COLS = ["actionNumber", "clock", "timeActual", "period", "actionType", "subType", "personId", "teamId",
               "teamTricode", "playerNameI", "shotResult", "assistPersonId", "foulDrawnPersonId",
               "jumpBallWonPersonId", "jumpBallLostPersonId", "jumpBallRecoverdPersonId", "scoreHome", "scoreAway",
               "orderNumber", "xLegacy", "yLegacy", "shotDistance", "area", "description", "qualifiers",
               "descriptor", "blockPersonId", "stealPersonId", "possession"]


def get_json(url, tries=3):
    from curl_cffi import requests as creq
    import requests
    last = None
    for i in range(tries):
        for how in ("chrome", "plain"):
            try:
                if how == "chrome":
                    r = creq.get(url, headers=HDR, impersonate="chrome", timeout=30)
                else:
                    r = requests.get(url, headers=HDR, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
                last = f"{how} {r.status_code}"
            except Exception as e:  # noqa
                last = f"{how} {type(e).__name__}: {e}"
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def season_label(season):  # 2026 -> "2026-27"
    return f"{season}-{str(season + 1)[-2:]}"


def live_dir(season, *sub):
    d = os.path.join(ROOT, "live", str(season), *sub)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ schedule
def fetch_schedule(season):
    js = get_json(f"{CDN}/staticData/scheduleLeagueV2.json")
    lg = js["leagueSchedule"]
    rows = []
    for day in lg["gameDates"]:
        for g in day["games"]:
            gid = g["gameId"]
            rows.append(dict(
                game=int(gid), gameId=gid, kind={"1": "pre", "2": "reg", "3": "allstar", "4": "post", "5": "playin",
                                                  "6": "cup"}.get(gid[2], gid[2]),
                date_et=(g.get("gameDateEst") or "")[:10], time_utc=g.get("gameDateTimeUTC"),
                home_id=g["homeTeam"]["teamId"], home=g["homeTeam"]["teamTricode"],
                away_id=g["awayTeam"]["teamId"], away=g["awayTeam"]["teamTricode"],
                status=g.get("gameStatus"), home_pts=g["homeTeam"].get("score"), away_pts=g["awayTeam"].get("score"),
                label=g.get("gameLabel", ""), sublabel=g.get("gameSubLabel", ""),
                cup=g.get("gameSubtype", "")))
    df = pd.DataFrame(rows)
    feed_season = lg.get("seasonYear")
    out = live_dir(season)
    df.to_csv(os.path.join(out, "schedule.csv"), index=False)
    reg = df[df.kind == "reg"]
    print(f"schedule feed season {feed_season}: {len(df)} games, regular season {len(reg)}, "
          f"final {int((reg.status == 3).sum())}, dates {reg.date_et.min()} .. {reg.date_et.max()}")
    return df


# ------------------------------------------------------------------ play-by-play + box
def compact_actions(js):
    gid = int(js["game"]["gameId"])
    acts = pd.DataFrame(js["game"]["actions"])
    for c in ACTION_COLS:
        if c not in acts:
            acts[c] = None
    acts = acts[ACTION_COLS].copy()
    acts["qualifiers"] = acts["qualifiers"].map(lambda q: "|".join(q) if isinstance(q, list) else q)
    acts["gameId"] = gid
    return acts


def compact_box(js):
    g = js["game"]
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = g[side]
        for p in t.get("players", []):
            s = p.get("statistics", {})
            rows.append(dict(game=int(g["gameId"]), team=t["teamId"], abbr=t["teamTricode"], home=side == "homeTeam",
                             player=p["personId"], name=p.get("name"), status=p.get("status"),
                             starter=p.get("starter"), played=p.get("played"),
                             not_playing=p.get("notPlayingReason"), not_playing_desc=p.get("notPlayingDescription"),
                             minutes=s.get("minutes"), pts=s.get("points"), pm=s.get("plusMinusPoints")))
    meta = dict(game=int(g["gameId"]), date_et=(g.get("gameEt") or "")[:10], home_id=g["homeTeam"]["teamId"],
                htm=g["homeTeam"]["teamTricode"], away_id=g["awayTeam"]["teamId"], vtm=g["awayTeam"]["teamTricode"],
                home_pts=g["homeTeam"].get("score"), away_pts=g["awayTeam"].get("score"), status=g.get("gameStatus"),
                periods=g.get("period"))
    return pd.DataFrame(rows), meta


def fetch_games(season, since=None, until=None, limit=None, kinds=("reg",), idrange=None):
    if idrange:  # past seasons (schedule feed only covers the current one): regular-season ids 002YY00001..
        a, b = map(int, idrange.split("-"))
        todo = pd.DataFrame({"gameId": [f"002{str(season)[-2:]}{i:05d}" for i in range(a, b + 1)], "date_et": ""})
        since = until = None
    else:
        sch_f = os.path.join(live_dir(season), "schedule.csv")
        sch = pd.read_csv(sch_f, dtype={"gameId": str}) if os.path.exists(sch_f) else fetch_schedule(season)
        todo = sch[sch.kind.isin(kinds) & (sch.status == 3)]
    if since:
        todo = todo[todo.date_et >= since]
    if until:
        todo = todo[todo.date_et <= until]
    pdir, bdir = live_dir(season, "pbp"), live_dir(season, "box")
    todo = todo[[not os.path.exists(os.path.join(pdir, f"{g}.csv.gz")) for g in todo.gameId]]
    if limit:
        todo = todo.head(limit)
    print(f"{len(todo)} games to fetch")
    metas, n_ok = [], 0
    for gid in todo.gameId:
        try:
            pbp = get_json(f"{CDN}/liveData/playbyplay/playbyplay_{gid}.json") or \
                get_json(f"{S3}/playbyplay/playbyplay_{gid}.json")
            box = get_json(f"{CDN}/liveData/boxscore/boxscore_{gid}.json") or \
                get_json(f"{S3}/boxscore/boxscore_{gid}.json")
            if pbp is None or box is None:
                print("missing", gid); continue
            bx, meta = compact_box(box)
            if meta["status"] != 3:
                print("not final", gid); continue
            compact_actions(pbp).to_csv(os.path.join(pdir, f"{gid}.csv.gz"), index=False, compression="gzip")
            bx.to_csv(os.path.join(bdir, f"{gid}.csv.gz"), index=False, compression="gzip")
            metas.append(meta); n_ok += 1
            time.sleep(0.3)
        except Exception as e:  # keep going; the next run retries
            print("error", gid, e)
    if metas:
        mf = os.path.join(live_dir(season), "games_meta.csv")
        old = pd.read_csv(mf) if os.path.exists(mf) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(metas)]).drop_duplicates("game", keep="last").sort_values("game") \
            .to_csv(mf, index=False)
    print(f"fetched {n_ok}")


# ------------------------------------------------------------------ injuries + odds (ESPN)
def fetch_injuries(season):
    js = get_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
    rows = []
    for t in js.get("injuries", []):
        for it in t.get("injuries", []):
            a = it.get("athlete", {})
            rows.append(dict(team=t.get("displayName"), abbr=(a.get("team") or {}).get("abbreviation"),
                             name=a.get("displayName"), status=it.get("status"), date=it.get("date"),
                             type=(it.get("type") or {}).get("description"),
                             detail=(it.get("details") or {}).get("detail"),
                             side=(it.get("details") or {}).get("side"),
                             return_date=(it.get("details") or {}).get("returnDate"),
                             comment=it.get("shortComment")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(live_dir(season), "injuries.csv"), index=False)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d")
    df.to_csv(os.path.join(live_dir(season, "injuries_hist"), f"{stamp}.csv.gz"), index=False, compression="gzip")
    print(f"injuries: {len(df)} rows, statuses {df.status.value_counts().to_dict() if len(df) else {}}")


def fetch_odds(season, date=None):
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    if date:
        url += f"?dates={date}"
    js = get_json(url)
    rows = []
    for ev in js.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
        for o in comp.get("odds", []) or []:
            rows.append(dict(date=ev.get("date"), home=teams.get("home"), away=teams.get("away"),
                             provider=(o.get("provider") or {}).get("name"), details=o.get("details"),
                             spread=o.get("spread"), over_under=o.get("overUnder"),
                             home_ml=(o.get("homeTeamOdds") or {}).get("moneyLine"),
                             away_ml=(o.get("awayTeamOdds") or {}).get("moneyLine")))
    df = pd.DataFrame(rows)
    stamp = date or dt.datetime.utcnow().strftime("%Y%m%d")
    if len(df):
        df.to_csv(os.path.join(live_dir(season, "odds"), f"{stamp}.csv"), index=False)
    print(f"odds: {len(df)} rows for {stamp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["schedule", "games", "injuries", "odds"])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--since"); ap.add_argument("--until"); ap.add_argument("--date")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--idrange", help="e.g. 1101-1230 (regular-season game numbers, past seasons)")
    a = ap.parse_args()
    if a.what == "schedule":
        fetch_schedule(a.season)
    elif a.what == "games":
        fetch_games(a.season, a.since, a.until, a.limit, idrange=a.idrange)
    elif a.what == "injuries":
        fetch_injuries(a.season)
    else:
        fetch_odds(a.season, a.date)
