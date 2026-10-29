"""Defense roadmap #3: defensive events the parser discards. Per player-season counts:
  offd  offensive fouls drawn (all), chg  charges drawn (subset), sfl/nsfl shooting / non-shooting personal fouls
  committed, blk_rim / blk_paint / blk_jump blocks by zone of the blocked shot (RA / paint non-RA / mid+3).
v2 (<=2024): fouls EVENTMSGTYPE 6 (4 OFF.Foul, 26 charge; drawer = PLAYER2), shooting = 2; blocks = PLAYER3 on misses,
zone from shotdetail. cdn (2025+): foul subType/descriptor, foulDrawnPersonId, blockPersonId, zone from coordinates."""
import sys, os, numpy as np, pandas as pd
import parse_pbp as P, cdn_zone

TECHY = [6, 9, 10, 11, 12, 13, 16, 17, 18, 25, 30]  # away-from-play/clear path/techs/etc: not counted as sfl/nsfl


def v2(season):
    d = pd.read_csv(P.read_csv_from_tar(f"nbastats_{season}"), low_memory=False).drop_duplicates(["GAME_ID", "EVENTNUM"])
    p1, p2, p3 = (d[c].fillna(0).astype(np.int64) for c in ["PLAYER1_ID", "PLAYER2_ID", "PLAYER3_ID"])
    t1, t2 = d.PLAYER1_TEAM_ID.fillna(0), d.PLAYER2_TEAM_ID.fillna(0)
    et, at = d.EVENTMSGTYPE, d.EVENTMSGACTIONTYPE
    foul = et == 6
    off = foul & at.isin([4, 26]) & (p2 > 0) & (t2 != t1) & d.PERSON2TYPE.isin([4, 5])
    rows = [pd.DataFrame({"player": p2[off], "offd": 1, "chg": (at[off] == 26).astype(int)})]
    pers = foul & (p1 > 0) & d.PERSON1TYPE.isin([4, 5]) & ~at.isin([4, 26] + TECHY + [14, 15])
    rows.append(pd.DataFrame({"player": p1[pers], "sfl": (at[pers] == 2).astype(int), "nsfl": (at[pers] != 2).astype(int)}))
    blk = (et == 2) & (p3 > 0) & d.PERSON3TYPE.isin([4, 5])
    sd = pd.read_csv(P.read_csv_from_tar(f"shotdetail_{season}"), usecols=["GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC"]) \
        .drop_duplicates(["GAME_ID", "GAME_EVENT_ID"])
    z = d[blk][["GAME_ID", "EVENTNUM"]].merge(sd, left_on=["GAME_ID", "EVENTNUM"], right_on=["GAME_ID", "GAME_EVENT_ID"],
                                              how="left").SHOT_ZONE_BASIC.map(P.ZONE).fillna(0).to_numpy()
    rows.append(zone_rows(p3[blk].to_numpy(), z))
    return rows


def zone_rows(pl, z):
    return pd.DataFrame({"player": pl, "blk_rim": (z == 1).astype(int), "blk_paint": (z == 2).astype(int),
                         "blk_jump": (z >= 3).astype(int), "blk_nz": (z == 0).astype(int)})


def cdn(season, df=None):
    d = df if df is not None else pd.read_csv(P.read_csv_from_tar(f"cdnnba_{season}"), low_memory=False)
    d = d.drop_duplicates(["gameId", "actionNumber"])
    pid, fd = d.personId.fillna(0).astype(np.int64), d.foulDrawnPersonId.fillna(0).astype(np.int64)
    foul = d.actionType == "foul"
    off = foul & (d.subType == "offensive") & (fd > 0)
    rows = [pd.DataFrame({"player": fd[off], "offd": 1, "chg": (d.descriptor[off] == "charge").astype(int)})]
    pers = foul & (d.subType == "personal") & (pid > 0) & ~d.descriptor.isin(
        ["flagrant-type-1", "flagrant-type-2", "away-from-play", "clear-path", "double"])
    rows.append(pd.DataFrame({"player": pid[pers], "sfl": (d.descriptor[pers] == "shooting").astype(int),
                              "nsfl": (d.descriptor[pers] != "shooting").astype(int)}))
    sh = d[d.actionType.isin(["2pt", "3pt"]) & (d.blockPersonId.fillna(0) > 0)]
    z = cdn_zone.zone_from_xy(sh.xLegacy, sh.yLegacy, sh.actionType == "3pt")
    rows.append(zone_rows(sh.blockPersonId.astype(np.int64).to_numpy(), z))
    return rows


def extract(season):
    rows = cdn(season) if season >= 2025 else v2(season)
    out = pd.concat(rows).fillna(0).groupby("player").sum().astype(int).reset_index()
    out["season"] = season
    out.to_parquet(f"{P.OUT}/defev_{season}.parquet", index=False)
    print(season, out[["offd", "chg", "sfl", "nsfl", "blk_rim", "blk_paint", "blk_jump", "blk_nz"]].sum().to_dict(), flush=True)


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        extract(s)
