"""Parse fetched live games (live/{season}/pbp/*.csv.gz + games_meta.csv) with JIMBO's parser.
Zones come from shot coordinates (cdn_zone.py, 99.6% agreement with stats.nba.com zones on 2025-26)."""
import glob, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import parse_pbp as P  # noqa: E402


def load(season):
    files = sorted(glob.glob(os.path.join(ROOT, "live", str(season), "pbp", "*.csv.gz")))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    meta = pd.read_csv(os.path.join(ROOT, "live", str(season), "games_meta.csv"))
    meta = meta[meta.game.isin(df.gameId.unique())].copy()
    meta["date"] = meta["date_et"].astype(str).str.replace("-", "").astype(int)
    return df, meta


def parse(season, out):
    df, meta = load(season)
    P.run(season, cdn_df=df, meta=meta[["game", "date", "htm", "vtm"]], xy_zones=True, out=out)


if __name__ == "__main__":
    parse(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "state", "season"))
