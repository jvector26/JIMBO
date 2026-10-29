"""Build the slim handoff zip tree (po.json, sep_* added 2026-10-18). usage: python3 code/make_core.py /home/claude OUTDIR ; then (cd OUTDIR && zip -q -9 -r ../jimbo_handoff.zip .)
Keep-list = files the production chain, build_state, the research bar and rebuilds read (traced 2026-10-06). Add new
production inputs to KEEP when a change creates one; research outputs stay out (log results in RESEARCH_LOG.md)."""
import os, sys, shutil, glob, re
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ORIG, CORE = sys.argv[1], sys.argv[2]
os.makedirs(CORE, exist_ok=True)

# ---- data: explicit keep list (traced reads of the production chain + research-bar inputs + rebuild inputs)
KEEP = """
aging.csv age_curves.csv backtest_components.parquet blend.json blend_curve.json bt_split.parquet bt_split_ratings.parquet
calib.csv dash_sched_2026.json depth_config.json depth_tier.json depth_2026_log.csv direction_features.parquet draft_r1.parquet
features.parquet games_all.parquet hist_lines.csv hustle.json hustle_2026.csv inseason.json lineup_hist.parquet
min_ba.npy min_bm.npy min_recal.json ncaa_draft_match.parquet ncaa_scores.parquet pace_2025.json player_seasons.parquet
players_fa_2026.csv players_proj_2026.csv pos_2026.csv positions.parquet preseason_backtest.parquet ratings_2025.parquet
returner.json rookie_min_coef.npy rookie_value.json rosters_2026.csv rosters_2026_base.csv sim_2026.csv slope_refit.json
teams_proj_2026.csv weights_bt.json weights_prod.json win_totals_2026.csv bio.json birthdates.parquet
po.json sep_k.json sep_game_grid.csv
""".split()
KEEP += [os.path.basename(p) for p in glob.glob(f"{ORIG}/data/pre_[PT]_*.parquet")]
KEEP += [os.path.basename(p) for p in glob.glob(f"{ORIG}/data/games_[0-9]*.parquet")]
KEEP += [os.path.basename(p) for p in glob.glob(f"{ORIG}/data/names_[0-9]*.parquet")]

def put_parquet(src, dst, cols=None, df_fn=None):
    t = pq.read_table(src, columns=cols)
    if df_fn is not None:
        t = pa.Table.from_pandas(df_fn(t.to_pandas()), preserve_index=False)
    pq.write_table(t, dst, compression="zstd", compression_level=19)

os.makedirs(f"{CORE}/data/ncaa", exist_ok=True)
for b in sorted(set(KEEP)):
    s, d = f"{ORIG}/data/{b}", f"{CORE}/data/{b}"
    if not os.path.exists(s):
        print("MISSING", b); continue
    if b.endswith(".parquet"): put_parquet(s, d)
    else: shutil.copy2(s, d)

# slim game-level minutes (2014-25): only the columns research scripts need for actual-minute game diffs
put_parquet(f"{ORIG}/data/game_minutes.parquet", f"{CORE}/data/game_minutes.parquet",
            cols=["season", "game", "player", "team", "min", "started", "date"])
# slim college box scores (newcomer_pos.py inputs): drop logos/colours/urls
NC = ["game_id", "season", "season_type", "game_date", "athlete_id", "athlete_display_name", "team_id", "team_display_name",
      "minutes", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
      "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "rebounds", "assists", "steals",
      "blocks", "turnovers", "fouls", "points", "starter", "did_not_play", "athlete_position_abbreviation", "opponent_team_id"]
for p in sorted(glob.glob(f"{ORIG}/data/ncaa/player_box_*.parquet")):
    put_parquet(p, f"{CORE}/data/ncaa/{os.path.basename(p)}", cols=NC)

# ---- code: everything except backups, old templates, caches
BACKUP = re.compile(r"(_pre_|^dash_template\.html$|__pycache__)")
for root, dirs, files in os.walk(f"{ORIG}/code"):
    dirs[:] = [x for x in dirs if x != "__pycache__"]
    rel = os.path.relpath(root, ORIG)
    for fn in files:
        if BACKUP.search(fn): continue
        os.makedirs(f"{CORE}/{rel}", exist_ok=True); shutil.copy2(f"{root}/{fn}", f"{CORE}/{rel}/{fn}")
for fn in ["dashboard.html", "rebuild.sh", "rebuild_git.sh", "defev_rebuild.sh", "HANDOFF.md", "RESEARCH_LOG.md"]:
    shutil.copy2(f"{ORIG}/{fn}", f"{CORE}/{fn}")
for fn in glob.glob(f"{ORIG}/gh_unpushed_*.patch"):   # repo commits that could not be pushed (apply with git am)
    shutil.copy2(fn, f"{CORE}/{os.path.basename(fn)}")
os.makedirs(f"{CORE}/raw", exist_ok=True)  # rebuild.sh downloads here
print("done")
