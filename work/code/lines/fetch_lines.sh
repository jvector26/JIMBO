#!/bin/bash
# Historical NBA game lines for research (session 47, 2026-11-09). Sources are public GitHub repos; data is NOT stored in this repo.
#  /tmp/odds/nba.json     FinnedAI/sportsbookreview-scraper data/nba_archive_10Y.json  (2011-12..2021-22, open + close spreads)
#  /tmp/odds/odds.sqlite  kyleskom/NBA-Machine-Learning-Sports-Betting Data/OddsData.sqlite (2007-08..2024-25, 2025-26 to Jan 7)
# Then, from /home/claude/lines (scripts write their outputs to the current directory):
#  python3 build_lines.py; python3 lines_table.py  -> hist_game_lines.csv (home-favoured-by points: open, close, line)
#  python3 info_test.py        preseason-ratings game backtest vs lines
#  (rebuild_git.sh 2013 2025; code/elo/elo_m4_preds.py) python3 info_m4.py      in-season M4 + ridge 80, actual minutes
#  python3 m4_trailing_preds.py; python3 info_trailing.py                        same with trailing 5-game minutes
#  LINES BAR: python3 make_variants.py; PRE_TAG=woff4_ python3 m4_var_preds.py 2015 .. 2025 (also shr_, neff_; ~4 min each; base = m4tr_{S} from
#  m4_trailing_preds.py); python3 lines_bar.py woff4_ shr_ neff_; python3 lines_decomp.py  (needs sep_poss.parquet: code/sepposs/sep_poss.py, stints 2011+)
set -e
mkdir -p /tmp/odds /home/claude/lines && cd /tmp/odds
git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/FinnedAI/sportsbookreview-scraper finned
git -C finned cat-file blob HEAD:data/nba_archive_10Y.json > nba.json
git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting kyle
git -C kyle cat-file blob HEAD:Data/OddsData.sqlite > odds.sqlite
cp /home/claude/code/lines/*.py /home/claude/lines/
