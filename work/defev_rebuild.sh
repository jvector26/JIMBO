#!/bin/bash
cd /home/claude
for y in $(seq $1 $2); do
  for n in nbastats_$y shotdetail_$y; do
    [ -f raw/$n.csv ] || { curl -sL -m 600 -o raw/$n.tar.xz https://github.com/shufinskiy/nba_data/raw/main/datasets/$n.tar.xz; }
  done
  (cd code && python3 defev.py $y) 2>&1 | tail -1
  if [ $y -ge 2015 ] && [ ! -f data/player_games_$y.parquet ]; then (cd code && python3 parse_pbp.py $y) 2>&1 | tail -1; fi
  rm -f raw/*_$y.tar.xz raw/*_$y.csv
done
