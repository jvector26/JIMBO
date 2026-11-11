#!/bin/bash
# raw archives from the local blobless clone of shufinskiy/nba_data (see rebuild_git.sh), then the extra pass parse_x.py
R=/home/claude/shufinskiy/nba_data
cd /home/claude && mkdir -p raw fund
for y in $(seq ${1:-2012} ${2:-2025}); do
  [ -f fund/xg_$y.parquet ] && continue
  pbp=nbastats_$y; [ $y -eq 2025 ] && pbp=cdnnba_$y
  for n in $pbp shotdetail_$y; do git -C $R -c gc.auto=0 cat-file blob HEAD:datasets/$n.tar.xz > raw/$n.tar.xz; done
  python3 code/fund/parse_x.py $y 2>&1 | tail -2
  rm -f raw/*_$y.tar.xz raw/*_$y.csv
done
