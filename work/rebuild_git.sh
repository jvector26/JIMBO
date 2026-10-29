#!/bin/bash
# like rebuild.sh, but takes the archives from the local blobless clone of shufinskiy/nba_data (github.com/raw is blocked here)
R=/home/claude/shufinskiy/nba_data
cd /home/claude && mkdir -p raw data
for y in $(seq ${1:-2011} ${2:-2025}); do
  if [ -f data/stints_$y.parquet ]; then continue; fi
  pbp=nbastats_$y; [ $y -eq 2025 ] && pbp=cdnnba_$y
  for n in $pbp shotdetail_$y; do
    git -C $R -c gc.auto=0 cat-file blob HEAD:datasets/$n.tar.xz > raw/$n.tar.xz
  done
  (cd code && python3 parse_pbp.py $y) 2>&1 | tail -1 | tee -a rebuild.log
  rm -f raw/*_$y.tar.xz raw/*_$y.csv
done
