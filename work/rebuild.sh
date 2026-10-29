#!/bin/bash
cd /home/claude && mkdir -p raw data
for y in $(seq ${1:-2011} ${2:-2025}); do
  if [ -f data/stints_$y.parquet ]; then continue; fi
  pbp=nbastats_$y; [ $y -eq 2025 ] && pbp=cdnnba_$y
  for n in $pbp shotdetail_$y; do
    curl -sL -m 600 -o raw/$n.tar.xz https://github.com/shufinskiy/nba_data/raw/main/datasets/$n.tar.xz
  done
  (cd code && python3 parse_pbp.py $y) 2>&1 | tail -1 | tee -a rebuild.log
  rm -f raw/*_$y.tar.xz raw/*_$y.csv
done
echo ALLDONE >> rebuild.log
