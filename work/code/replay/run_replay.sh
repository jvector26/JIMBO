#!/bin/bash
# multi-season nightly replay: build_state S --replay, then nightly.py at each checkpoint (default config) -> out/S_cp/latest.json
cd /home/claude/rp
for S in "$@"; do
  python3 pipeline/build_state.py $S --replay > /dev/null
  for cp in 0 5 10 20 30 41 55 70; do
    d=$(python3 -c "import json;print(json.load(open('checkpoints.json'))['$S']['$cp'])")
    python3 pipeline/nightly.py --season $S --state state/${S}_replay --parsed /home/claude/data --schedule-from-parsed \
      --asof ${d:0:4}-${d:4:2}-${d:6:2} --sims 500 --out out/${S}_$cp --no-history > out/log_${S}_$cp.txt 2>&1 || echo "FAIL $S $cp"
  done
  echo "done $S"
done
