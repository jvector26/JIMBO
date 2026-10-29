#!/bin/bash
# JIMBO sandbox setup (replaces the old handoff zip). Run once per new chat after cloning the repo to /home/claude/jimbo:
#   bash /home/claude/jimbo/work/setup.sh
# Links the working tree into /home/claude (scripts expect /home/claude/data, /home/claude/code, relative data/, and the
# repo at /home/claude/gh), so edits land straight in the repo: commit and push them like any other change.
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
for x in code data raw dashboard.html HANDOFF.md RESEARCH_LOG.md rebuild.sh rebuild_git.sh defev_rebuild.sh; do
  t=/home/claude/$x
  if [ -e "$t" ] && [ ! -L "$t" ]; then echo "SKIP $t: exists and is not a link (move it away first)"; continue; fi
  ln -sfn "$R/work/$x" "$t"
done
[ -e /home/claude/gh ] && [ ! -L /home/claude/gh ] && echo "SKIP /home/claude/gh: exists" || ln -sfn "$R" /home/claude/gh
pip install -q --break-system-packages pyarrow statsmodels 2>/dev/null
git -C "$R" config user.name >/dev/null || git -C "$R" config user.name "Claude"
echo "JIMBO sandbox ready: cd /home/claude; read HANDOFF.md"
