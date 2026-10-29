"""Head-to-head at matched settings: variant (separated poss) vs total poss, same A and P1."""
import sys, io, contextlib, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/code/sepposs"); sys.path.insert(0, "/home/claude/code")
with contextlib.redirect_stdout(io.StringIO()):
    import sep_game as SG
G = SG.G; rows = []
for P1 in [8000, 10000, 12000, 15000]:
    for a in [.25, 1 / 3, .5]:
        mt = SG.mult(G.Praw, a, P1); pt = SG.pred(SG.diffs2(mt, mt))
        for nm, co, cd in [("neff", "c_neff_o", "c_neff_d"), ("1-top1", "c_sh1", "c_sh1"), ("sep2", "c_sep2", "c_sep2")]:
            r = SG.score(SG.pred(SG.diffs2(SG.mult(G.Praw * G[co], a, P1), SG.mult(G.Praw * G[cd], a, P1))), pt)
            r.update(var=nm, P1=P1, a=round(a, 2)); rows.append(r)
R = pd.DataFrame(rows)[["var", "P1", "a", "d", "better", "g", "t", "h1", "h2"]]
print(R.round(4).sort_values(["var", "P1", "a"]).to_string(index=False))
