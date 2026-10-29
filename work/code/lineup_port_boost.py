import numpy as np, pandas as pd
from lineup_port_bt import G, B, y, seas, diffs, loso, fixed
b0, s0 = fixed(diffs(np.ones(len(G))))
for cut in [2500, 4000]:
    for k in [-.3, .2, .4, .6]:
        m = 1 + k * (G.P < cut)
        l, s = fixed(diffs(m)); h1 = np.sqrt(np.mean(s[:5]**2)) - np.sqrt(np.mean(s0[:5]**2)); h2 = np.sqrt(np.mean(s[5:]**2)) - np.sqrt(np.mean(s0[5:]**2))
        print(f"thin<{cut} x{1+k:.1f}: LOSO {l:.4f} (d {l-b0:+.4f}), seasons better {int((s<s0).sum())}/11, halves {h1:+.4f} {h2:+.4f}")
