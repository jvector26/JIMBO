"""Judgment shrink on lineup residuals (user call, 2026-10-07; RESEARCH_LOG 'LINEUP SHRINK').
Extra shrink toward zero for thin windows, symmetric: res x m, m = 1 - A * clip(1 - poss_window / P1, 0, 1).
Cost when adopted: game LOSO +0.006 (13.4577 -> 13.4635), preseason model RMSE 8.643 -> 8.631, blend 8.139 -> 8.144.
Set A = 0 to switch off."""
import numpy as np
import os
A, P1 = float(os.environ.get("LINEUP_SHRINK_A", 1 / 3)), 10000.0   # env override for comparisons only

def mult(poss_window):
    return 1 - A * np.clip(1 - np.asarray(poss_window, dtype=float) / P1, 0, 1)

def apply(rt):
    """rt: ratings frame with res_o, res_d, off, def, total, lineup_impact, poss_window. Returns a shrunk copy."""
    rt = rt.copy(); m = mult(rt.poss_window.fillna(0))
    do, dd = (1 - m) * rt.res_o, (1 - m) * rt.res_d
    rt["res_o"] -= do; rt["res_d"] -= dd; rt["off"] -= do; rt["def"] -= dd; rt["total"] -= do + dd
    rt["lineup_impact"] = rt.res_o + rt.res_d; rt["lineup_mult"] = m
    return rt
