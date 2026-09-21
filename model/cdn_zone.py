"""Shot zone from cdn legacy coordinates (xLegacy, yLegacy in 1/10 ft, hoop at 0,0), mimicking stats.nba.com
SHOT_ZONE_BASIC (used for 1996-2025 via shotdetail). 1 RA, 2 paint non-RA, 3 mid, 4 corner 3, 5 above-break 3, 6 backcourt."""
import numpy as np

def zone_from_xy(x, y, is3):
    x = np.asarray(x, float); y = np.asarray(y, float); is3 = np.asarray(is3, bool)
    r = np.hypot(x, y)
    z = np.full(x.shape, 3)
    paint = (np.abs(x) < 80) & (y < 142.5)
    z[paint] = 2
    z[r <= RA_R] = 1
    z[is3 & (np.abs(x) >= 220) & (y < CORNER_Y)] = 4
    z[is3 & ~((np.abs(x) >= 220) & (y < CORNER_Y))] = 5
    z[y > BACK_Y] = 6
    z[np.isnan(x) | np.isnan(y)] = 0
    return z

RA_R, CORNER_Y, BACK_Y = 40.0, 87.5, 417.5  # fit to 2025-26 shotdetail: 99.6% zone agreement (cdn "area" field only 97.0%)
