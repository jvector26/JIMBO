"""RealGM depth chart -> minutes tier (S starter / R rotation / L limited / N not on his team's chart).
Shared by the sandbox (code/depth_apply.py -> project.py) and the nightly pipeline (pipeline/nightly.py, preseason only).
Tier model (code/depth_test.py, depth_fit.py; 2015-25 backtest): want = a[tier] + b[tier] * m0, clip 0..3000,
m0 = JIMBO's recalibrated preseason minutes (incl. age multiplier), then the usual team allocation to 19,680."""
import gzip, re, unicodedata, numpy as np, pandas as pd

TEAMS = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
         "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
         "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
         "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
         "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
         "Orlando Magic": "ORL", "Philadelphia Sixers": "PHI", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
         "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
         "Utah Jazz": "UTA", "Washington Wizards": "WAS"}
TIER = {"depth_starters": "S", "depth_rotation": "R", "depth_limpt": "L"}
# RealGM slug (normalised) -> our name (normalised). Add here when a chart name fails to match.
ALIAS = {"herbjones": "herbertjones", "moewagner": "moritzwagner", "dennisschroeder": "dennisschroder",
         "enesfreedom": "eneskanter", "nicolasclaxton": "nicclaxton", "ronholland": "ronaldholland",
         "ejharkless": "elijahharkless", "cameronchristie": "camchristie", "bjbostonjr": "brandonboston",
         "jameshuff": "jayhuff"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-]", " ", s); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


def parse(html):
    out = []
    for m in re.finditer(r"<h2[^>]*>(?:\s*<img[^>]*>)?\s*(\d{4})-(\d{4}) ([^<]+?) Depth Chart\s*</h2>(.*?)</table>", html, re.S):
        yr, team, body = int(m.group(1)), m.group(3).strip(), m.group(4)
        for ri, tr in enumerate(re.finditer(r'<tr class="(depth_\w+)">(.*?)</tr>', body, re.S)):
            tier = TIER.get(tr.group(1))
            cells = re.findall(r'<td data-th="[^"]*" class="depth-chart-cell"[^>]*>(.*?)</td>', tr.group(2), re.S)
            for ci, c in enumerate(cells):
                for a in re.finditer(r'/player/([^/]+)/Summary/(\d+)"', c):
                    out.append(dict(label_year=yr, team_name=team, tier=tier, row=ri, col=ci, slug=a.group(1), rgm=int(a.group(2))))
    return out


def load_chart(path, season):
    """Latest chart file -> one row per player (his highest listing), columns team, tier, row, col, slug, rgm, nk."""
    R = pd.DataFrame(parse(gzip.open(path).read().decode("latin-1")))
    if R.empty: return R
    R["team"] = R.team_name.map(TEAMS)
    R = R[(R.label_year == season) & R.team.notna()]
    R = R.sort_values(["team", "row", "col"]).drop_duplicates("rgm")
    R["nk"] = R.slug.map(norm).map(lambda k: ALIAS.get(k, k))
    return R.reset_index(drop=True)


def match(R, people):
    """people: DataFrame [key, name, team] of everyone we know (rostered + FA + unrostered). Returns R with column 'key'
    (None if unmatched). Exact normalised name first (ties -> same team), then unique last name on the same team."""
    pp = people.assign(nk=people["name"].map(norm))
    by = {k: g for k, g in pp.groupby("nk")}
    pp["last"] = pp["name"].astype(str).str.split().map(lambda w: norm([x for x in w if norm(x)][-1]) if len(w) else "")
    keys, how = [], []
    for nk, tm, slug in zip(R.nk, R.team, R.slug):
        g = by.get(nk)
        if g is not None and len(g) > 1: g = g[g.team == tm] if (g.team == tm).any() else g.iloc[:1]
        if g is not None and len(g): keys.append(g.key.iloc[0]); how.append(0); continue
        parts = [w for w in slug.split("-") if not re.fullmatch(r"(jr|sr|ii|iii|iv)", w.lower())]
        ln = norm(parts[-1]) if parts else ""
        c = pp[(pp["last"] == ln) & (pp.team == tm)]
        keys.append(c.key.iloc[0] if len(c) == 1 else None); how.append(1)
    # one chart row per key (2026-09-30: new camp signing Kobe Johnson matched Jalen Johnson by last name ->
    # duplicate key -> nightly crashed). Exact-name matches win; a last-name match never takes a key already used.
    exact = {k for k, h in zip(keys, how) if h == 0 and k is not None}
    seen = set(); out = []
    for k, h in zip(keys, how):
        if k is None or (h == 1 and k in exact) or k in seen: out.append(None); continue
        seen.add(k); out.append(k)
    return R.assign(key=out)


def tiers(P, R, exempt=()):
    """P: [key, team] (team after roster moves). Tier = his listing on HIS team's chart, else N; exempt keys -> 'X' (no change)."""
    own = R.dropna(subset=["key"]).set_index("key")
    t = [own.tier.get(k) if (k in own.index and own.team.get(k) == tm) else "N" for k, tm in zip(P.key, P.team)]
    return pd.Series(["X" if k in set(exempt) else x for k, x in zip(P.key, t)], index=P.index)


def want(m0, tier, coef):
    """Tier model. tier X (exempt) or FA -> m0 unchanged."""
    a = np.array([coef[t][0] if t in coef else 0.0 for t in tier]); b = np.array([coef[t][1] if t in coef else 1.0 for t in tier])
    return np.clip(a + b * np.asarray(m0, float), 0, 3000)


def allocate(x, total=19680.0, cap=3000.0):
    """Team minutes: crowded rosters lose minutes from the bottom first; thin rosters scale up (project.py rule)."""
    x = pd.Series(x, dtype=float).clip(0, cap).copy(); ex = x.sum() - total
    if ex > 0:
        for i in x.sort_values().index:
            c = min(ex, x[i]); x[i] -= c; ex -= c
            if ex <= 0: break
    elif x.sum() > 0:
        for _ in range(10):
            room = x < cap
            if x[room].sum() <= 0: break
            x[room] *= (total - x[~room].sum()) / x[room].sum(); x = x.clip(upper=cap)
    return x


def balance_starters(P, chart, user, team_ov, n=5):
    """Always five starters per team (user call 2026-10-09; the app's engine has the same rule, balanceRoles).
    P: rows [key, team, m0] (team before user team moves); chart: chart tier per row (S/R/L/N/X); user: {key: role};
    team_ov: {key: team} user team moves. Players with a user role keep it; players a user moved off their chart team
    don't count. Short of five -> the best non-starter by chart (R, then L, then N; most m0, then key) moves up to S.
    Over five -> the chart starter with the fewest m0 moves down to R. An exempt (X) player without a user role fills a
    starter slot when the chart lists fewer than five. Returns {row index: new role} for the automatic moves only."""
    rk = {"R": 0, "L": 1, "N": 2}; out = {}
    df = pd.DataFrame({"key": P.key.values, "team": [team_ov.get(k, t) for k, t in zip(P.key, P.team)],
                       "m0": pd.to_numeric(P.m0, errors="coerce").fillna(0).values,
                       "c": [None if k in team_ov else (c if isinstance(c, str) else None) for k, c in zip(P.key, chart)],
                       "u": [user.get(k) for k in P.key]}, index=P.index)
    df["d"] = df.u.where(df.u.notna(), df.c)
    for t, g in df.groupby("team"):
        if t == "FA": continue
        free = g[g.u.isna()]
        nS = int((g.c == "S").sum()); nX = int(((g.c == "X") & g.u.isna()).sum())
        target = n - min(nX, max(0, n - nS)); cur = int((g.d == "S").sum())
        if cur < target:
            c = free[free.c.isin(list(rk))]
            c = c.assign(_r=c.c.map(rk)).sort_values(["_r", "m0", "key"], ascending=[True, False, True], kind="mergesort")
            for i in c.index[:target - cur]: out[i] = "S"
        elif cur > target:
            c = free[free.c == "S"].sort_values(["m0", "key"], ascending=[True, True], kind="mergesort")
            for i in c.index[:cur - target]: out[i] = "R"
    return out
