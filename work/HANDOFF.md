# JIMBO handoff: current state (read this first)
JIMBO = judgment-induced model for basketball outcomes. NBA 2026-27 player ratings, minutes, team win totals, playoff odds,
game odds, nightly in-season updates. Updated 2026-10-29 (session 35: matchups tab removed, injury box; session 34: NO MORE ZIP; everything lives in the repo, sandbox tree in work/). 
Full history of every test and decision: RESEARCH_LOG.md (1,650 lines; don't read it whole:
`grep -n "^## " RESEARCH_LOG.md`, then read the section you need).
Design spec (Claude Doc): https://claude.ai/code/artifact/e5eebcf8-5649-4c33-b974-23aa18e5c675
To start a chat: say "continue JIMBO". Claude attaches jvector26/JIMBO, runs work/setup.sh, reads this file (repo CLAUDE.md says the same).
The claude.ai project doc HANDOFF-3.md only points here; this file is the one source.

## SESSION 37 (2026-10-29, user question: back-to-backs) -- TESTED, NOT APPLIED (awaiting OK)
B2B was display-only. Night-2 team effect ~ -2.2 pts total (FE, t -12.6), -1.7 with actual minutes (research bar: LOSO 13.4633 -> 13.4367,
t -4.2, both halves). Older 30+ mpg players sit +8-9 pts more on night 2. Season-sim effect +-0.3 wins. Proposal + numbers: RESEARCH_LOG
"BACK-TO-BACKS"; code/b2b/.

## SESSION 36 (2026-10-29, user request, display + pipeline data only) -- SHIPPED gh 803d9a8
Tonight tab day stepper: ‹ › arrows step back through every game day so far and forward up to 3 GAME days; "Back to tonight" link.
- Past days ("Results"): from vegas.json rows, which now carry [.., home pts, away pts, game id, JIMBO p_home] (fields 6-9; vs Vegas
  tab still reads 0-5). Card = final score (winner bold), JIMBO pre-tip margin + win chance, line, final margin, who was closer
  (miss vs miss), whether JIMBO's side vs the line covered; day summary (closer X of N, side W-L, average miss) via vgStats.
  Not tappable (idea: tap -> box score from gamelog.json).
- Future days ("Coming up"): nightly.py day_games(day) builds cards for game_day (= tonight) and latest.json "upcoming"
  [{day, games}] for the next 3 game days, same team strengths as tonight (today's rotations + injury report; injured players'
  return dates are NOT applied to later days), own rest + odds/{day}.csv. The page recomputes spreads client-side
  (tonightGames(list, day)), so user edits apply; game page opens for any of these days (allGames()).
- Workflow fetches ESPN odds for today+1..+3 (ESPN usually posts ~1 day ahead; "No line yet" otherwise).
sw.js jimbo-a008253994 (hand bump). Tested: playwright phone + desktop, every day back to 10-20, no overflow, no page errors; skill drill passes.

## SESSION 35 (2026-10-29, user requests, display only) -- SHIPPED
1. Head to head / Matchups tab REMOVED (user: "I don't use it, and won't"): tab, panel, bottom-nav button, renderH2H and its CSS
   gone from code/dash_v2.html (also leaves the claude.ai artifact at its next rebuild; v43 still has it).
2. Player card injury box (app, SEASON only, injCard()): under the name for anyone on the ESPN report: status, side + body part +
   type ("Out · left knee inflammation"), expected back date + about N games (z.miss), "Out for the season" when ret > 2027-04-20
   (ESPN's Jul 1 placeholder), reported date. Same wording (injWhat) on the game-page rotation note and the This-season tab.
   nightly.py: new per-player fields ib (body part: '(knee)' after the name, else first known word in the note), is (side),
   iu (report date, ET). Still NO ESPN comment text on the public site (session 13 rule). Ratings/minutes unchanged.
   sw.js cache jimbo-a008253993 (hand bump: payload unchanged).

## SESSION 34 (2026-10-29, housekeeping) -- read first
ZIP RETIRED: the sandbox tree (code/, data/, raw/, dashboard.html, this file, RESEARCH_LOG.md, rebuild scripts) is now
work/ in the repo; work/setup.sh links it into /home/claude, so edits land in the repo: commit + push them (git pull --rebase first).
work/.gitignore ignores data/* and raw/*: tracked files update normally, but a NEW production input needs `git add -f`;
research outputs and the big rebuildable data stay out (as before). The repo is public: nothing private goes in work/.
End-of-session housekeeping = update this file (+ RESEARCH_LOG.md) and push. Project doc HANDOFF-3.md is now just a pointer here.
TEMPLATE SYNCED. Sessions 29-33 had no zip, so they patched only the built site/index.html and left diffs in repo tools/.
code/dash_v2.html + code/appsrc/app.js now carry all seven, applied in order: vegas_tab, wins_chart, season_tab,
season_leaders, season_pure, rookie_note, tonight_trim. Proof: build_app.py output = live site/index.html byte for byte
(repo b0e3db2), dash.json identical (version 2e949ae8f1). The tools/*_template.patch files in the repo are now HISTORY ONLY:
do not apply them again. Only remaining build difference: live sw.js cache name was hand-bumped (jimbo-a008253992); a fresh
build_app.py writes jimbo-<payload hash>. Bump it by hand (or change the payload) whenever index.html changes without a
payload change, or phones keep the old page.
Tests: code/tests/skill_drill_test.py clicks the Projection/Career player tabs first (session 31 note); all test scripts now use
plain p.chromium.launch() (sandbox browser path changes between images). Skill drill (phone + desktop) and test_dashboard.js pass.
What sessions 27-33 did (full notes: RESEARCH_LOG bottom + project docs claude/*.md):
- s27 10-19 college BPM vs rookie prior: nothing applied (2nd-round/UDFA BPM prior is a backlog lead).
- s28 10-20 opening night: final pre-opener rebuild, odds fixes, artifact v43 (below).
- s29 10-22 "vs Vegas" app tab (pipeline logs live/2026/pregame.csv, site/data/vegas.json).
- s30 10-23 team-view wins chart starts at opening night.
- s31 10-26 player page tabs This season | Projection | Career + Players-tab season leaderboard. USER RULE: "This season" views
  are pure 2026-27 numbers, no shrink/prior. s32 10-27 rookie-note fix (Haliburton).
- s33 10-28 tonight's minutes fix (healthy DNPs count, prior cap, trim by minutes x chance he plays; JIMBO_GAME_VARIANT "abd").
OWED (from s33): 2014-25 replay of the tonight rule including non-appearing players (needs player_games + pre_P_{S});
season-minutes rule still treats never-appeared healthy players as 100% prior (same fix (a) could apply; needs inseason_minutes.py replay).

## SESSION 28 (2026-10-20, opening night) -- read first
Shipped with user OK (gh 1eea2da, nightly 437df06 verified; artifact v42; then v43 below): FINAL PRE-OPENER REBUILD (chart 20261019, Duren de-exempted:
data/depth_config.json exempt = {}), state/2026 rebuilt, app rebuilt. Reason it mattered: nightly forecast = model margin + FIXED preseason gap
(teams_pre), so the camp cuts had moved forecasts 1:1 instead of through the 0.373 blend (TOR 50.1 -> 47.8, OKC 65.4 -> 63.9, DAL 33.2 -> 34.8...).
The old note "the rebuild only moves pre_w" was wrong. Same lesson for any future preseason-state change.
Odds fix: nba_fetch.fetch_odds maps ESPN SA/GS/NY/NO/UTAH/WSH to tricodes and merges with the day's earlier file (ESPN drops odds at tip);
nightly maps older files too. Two extra cron slots (16:29, 22:11 UTC): scheduled runs were landing 3-7 h late.
Later same night: the 10-19 chart (fetched ~7 pm ET) still had Kevin Porter Jr. (waived 10-19 evening). Fix (gh d357361 + rebuild): --until 2026-10-20,
nightly _cut = first game date (opening-day chart counts), fetched realgm_20261020 (dropped KPJ, Pedulla, Knecht, Clayton, Peavy, Brea, Flagler,
Powell, Huntley; Timme -> PHX), state rebuilt on it, artifact v43. MIL 28.1 -> 26.8. So decision (a) was NOT moot: the chart needs a run after the cuts settle.
Depth chart: fetching stopped 10-20 (--until), and nightly only ever uses a chart dated on/before the first game (_cut), so in season roles come
from actual minutes (w = n/(n+5)), ESPN injuries, teams from box scores (last team played for), user edits.
User cleared his 7 role edits + reset sliders in the app on 10-19 (his action). Branch pending/opening-night-2026-10-20 still on GitHub
(proxy blocks branch deletes; harmless, user can delete). Details: project doc claude/opening_night_check_2026-10-20.md.

## ONE CODE TREE (since 2026-10-18)
The 2026-10-07 fork (session 21 playoffs vs session 22 shrink) and session 23's built-page-only role patch are RECONCILED in this
zip: it has shrink + playoffs + roles + sepposs research code. The old warnings (apply_addon.py, patch_dash_roles.py, "never run
build_dashboard.py from 10-06c") no longer apply; do not apply old add-ons on top. Proof: with the depth chart pinned to
20261007 (DEPTH_DATE=20261007 python3 code/depth_apply.py), the chain reproduces the live dash.json (0 player-field differences,
meta.po and shrink included) and build_app.py reproduces the live site/index.html. Details: RESEARCH_LOG "CLEANUP".
Re-verified 2026-10-18 evening (session 26) on the 2026-10-18c zip: code/dash_v2.html now also carries the Clingan display
regroup (commit 6658eb2, artifact v39) and "Forecast vs line" (c2e5e0b, artifact v40) and the team-view postseason line (9631394, v41); build_app.py output = live site/index.html
except the dash.json?v= version hash (py_sim Monte Carlo noise only). RULE: change code/dash_v2.html first, then build; never
patch only the built pages (if forced to, write the exact edit into the handoff so the template gets it).
Convention: simulate.py's sim_2026.csv / meta.py_sim = series home court 4.0, NO po_delta (baseline for test_dashboard.js);
the page and nightly add po_delta (model/po.json, weight 0.5) on top.

## Where the season stands (2026-10-29)
In season since 10-20 (NBA Cup group play from 10-30). Nightly runs on GitHub Actions (cron slots drift 3-7 h late; dispatch by
hand if latest.json asof is stale). Depth chart frozen at the opening-day chart (realgm_20261020); roles now come from actual
minutes, ESPN injuries and user edits. The pre-opener plan (decisions a-c) is done; (d) five-starter rule in project.py stays open, low value.

## User and working style
- Knows basketball well; test his intuitions against data and report honestly, including when he's right. Bar for any
  model change: see "Research bar". Never change the model without his OK; report a failed test plainly.
- Can't run code; Claude runs everything. Usually on his phone: short answers, lead with the result.
- Does NOT want to be quizzed about player roles (he edits in the app). Doesn't want public models (EPM/DARKO/LEBRON)
  used as training targets. Doesn't bet; win-total comparisons are paper only.

## Setup (each new chat)
1. add_repo (jvector26/JIMBO, push), clone to /home/claude/jimbo, `bash /home/claude/jimbo/work/setup.sh` (links work/* into /home/claude,
   /home/claude/gh -> the repo, installs pyarrow + statsmodels). Work from /home/claude as always.
2. (old step: separate clone at /home/claude/gh; now the same repo via the link.)
3. Pushing: call add_repo (owner jvector26, repo JIMBO, access push); the Claude GitHub App is installed on the repo
   (2026-10-06), so the session's git proxy supplies credentials: no token needed (add_repo also wants a clone at
   /home/claude/jimbo for register_repo_root; pushing from /home/claude/gh works too). `git pull --rebase` first (nightly bot
   commits twice a day). Dispatch the nightly with REST (GraphQL is blocked, so `gh workflow run` fails):
   `gh api -X POST repos/jvector26/JIMBO/actions/workflows/nightly.yml/dispatches -f ref=main`.
   The gh API also needs add_repo first (otherwise HTTP 403 "GitHub access to this repository is not enabled").
4. Big rebuildable data is NOT in the repo: stints_{y}, player_games_{y}, player_games_all. `bash rebuild_git.sh 2012 2025` (rebuild.sh's github raw downloads are blocked now; rebuild_git.sh
   needs `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --no-checkout https://github.com/shufinskiy/nba_data
   /home/claude/shufinskiy/nba_data` first)
   (~15 s/season, foreground only; background jobs get killed) when a test needs stint- or player-game-level data.
   player_games_all: build_seasons.py FIRST BLOCK only (running it whole rewrites player_seasons without bio columns).
   data/sep_poss.parquet (sepposs research) regenerates from stints in ~1 min (code/sepposs/sep_poss.py).
5. Deliverables back to the user: SendUserFile limit is 30 MB. No zip to send any more: push to the repo instead.
6. Artifact read returns the page inside a hosting wrapper (an extra doctype/head at the top and "</body></html>" at the
   end); strip it (keep from `<!doctype html>\n<html lang="en">`) before republishing an edited copy.

## Where things live
- Published dashboard (claude.ai artifact, preseason view, db capability for saved edits):
  https://claude.ai/artifact/CfaYAX9qHUugsMpMjvLVLf  -> rebuild: build_dashboard.py -> dashboard.html -> publish same URL. Now v43.
  Preseason view only: the in-season features (vs Vegas, This season tabs, tonight cards) live in the phone app.
- Phone app (PWA): https://jvector26.github.io/JIMBO/ from repo site/ (pages.yml deploys after each nightly / site push).
- Nightly pipeline (GitHub Actions, repo .github/workflows/nightly.yml, 10:17 + 20:43 UTC; moved off :00 on 2026-10-08
  because GitHub delayed top-of-hour crons 3-7 h; scheduled runs are best-effort, so if latest.json asof is stale, just dispatch):
  nba_fetch (schedule, pbp, box, injuries ESPN, odds ESPN, RealGM depth chart until the --until date) -> pipeline/nightly.py
  (M4 in-season ratings, minutes rule, team offsets, rest-of-season sim) -> site/data/latest.json + history.json. Repo state/2026/ =
  preseason model packaged by pipeline/build_state.py (SANDBOX ONLY; rebuild + push whenever projections change).
- User edits: app -> overrides.json in the repo (players {key: t, g, mpg, r, d}); artifact -> db doc settings/current.
  Keys "p<player_id>" or "n:<name>|<team>".
- Standings columns (session 26, user request 2026-10-18): "Forecast vs line" = forecast minus line (includes the roster-age term,
  follows the model-vs-Vegas slider), colored at |gap| >= 2; replaced "Model vs line" (model alone minus line, colored at >= 3).
  Sort key still 'edge'. Model, Line and Age adj columns unchanged. User declined baking the age term into the model column.
- Team view postseason line (session 26, user request 2026-10-18; app 9631394, artifact v41): poPath() inside the .roster block
  shows Top-6 seed (acc.top6), Play-in (acc.pin = finished 7th-10th), Makes playoffs (acc.po = top 6 + play-in winners), Title,
  all from the page sim already computed. "Playoffs" everywhere means the 8-team bracket, not reaching the play-in.
- Skill drill-down (session 25, user request 2026-10-18; artifact v38 + app commit e20ef20, model data = 10-07 build): in the
  player view, tap a skill bar -> per-stat table: projection contribution (slope x f x slider weight; rows sum to the bar),
  2025-26 contribution (hist.v), 2025-26 actual stat with league average (payload hist[k].raw + meta.raw, from player_seasons +
  features poss). Since 6658eb2 / v39 (Clingan question): the 3PT-contest part of hustle (p.hb[1]) shows under perimeter defense
  (its own row, projection only) and the hustle bar is loose balls only; ratings unchanged. Open rows survive re-renders
  (pvSkOpen). Test: code/tests/skill_drill_test.py <dashboard.html or http URL of a built site> OUTDIR.
- Roles UI (session 23): Team view shows the CURRENT role: plain = chart default, bold purple = user's role edit, bold italic magenta
  "(auto)" = moved to keep five starters. Rule (page engine balanceRoles/assignRoles; nightly model/depth_chart.py balance_starters,
  logged as DEPTH.auto_tiers): user roles kept; short of 5 -> best non-starter by chart (R, then L, then N; most m0) moves up to S; over 5 ->
  chart starter with the fewest m0 moves to R; exempt X fills a slot when the chart lists < 5. project.py (sandbox) does NOT have it.

## Production chain (verified 2026-10-18; build_app re-verified 2026-10-29 from work/)
cd /home/claude:  code/depth_apply.py (latest gh/live/2026/depth chart -> data/rosters_2026.csv; DEPTH_DATE=YYYYMMDD pins a chart)
-> code/project.py -> code/newcomer_pos.py -> code/simulate.py -> code/build_dashboard.py -> node code/test_dashboard.js
+ python3 code/ui_edit_test_db.py /home/claude/dashboard.html (ABSOLUTE path; preinstalled playwright chromium) -> (cd gh && python3 pipeline/build_state.py 2026) -> local nightly check
(python3 pipeline/nightly.py --out /tmp/x, then node JSON.parse /tmp/x/latest.json: browsers reject NaN/Infinity) -> code/build_app.py
-> commit/push gh -> dispatch nightly -> republish the artifact from dashboard.html. Whole chain runs in ~20 s.
Sanity values: Ajay Mitchell rating 3.70 (shrink on); test_dashboard "max |model wins - project.py|" ~0.001; py_sim diff vs page sim: wins <= 0.4, title <= 0.01.
Ratings refit (rare): code/production.py (needs stints 2023-2025) -> ratings_2025.parquet. After changes to preseason
modelling: preseason_backtest.py (needs stints + player_games 2011-2025) -> score_preseason.py -> minutes_age_fix.py ->
refit_blend.py (this order: score_preseason rewrites preseason_backtest.parquet) -> regenerate blend_curve.json.
Quirk: simulate.py is not fully deterministic despite seed=7 (schedule graph draws), so p10/p90 can shift by 1 and means
by ~0.03 wins between runs. The dashboard pins its schedule in data/dash_sched_2026.json (keep that file).

## The model in one screen
- Ratings = points per 100 possessions vs average. Box part: 18 features in 12 components (three_pt, rim_finishing,
  pullup_mid, foul_drawing, self_creation, playmaking, ball_security, off_rebounding, def_rebounding, rim_protection,
  perimeter_def) with weights W learned from stint outcomes (weights_prod.json; backtest W weights_bt.json = 1996-2013);
  def_rebounding weight halved (baked into both files). Window = 3 seasons; box decay (.1,.4,1), lineup residual decay
  (.3,.6,1). Lineup residual (ridge RAPM toward the box prior, lam 2000) counts 0.6 on offense, 0.8 on defense.
  JUDGMENT SHRINK (user call 2026-10-07): residual x m, m = 1 - (1/3) x clip(1 - poss_window/10,000, 0, 1), symmetric,
  everyone (code/lineup_shrink.py, hooked right after ratings_2025 loads in project.py and build_dashboard.py; env
  LINEUP_SHRINK_A=0 = off, comparisons only). Cost: game LOSO +0.006, preseason model 8.643 -> 8.631, blend 8.139 -> 8.144.
  3PT accuracy uses a career-to-date + FT% talent prior (shooting.py, 1.5x). Hustle term (loose balls + 3PT contests,
  hustle_2026.csv) added to defense. d_oppxq / d_opprim are RAW on-court opponent shot profile (team-context heavy,
  kept on purpose: teammate-adjusted versions and mover discounts both predicted worse).
- Projection: slope 1.0501 x rating + age curve (data/aging.csv: <=21 +0.88 ... 36+ -0.83) + returner -0.35 for players
  back from a missed season; rookies from college stats (rookie_value.json, ncaa_scores.parquet, capped) or pick+age;
  other newcomers -4.31 (calib.csv).
- Minutes: learned model + age multipliers + recalibration (min_recal.json) + RealGM depth-chart tiers (depth_tier.json);
  team total 19,680, bottom-trim when crowded, cap 3,000. Always five starters. In season: nightly
  blends toward actual usage (w = n/(n+5)), ESPN injuries, user overrides; tonight's minutes use the game-level rule (DTD = 50% scenarios).
- Team: net = sum(rating x min)/3936 zero-summed; margin = net x pace/100; wins via sim (game p = Phi((dm + 2.6)/12.8)).
  Playoffs (2026-10-07): play-in + series use playoff strength = margin + 0.5 x playoff-minutes change (model/po.json,
  clipped to the 2014-25 range) and series home court 4.0 (play-in 2.6). Same code in nightly, simulate.py, app/dashboard JS.
- Forecast blend (blend.json, constrained): 41 + 0.373(model-41) + 0.627(Vegas-41) - 0.595(roster age - avg);
  tau 2.75 (model alone 3.03; = fitted x tau_cal 1.1365, replay-calibrated 2026-10-06). Vegas lines data/win_totals_2026.csv (BetMGM ~Sep 11). Dashboard has a model-vs-Vegas slider.
  The age term is fitted on the blend (it corrects a bias Vegas and the model share), so it stays outside the model column;
  possible later test: a model-only roster-age term in the preseason backtest, then refit the blend (user: "fine as is" for now).

## Research bar (how every proposed change is tested)
1. Game backtest: each season 2015-16..2025-26, ratings from the prior 3 seasons only, predict every game with ACTUAL
   minutes; leave-one-season-out fit; score = RMSE of margin. Files: data/backtest_components.parquet (per-feature game
   diffs) + data/bt_split.parquet (res_o/res_d). Base = box + 0.6 res_o + 0.8 res_d = LOSO 13.4633
   (code/bt_split_score.py; helper patterns in code/wcal.py, code/mover_context.py). Player minutes per game for new
   diffs: data/game_minutes.parquet (2014-2025, slim) or rebuild player_games. With the judgment shrink the game
   baseline is 13.4635 (code/lineup_port_bt.py harness; no shrink 13.4577).
2. Pass = lower LOSO, season-clustered t > 2.5, same sign in 2015-19 and 2020-25.
3. Then the preseason team backtest: each October 2014-25 projected from prior data + opening rosters
   (data/pre_P_{S}, pre_T_{S}); code/age_eval.py [TAG] scores it: model 8.643 wins RMSE, Vegas 8.440, blend LOSO 8.139
   (with judgment shrink: model 8.631, blend 8.144).

## Already tested and REJECTED or neutral (don't redo without a new angle; details in RESEARCH_LOG)
Market biases (big markets, close-game luck, star moves, playoff runs) | contract year (rating and minutes) |
young-player development beyond age bins | team-specific uncertainty | seller/tank signals | tank season down-weighting |
teammate-adjusted defensive sub-outcomes | charges/off fouls drawn, foul types, block zones | position-interacted
defensive weights | guards' rim-protection penalty (positional baseline, kept) | shift-sign model | bigs over-rating
(Clingan), heavier offensive ridge for small samples | flatter window decay | category weights (all confirmed) |
crowded-roster minutes rule alternatives | lineup portability for young/movers | young x rating age bump (suggestive,
retest after 2026-27) | chemistry/familiarity; new-team penalty -0.46/mover (fails halves, retest after 2026-27) |
mover discount on raw opponent shot profile (2026-10-06, Peyton Watson question) | team halo in lineup residuals (2026-10-06,
Ajay Mitchell; movers' halo -0.54 t -1.2, flips by half, recent era -1.16 t -2.2: retest after 2026-27) | thin-sample
high-lineup leverage in team forecasts (no forecast bias; they out-play projected minutes 1.11x) | diminishing returns (real for DREB and
self-creation at lineup level, not added to team projections) | rim DFG% / defended FG% (noise) | separated-possessions
shrink for stapled teammates (2026-10-15; beats total poss at every setting, both halves, but game -0.003, t ~2.0;
direct test neff t 2.50, 2015-19 t 0.3 / 2020-25 t 4.1; preseason flat; 2026 effect max 0.18: retest neff after 2026-27) |
contested-3 for centers (2026-10-18, Clingan +0.44; centers' coef -0.03 vs non-centers +0.27, difference t -0.7, fails halves;
user: leave it; retest after 2026-27).
APPLIED over time: always-five-starters role rule (user call, 2026-10-09), thin-sample lineup shrink (judgment, 2026-10-07), playoff minutes (half weight, 2026-10-07), lineup 0.6/0.8 split, dreb 0.5x, box decay (.1,.4,1), age-curve refit, slope refit, 3PT talent prior,
college rookie prior, returner -0.35, hustle term, minutes recal, depth-chart tiers, game-level tonight minutes.

## Backlog
Done at opener: camp-invitee cleanup is handled by the daily RealGM chart (87 players dropped on the
10-18 chart); LAL (19 on chart) and MEM (20) still list more than 15 + 3 two-way, so cuts there are pending. Watch depth fetch
(live/2026/depth/fetch_log.txt; manual fallback: web_fetch the RealGM page, save as realgm_YYYYMMDD.html.gz).
After the opener: opening-week check (first real games parse, results lock, NBA-Cup TBD fill, tonight cards vs ESPN lines); check the cron
times (10:17 / 20:43 UTC) fire close to schedule, else an external trigger that dispatches via REST.
Later: in-playoffs sim (po_delta is 0 once G_rem = 0; needs real playoff bracket + results) | FantasyPros archived minutes test | tier for tonight's minutes; L-tier newcomer value 250 (untested) | forecast
response to chart/override changes (nightly moves 1:1; blend logic says ~0.37) | thin-roster scale-up of camp players |
line-movement history | pre-tip run for late injuries | sim in a Web Worker | in-season hustle refresh | push
notifications | coaching effect (B3) | split 3PT bar into volume vs accuracy (UI) | college prior for 2nd-round/undrafted and
2nd/3rd-year players | talent 3P% in stint luck adjustment | team-margin concavity | position-specific spacing/rim/OREB
weights | NCAA stage A+B refit together (never A alone: rerunning ncaa_model.py alone shifts cscores up to 0.66) |
app edge case: role edit + team move in one edit | rest/travel/altitude for game odds | teammate-fit interactions in the
lineup residual (untested beyond DREB / self-creation diminishing returns) | garbage time in box features (user, 2026-10-15): RAPM/stint fits already drop
garbage stints (impact.load_rows: Q4 start with |margin| >= 20, or last 6 min with >= 15), but features.py box rates and
on-court possessions include them; test dropping them from the box part (game backtest + preseason as usual) | model-only
roster-age term (see blend note; after the opener) | retest after 2026-27:
sepposs neff shrink, team halo, new-team penalty, young x rating bump, judgment shrink (does it help on 2026-27 preseason ratings?), contested-3 for centers.
Restored 2026-10-18 from MASTER BACKLOG (RESEARCH_LOG) details that the short list had dropped: alloc_test.py minutes floor 0.25
option (ties on teams, better player minutes; offered with item 8) | non-college rookie scale (x1.16-1.26 hint, with NCAA refit) |
better same-day lineups (pipeline-dependent, with rest/travel) | display-only 'vs position' baselines for player-view component bars
(offered session 17, Jrue Holiday question) | B3 coaching effect has a written plan in its RESEARCH_LOG section.

## What is in work/ and what is not
In: all code except backup copies; data the production chain, build_state, the research bar and rebuilds read (parquet zstd).
Slimmed: game_minutes.parquet, data/ncaa/player_box_*. Not in (regenerate from their script after rebuild_git.sh): stints_{y},
player_games_*, research outputs of finished tests (*_bt.parquet, defsub*, bigs_*, chem_*, tank_* ...), game_minutes_active.parquet,
ncaa_player_seasons.parquet (ncaa_build.py from hoopR raw files), sep_poss.parquet. code/make_core.py (old zip builder) is obsolete.
code/clingan_c3_test.py was never saved (session 25); the test is described in RESEARCH_LOG.
