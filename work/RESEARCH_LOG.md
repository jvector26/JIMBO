# JIMBO research log (full history, sessions 1-20)
<!-- Moved here from HANDOFF.md on 2026-10-06. Read HANDOFF.md first; use this log by section (grep "^## ") when a topic comes up.
     File/backup names mentioned below may refer to files dropped from the slim zip (see HANDOFF.md "Dropped from the zip"). -->

The model is called JIMBO; use that name. Full functional design spec (Claude Doc, the rebuild blueprint):
https://claude.ai/code/artifact/e5eebcf8-5649-4c33-b974-23aa18e5c675  (read it with the docs tools if details are needed).
Upload `nba_model_handoff.zip` to the new chat and say: "Read HANDOFF.md and continue JIMBO." Unzip to /home/claude
so paths match (scripts read/write `data/`, i.e. /home/claude/data). Everything below was decided with the user; don't re-litigate without cause.

## Goal
Statistical model from free NBA.com data that projects every player and all 30 teams for 2026-27, updates daily in season.
User can't run Python locally; Claude runs everything in its sandbox. User wants interpretability, sliders, and overrides.

## Locked decisions
- Outputs: team win totals + playoff odds, game-by-game win probabilities, comparison vs betting lines.
  (Player stat lines not requested for display, but player projections exist internally.)
- Minutes: model predicts, user can override any player. User also wants per-player rating overrides (e.g. Tyler Herro).
- 12 components: three_pt, rim_finishing, pullup_mid, foul_drawing, self_creation, playmaking, ball_security,
  off_rebounding, def_rebounding, rim_protection, perimeter_def, lineup_impact (plus a "shooting efficiency emphasis" slider
  that scales only o_3eff, o_rimeff, o_mideff, o_fteff). Durability separate.
- Sliders re-center ratings so minutes-weighted league average stays 0. Dashboard shows backtest accuracy for current sliders.

## Data (rebuildable)
Source: github.com/shufinskiy/nba_data mirror of NBA.com (sandbox can reach github; stats.nba.com is blocked).
`list.txt` = https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt. Seasons labeled by start year (2025 = 2025-26).
- 1996-2024: `nbastats_{y}` (v2 pbp); 2025: `cdnnba_2025`; zones/dates from `shotdetail_{y}`.
- `code/parse_pbp.py` -> player_games_{y}, stints_{y} (10-man lineups; per side pts, fg2/fg3, ft, oreb, dreb, tov, x3/xft luck
  inputs, zone FGA za0-6, rim makes, margin_start), games_{y}. Validated: 100% of final scores match; two-feed cross-check 2024.
  Fixes baked in: dedupe EVENTNUM, sort by game clock, FT made = SCORE present, Charlotte id 1610612766->1610612740 for <=2001.
  Needs `shooter_pcts` from a prior player_games parse (run parse twice on a fresh rebuild, or accept league-average first pass).
- `build_seasons.py` -> player_games_all, games_all, player_seasons. `add_bio.py` (bio.json, HoopsHype repo) -> age/height/pos.
- Rebuild order: download tars (see earlier chat) -> parse_pbp.py 1996..2025 -> build_seasons.py -> add_bio.py -> features.py.
  Large files (stints, player_games_all) are NOT in this zip (size); rebuild takes ~15 min.
- Memory: sandbox has ~3 GB. impact.fit_weights uses integer-array loading + chunked normal equations + CG for >20k params.

## Impact model (impact.py)
- y = luck-adjusted points/100 poss per stint side (2*FG2M + 3*x3 + xft), garbage time dropped, weighted by possessions.
- Step 1 `fit_weights`: component weights W jointly with ridge player-season residuals (lam 4000), season intercepts, home.
  Production W: all 1996-2025 (weights_prod.json). Backtest W: 1996-2013 (weights_bt.json). HCA ~2.7 pts/100 recent.
- Step 2 `fit_window(N)`: seasons N-2..N, decay (0.3,0.6,1.0), prior strength ALPHA=0.5 on W, ridge LAM=2000 on residuals.
  rating = box components + lineup residual. ratings_2025.parquet = as of end of 2025-26 (has f:<feature> columns, res_o/res_d).
- LINEUP WEIGHT = 0.6 (new default). Evidence: backtest optimum 0.5-0.75x; dropping it hurts (t=4.4); persistence across
  separate 3-yr windows r=0.39 (box parts 0.81); centers no more persistent than others.
- Rebounding may be rich: backtest best at ~0.5x (13.477 vs 13.495), not significant; left at 1.0, user can slide.
- Center tilt is real: removing position gap hurts predictions (t=5.1).

## Backtest (13,206 games, 2014-15..2025-26; ratings through N predict N+1 with actual minutes)
Component model a0.5/l2000: 13.517 error, 66.1% winners (lineup 1.0). With lineup 0.6: ~13.49. Plain RAPM 13.530.
Prev-season MOV 14.13; home-only 14.71. `backtest_components.parquet` has per-feature game diffs for instant slider scoring.
Known past bug: NaN features from 2 tiny-sample players zeroed windows 2017-2020 in early backtests (fixed via fillna).

## Projections (project.py -> players_proj_2026.csv, teams_proj_2026.csv)
- Calibration (aging.py, lineup 0.6): slope 0.971, aging bins (<=21 -0.57 ... 34-35 -0.52, 36+ -1.26), rookie values
  pre-calibration: picks 1-3 -1.43, 4-10 -3.39, 11-30 -4.30, other newcomers -4.68.
- Minutes (history players): 50/50 blend of learned model (min_bm/min_ba; availability counts only from 20+ mpg seasons)
  and recency rule. R^2 0.534 out of sample. Shows mpg_when_playing and exp_games. Injured regulars bounce back to ~57%.
- Rookies (2026 first-rounders): minutes from rookie_min_coef (log pick, team prior win%, draft age); 844 rookies 1997-2025.
- Team minutes = 19,680; crowded rosters trimmed from the bottom; cap 3,000. Team net = sum(rating*min)/3936, zero-summed.
  Wins = 82*Phi(margin/12.8), margin = net*pace/100 (pace from 2025-26).

## Rosters (rosters_2026_raw.txt -> build_rosters.py)
RealGM depth charts 2026-09-18 for 25 teams; POR/SAC/SAS/UTA/WAS = 2025-26 roster + documented moves (Hoops Rumors).
Key moves: Giannis->MIA; LeBron, Jaylen Brown->PHI; Kawhi->TOR (Ingram, Dick->LAC); Ja Morant->POR; KD->HOU;
LaMelo->MIN; Anthony Davis, Trae, Middleton, Ayton->WAS. Draft: Dybantsa WAS 1, Peterson UTA 2, Boozer MEM 3.
Draft data: github.com/gtkacz/nba-real-draft-history (git clone works; GitHub API is rate-limited).

## Current projected wins (lineup 0.6; after 2026-09-18 roster fixes)
Model: OKC 58.3, CLE 53.8, BOS 53.0, SAS 52.3, HOU 51.4, DET 48.8, MIN 48.6, ORL 48.6, LAL 47.6, TOR 47.3, POR 45.8, DEN 45.8,
NYK 45.5, PHX 44.7, MIA 41.4, ATL 40.3, GSW 40.0, CHA 39.7, PHI 36.9, IND 36.2, NOP 33.9, UTA 33.9, DAL 32.4, MIL 31.8, CHI 31.3,
MEM 31.1, SAC 30.5, LAC 29.2, WAS 25.3, BKN 24.9.
Roster fixes: "Nicolas Claxton" was unmatched (treated as -4.7 newcomer starter at CHI) -> alias added; Jalen Duren (unsigned
RFA, Pistons expected to keep him) was on no roster -> added to DET. Khaman Maluach absent from PHX depth chart (in FA pool).
Model is bearish vs consensus on NYK (champs), PHI; user reviewed MIL (roughly agrees after minutes/rookie fixes).

## Dashboard v2  [2026-09-18]
Published (same link, updated in place): https://claude.ai/artifact/CfaYAX9qHUugsMpMjvLVLf  (declares db capability)
Build: python code/build_dashboard.py -> dashboard.html (template code/dash_v2.html; old v1 template dash_template.html kept).
Verify: node code/test_dashboard.js (engine vs project.py: max 0.001 wins; sim vs simulate.py within MC noise; ~0.45 s/10k sims).
- Tabs: Standings (per conference: win strip w/ forecast dot, model ring, line tick, p10-p90 band; forecast, model, line,
  model-vs-market, P(over), playoffs, title, net; click team -> rotation with games/min/rating/contribution + Edit),
  Players (2026-27 projected ratings incl. slope+aging, component bar, FA pool via team filter/search), Head to head
  (any matchup + one team vs all 29, from blend margins; model-only prob shown).
- All recomputed in-page: sliders (lineup default 0.6, range 0-1.5), per-player overrides {t team|FA, g, mpg, r rating},
  minutes re-allocation (locked overrides; others trimmed/scaled to 19,680 as in project.py), team net zero-summed,
  blend (blend.json weights incl. roster age), 10k-season sim with fixed seed (common random numbers), backtest box.
- One fixed NBA-formula schedule (seed 2026) embedded; replace with dated schedule when loaded.
- Overrides+sliders saved to db doc settings/current {sliders, players:{key:{t,g,mpg,r}}, updated} and localStorage.
  Keys: "p<player_id>" or "n:<name>|<team>" for no-history players. READ THIS in the daily pipeline via
  Artifact read_db (collection "settings", doc_id "current") and apply the user's overrides.
- data/players_fa_2026.csv = free-agent pool (rated, >=100 min in 2025-26, no roster). data/pace_2025.json caches pace.

## Season simulation (simulate.py -> sim_2026.csv, sim_wins_2026.npy)  [added 2026-09-18]
- Schedule = NBA formula (div x4; conf non-div 6x4 + 4x3; other conf x2), random 4-regular 3-game graph per conference,
  Euler-oriented so every team is exactly 41 home. Re-drawn 40 times across 20k sims. Exact dated schedule NOT loaded:
  fixturedownload.com and NBA cdn are blocked from sandbox (fixturedownload robots-blocks web_fetch CSV; its HTML preview
  page is fetchable). Load the dated schedule inside the daily-update pipeline instead.
- Per sim: true margin ~ N(proj margin, TAU=2.2) (assumed, not fitted: implies ~7.5-win preseason RMSE incl. game luck),
  game p = Phi((m_h - m_a + 2.6)/12.8). Play-in 7-10, bo7 2-2-1-1-1 with home court to better record; ties random.
- Outputs: mean wins, p10/p90, playoffs, play-in, top6, seed1, r2, conf finals, finals, title, P(over line), edge.
- Market: data/win_totals_2026.csv (BetMGM ~Sep 11, all 30). Vig-free implied mean = line + 7*Phi^-1(p_over).
- Findings: model vs market corr 0.87, MAE 4.1 wins; model is compressed (model = 0.73*market + 10.7; sd 8.8 vs 10.4).
  Biggest gaps: PHI -12.6, SAC +9.6, SAS -9.2, WAS -7.6, IND -7.4, MIL +6.6, CLE +6.4, NYK -5.4, DET -5.5.
  Haliburton projected 43 games (injury-year rule) = likely override candidate. Sim needs pyarrow (pip install).
- Open question: trust in model vs market can't be settled without a true PRESEASON backtest (project each past
  October from prior data + opening rosters, compare with historical lines). Existing backtest uses actual minutes.

## Preseason backtest (preseason_backtest.py, score_preseason.py -> preseason_backtest.parquet)  [2026-09-18]
- Each October 2014-15..2025-26 projected from prior data only: W from 1996-2013, ratings window S-3..S-1, minutes and
  rookie-minute models refit on seasons < S, opening rosters = first team played for, if within that team's first
  15 games or same team as prior season. Vegas lines: data/hist_lines.csv (SportsOddsHistory/Covers, 360 team-seasons).
- Known small leaks (flatter the model): calib/aging from full sample; players who missed an ENTIRE season absent.
- Results excl. 2019-20 (330 team-seasons), RMSE in wins: Vegas 8.44, model 8.89, half-regressed prev record 9.92,
  flat 41 12.25. Model slope 0.99 (well calibrated: its narrower spread is CORRECT for its info); Vegas slope 0.88.
- Blend: actual-41 = 0.30*(model-41) + 0.66*(Vegas-41); leave-one-season-out RMSE 8.32 (beats Vegas 8.44); model
  coefficient t=2.8, so the model carries real information Vegas lacks. When they differ by 6+ wins (70 cases), actual
  landed on the model's side of the line 63% but Vegas was closer 63% -> model points the right way, overshoots.
- Projection uncertainty fit: model TAU 2.77 pts, blend 2.53 (was assumed 2.2). Now in simulate.py.
- Failure mode: tanking/sell-offs (TOR, WAS 2023; SAC, MEM 2025) and young-core leaps (OKC 2023; DET, CHA 2025).
  SAC 2026 (+9.7 vs market) fits the first pattern.
- Stints 2011-2025 were rebuilt this session with rebuild.sh (download+parse+delete, ~25 s/season; run in foreground,
  background jobs get killed). parse_pbp.shooter_pcts now falls back to player_seasons totals (no double pass).
- simulate.py now outputs model-only columns (edge vs market, P(over)) and blend_* columns (best forecast).

## Team direction signal (direction_features.py, test_direction.py, minutes_age_fix.py)  [2026-09-18]
- Tested (LOSO, no Vegas input): prior record, roster age, youth share/quality, vet quality, 'seller' (good vets on
  projected-bad team, several forms), continuity, rookie share. Seller signal FAILED (negative OOS gain): the TOR23/SAC25
  collapses were in-season decisions on teams the model rated fine -> handle via daily updates, not preseason.
- Roster age was the only real signal. Mechanism 1 (fixed): minutes. Actual/projected minutes by age: <=22 1.105,
  23-25 1.058, 26-28 1.017, 29-31 0.94, 32-34 0.911, 35+ 1.00 -> now multiplied into min_model in project.py
  (blend.json:min_age_mult). Team RMSE 8.89 -> 8.86.
- Mechanism 2: remaining age effect is mostly a VEGAS bias (oldest quintile Vegas -2.4 wins, youngest +1.6, monotone;
  model residuals small/non-monotone; concentrated 2020-25). Not added to model; added to blend only:
  actual-41 = 0.387*(model-41) + 0.702*(Vegas-41) - 1.07*(min-weighted roster age - league avg).
  LOSO RMSE: Vegas 8.44 | model 8.86 | model+Vegas 8.30 | +roster age 8.20. TAU blend 2.44 (blend.json).
- Blend can exceed both inputs (OKC 63.1: young + both agree). Oldest 2026 rosters: NYK +3.3 yrs, MIA +3.0, GSW +2.5.
- project.py no longer loads missing minutes_coef.npy; needs stints_2025 for pace (rebuild.sh 2025 2025).

## Contract-year test (contract_year.py)  [2026-09-19]  -> NOT added to model
- Data: data/contracts/ (sumitrodatta/contract-prediction-2023 FA classes 2012-2023 w/ deal lengths; gabriel1200/site_Data
  salaries 1991-2024 + current options/salaries incl. 2026-27 UFA/RFA flags in nba_options.csv).
- Ex-ante design: 'scheduled' expiry = previous signing ends that summer, or RFA; opt-outs/waivers ('early') chosen after
  the season, reported separately. Outcome vs prior 3 seasons + age + season FE, player-clustered SE, 2011-12..2022-23.
- Box rating: scheduled CY +0.11 pts/100 (t 1.4); vets +0.03; vet starters +0.13; rookie-scale RFA year +0.26 (t 1.7).
  On/off: CY -0.55 (t -2.0), vet starters -1.4 (t -2.3). 'early' FAs -0.29 box (waiver selection). Post-deal: box -0.08,
  on/off -0.75 (t -2.8, confounded by team changes). Conclusion: no usable contract-year boost.
- Possible real signal is MINUTES: expiring non-starter vets play ~15% fewer minutes than a crude expectation (t -5.7)
  (trades, buyouts, teams favouring controlled players). Untested against our actual minutes model: needs pre_P_{S}
  (player-level preseason projections, regenerate via preseason_backtest.py after rebuilding stints 2011-2025).
- Also scripted: code/bet_analysis.py (hypothetical win-total staking; see chat 2026-09-19).

## Hypothetical betting check (bet_analysis.py)  [2026-09-19]
Stake $1000 across 30 win totals proportional to |P(over)-0.5| (blend), actual BetMGM prices: model-implied +$242 (sim
p10 -$5, p90 +$477). LOSO replay 2014-25 at -110: +$217/season, 60% hit, 2 losing seasons of 11 (model+mkt only: +$123,
54.5%; the roster-age term adds the rest). Treat as overstated (backtest leaks; age signal found in-sample; 2026 bets
correlate -0.7 with roster age = one correlated bet). User does not intend to bet; paper-track only.

## Research roadmap (agreed with user 2026-09-19; items 1-2 done, see 'Roadmap results'; continue at 3)
Context: public models JIMBO is compared against: EPM, DARKO, LEBRON, ESPN BPI/RPM, (retired) FiveThirtyEight RAPTOR/Elo.
Priority order:
1. MARKET-BIAS HUNT (first; fast, data in hand). Data: preseason_backtest.parquet (model_wpct_agefix, mkt_wpct, wpct),
   direction_features.parquet (mw_age etc.), hist_lines.csv, games_all.parquet. 330 team-seasons excl. 2019-20.
   Test each as an added term in the LOSO blend (same method as test_direction.py; report LOSO RMSE gain, coef, t):
   a. big-market / "public" teams (LAL, NYK, GSW, BOS, CHI; maybe national-TV counts) overpriced?
   b. prior-year record vs prior-year point differential (close-game luck) -> market overreaction?
   c. headline offseason moves (large incoming-minutes share / star acquisitions) overpriced?
   d. recent playoff success (conf finals+ last year) overpriced?
   Beware multiple testing: ~4-6 tests, require t>2.5 and consistent sign across halves (2014-19 vs 2020-25).
2. RETURN FROM MAJOR INJURY. Players who missed (nearly) a full season: find in player_seasons (gp==0 or tiny after
   regular seasons); compare next-season rating vs projection. Relevant 2026-27: several stars tore Achilles in 2025
   incl. Haliburton (currently 43 projected games). Needs player-level preseason projections pre_P_{y} (not in zip:
   rebuild stints 2011-2025 via rebuild.sh, ~25 s/season, foreground; rerun preseason_backtest.py). Same rebuild
   also unlocks the pending expiring-contract MINUTES test (contract_year.py finding, t -5.7 vs crude model).
3. YOUNG-PLAYER DEVELOPMENT term beyond age bins: last-season improvement, age at draft, pick, late-season usage/minutes
   rise. Target failure mode: young-core leaps (OKC 2023, DET/CHA 2025).
4. FT% AS 3P% PRIOR: shrink 3P% toward prediction from FT% (+volume) instead of league average (features.py, k=250).
5. DIMINISHING RETURNS / FIT on stints: lineup-level nonlinearity (rebounding overlap, total spacing, usage overlap,
   two-big lineups). Motivation: rebounding looks over-weighted when additive (0.5x best in backtest).
6. TEAM-SPECIFIC UNCERTAINTY (TAU by team: star dependence x injury proneness, depth) -> playoff odds, P(over).
Pipeline-dependent signals (need NBA.com access from GitHub runners): hustle/tracking stats 2015-16+ for perimeter
defense; rest/travel/altitude for game odds; same-day injuries/lineups; better rookie priors from college stats.

## Roadmap results  [2026-09-19, session 3]
1. MARKET-BIAS HUNT (code/market_bias.py -> data/market_bias_features.parquet): ALL REJECTED. Base = LOSO blend
   model(agefix)+Vegas+roster age, RMSE 8.200; season-clustered t. Every term made LOSO worse:
   big-market (LAL/NYK/GSW/BOS/CHI) 8.217, coef +1.2 t 1.0, sign flips by half | prior-year close-game luck (wins vs
   pyth, sd 3.3) 8.211, coef 0.0 | top-30 star arrivals 8.232 / departures 8.235, t<0.6 | roster turnover 8.239 |
   prior-year conf finalist 8.208, t -1.1. Note: Vegas alone DOES overrate last year's conf finalists by ~2.6 wins,
   but JIMBO already captures it (no gain once model is in the blend). Market is efficient on these public narratives.
2. RETURN FROM MISSED SEASON (code/injury_return.py -> data/injury_return.parquet). Stints 2011-2025 rebuilt and
   pre_P_{S} regenerated (pre_T identical to before; pre_P now in zip). Returners = regular in N-1 (20+ mpg, 30+ gp),
   <20% availability in N; 69 cases 2014-25 (excl. 2019-20), 15 projected rating >2.
   - Rating: actual next-season box -0.35 below projection vs comparable players (t -2.7; 2014-18 -0.57, 2020-25 -0.22,
     same sign). Driven by role players (-0.42, t -2.8); stars -0.16 (ns). Passes the bar narrowly. NOT yet applied;
     candidate: -0.35 rating for returners (team effect ~0.3 wins per starter). Awaiting user decision.
   - Minutes: returners played +18% vs model (stars +33%), t 2.1, but 2014-18 only (+394 min, t 3.1); 2020-25 +44 (t 0.5).
     Fails the bar -> Haliburton-type 40-45 game projections are NOT clearly too low under current practice.
     Caveat: sample = players who did return (full-season absentees not on opening rosters).
   - 2026 returners: Tatum, Irving, VanVleet, Haliburton, Lillard, Kessler, Edey, Lively, D. Murray, Trae (15 gp), etc.
   - APPLIED 2026-09-19 (user OK): data/returner.json (adj -0.35 post-slope; regular in N-1 = 20+ mpg & 30+ gp, <20%
     availability in N). project.py column 'returner' + note; preseason_backtest.py same rule; existing pre_P/pre_T
     patched via code/apply_returner_bt.py (idempotent; equals full rerun). Dashboard shows 'Return from missed season'
     part (field rt). 17 rostered 2026 returners; team wins moved <=0.3 (DAL -0.3, BOS/LAC -0.2).
     Model RMSE 8.737->8.735, blend LOSO 8.142->8.141 (in-sample coef; tiny). Blend refit (later constrained, see below). sim + dashboard rebuilt, tests pass, republished.
3. EXPIRING-CONTRACT MINUTES (code/contract_minutes.py): REJECTED. Vs JIMBO's actual minutes model the crude -15% vanishes;
   bench vets in contract years play +76 min MORE than projected (t 3.1) but only in 2014-18 (2020-22 +30, t 0.8);
   starters -3. The minutes model already absorbs it.
4. ROOKIE + YOUNG-PLAYER PRIOR (code/young_prior.py builds game-level design -> data/young_prior_design.parquet;
   code/young_prior_fit.py). Same framework as aging.py (13.2k games 2015-25, actual minutes, season-clustered t).
   - Rookie value continuous: val = -2.90 - 1.60*(ln pick - 2) + 0.82*(draft age - 20)  (data/rookie_value.json).
     LOSO game RMSE 13.4582 -> 13.4462; 2021-25 holdout 14.055 -> 14.011; log-pick t -3.4, draft-age t 2.0, both same
     sign/size in 2015-19 and 2020-25. Older draftees are better AS ROOKIES (readiness). Height/intl add nothing.
     Slope and aging bins unchanged (0.971). Team-level preseason RMSE unchanged (8.890 -> 8.892): rookie minutes are small.
     APPLIED in project.py and preseason_backtest.py (pre_T/pre_P regenerated). 2026 wins moved: MEM -2.6 (Boozer, age 18,
     -1.43 -> -3.00), GSW +2.6 (Lendeborg, 23, pick 11: -4.30 -> -1.04), OKC +1.3, BKN/WAS +1.1; rest within +-0.8.
     sim_2026 + dashboard NOT yet regenerated with this change (pending user OK).
   - Young rated players (<=24): 2nd/3rd-year bumps ~0, pick & draft age ~0, box trend N-1->N +0.46 (t 2.2, sign flips
     by half), late-season mpg rise ~0. REJECTED: the age bins + 3-yr decayed window already capture development.
   - Real upgrade path: college/international stats for rookies (e.g. sportsdataverse/hoopR-mbb-data on GitHub; not yet
     verified reachable from sandbox - API rate-limited, git got 503) or via the GitHub-runner pipeline.
   Also fixed: project.py pace json float32 crash.
5. NCAA ROOKIE PRIOR  [APPLIED; dashboard republished same link]
   Data: sportsdataverse/hoopR-mbb-data ESPN player box scores 2006-2026 (github raw works:
   github.com/sportsdataverse/hoopR-mbb-data/raw/main/mbb/player_box/parquet/player_box_{Y}.parquet; Y = spring year),
   saved data/ncaa/. code/ncaa_build.py -> ncaa_player_seasons.parquet (per-100 on-court poss rates, ridge team SRS/SOS);
   code/ncaa_match.py -> ncaa_draft_match.parquet (527/528 college first-rounders 2006-26 matched by name+season).
   code/ncaa_model.py (stage A): ridge (lam 50000) of NBA rookie-season BOX rating on college rates + pick/age/height,
   leave-one-class-out -> ncaa_scores.parquet (cscore). Rookie box wRMSE: pick+age+height 1.115 -> +college 0.987.
   Stage B (game-level, young_prior.py/ncaa_gametest.py): val = -3.07 + 3.74*clip(cscore + 0.984, +-1) (t 8.4);
   non-college first-rounders keep pick+age formula. Game LOSO: old bins 13.4582, pick+age 13.4462, college 13.4252,
   capped 13.4197 (best). Uncapped over-rates rebounding/blocking bigs (Wiseman). Consistent both halves.
   Team preseason backtest: raw RMSE 8.890 -> 8.807, blend(model+Vegas) LOSO 8.318 -> 8.288 (stage-B scale in-sample).
   2026: capped at +0.67: Boozer, C. Wilson, Mara, Graves, Ejiofor, Reed; low: Peterson -4.2, Stirtz -4.4, Karaban -4.7,
   Dybantsa -3.1. Wins vs old bins: CHI +5.6, GSW +2.3, CHA +2.1, OKC/TOR +1.9, UTA -4.0, WAS -2.7.
   blend.json weights NOT refit (fit on old model); refit when convenient (score_preseason + test_direction method).
   v2 (after user asked why Dybantsa low): stage A had wrong-sign pick (+) and SOS (-) coefs from collinearity; dropped
   both (base = age, height). Game LOSO 13.4191, val = -3.10 + 3.78*clip(cscore - col_mean, +-1). Pick adds nothing at
   game level once college known (t -1.1). Dybantsa -2.59 (cscore -0.86): huge usage/scoring (41 pts/100, 39% usg) is
   barely rewarded because the target is ROOKIE-YEAR impact; few blocks/steals/OREB, high TOV. Same pattern rated
   Edwards, Cade, Tatum, B. Miller, Wiggins, Ingram low - all had modest rookie years. It is a rookie-year prior, not a
   career/star projection. Wins now: CHI 36.9, UTA 30.5, WAS 23.2, OKC 60.2 (dashboard republished).
   Ideas: second-round/undrafted college rookies; college prior for 2nd/3rd-year players (blend with thin NBA sample).
6. DIMINISHING RETURNS / LINEUP FIT (code/diminishing.py -> dr_rows.parquet; diminishing_fit.py; dr_team_nl.parquet)
   Stints 2012-25 (~800k side-rows), player component values from PRIOR seasons (no mechanical overlap), lineup sums per
   component + squared terms + interactions, season FE, poss-weighted, robust t. Findings (consistent both halves):
   - Def rebounding: diminishing (sq t 4.2). +2 sd rebounding lineup gets ~2/3 of additive value; -2 sd loses ~35% extra.
   - Self-creation: strongly diminishing (t -3.7, stronger 2019-25): +2 sd lineup gets ~40% of additive value
     ("too many ball-dominant creators"). Foul drawing, playmaking mildly diminishing.
   - 3PT: linear (no spacing synergy beyond additive). Interactions 3PTxrim, 3PTxplaymaking, OREBxDREB: nil.
   - Off rebounding: slightly INCREASING returns (commit to crashing). Big count (6'10"+): no multi-big penalty net.
   - Team-season size (actual lineups): sd 0.27 pts/100 (~0.7 wins), range -1.4 (MIL 2023, CLE 2014, PHI 2022 =
     star-creator overlap) to +0.8. Explains preseason-model misses only r 0.05 -> NOT added to projections.
   Possible later use: lineup/rotation tool; partially explains why rebounding looked over-weighted (0.5x best).
7. BLEND REFIT (2026-09-19, after NCAA prior v2): reran preseason_backtest.py -> score_preseason.py -> minutes_age_fix.py,
   refit on 330 team-seasons (method reproduces old 0.387/0.702/-1.07 exactly on old data). New blend.json:
   w_model 0.436, w_vegas 0.669, w_age -1.109, tau_blend 2.425, tau_model 2.73. RMSE: model 8.86->8.77, Vegas 8.44,
   blend LOSO 8.200->8.155. Forecasts moved <=0.5 wins. sim + dashboard rebuilt, tests pass, republished.
8. 3PT SHOOTING PRIOR (roadmap 4)  [APPLIED; dashboard republished]
   code/shooting_prior.py: next-season 3P% (weighted by 3PA) error, pct pts: season shrunk to league k250 3.885 ->
   CAREER-to-date shrunk to league k400 3.750 -> + career FT% (mostly for low-volume shooters) 3.706 (LOSO).
   Naive "shrink toward FT-based prior" (prior fit on high-volume careers) was WORSE - selection bias; use regression:
   rel3 = 0.003 + 0.868*career_k400 + 0.011*(FT-.75) + 0.183*(FT-.75)*400/(career3PA+400).
   code/shooting.py: new o_3eff = 1.5 x W3 x window 3PA rate x 3 x rel3 x poss/(poss+300); apply() replaces f:o_3eff and
   adjusts three_pt/off/total. Game backtest (shooting_bt.py): production 13.5026, career+FT 1x 13.4966, 1.5x 13.4939,
   2x 13.4930 (better in both halves; chose 1.5x conservative). 3PT eff was under-weighted (free coef 1.7-2.0x).
   Hooked into project.py, build_dashboard.py, preseason_backtest.py; backtest_components f:o_3eff replaced (old copy not
   kept in zip). Team preseason RMSE model 8.766 -> 8.737, blend LOSO 8.155 -> 8.142; blend refit (code/refit_blend.py):
   w_model 0.451, w_vegas 0.658, w_age -1.133, tau_blend 2.421, tau_model 2.72. Dashboard backtest 13.495 -> 13.487.
   Biggest risers: Curry +0.9, Klay +0.7, G. Allen, Hauser, MPJ, Kennard +0.4; fallers: Drummond, K. Murray, Dunn, Gordon -0.2.
   Team wins moved <=0.9 (BOS +0.9, CHA +0.8, GSW +0.6; DEN -0.6). Calibration (slope/aging/rookie) not refit (tiny change).
   Not done: using talent 3P% in the stint luck adjustment x3 (parse-level change).
9. TEAM-SPECIFIC UNCERTAINTY (roadmap 6; code/team_tau.py -> team_tau.parquet): REJECTED, single tau kept.
   Heteroskedastic MLE on LOSO blend residuals (330 team-seasons): var = luck + exp(a + b.x), held-out log-lik vs constant.
   No feature helped: star value, star share of team value, injury-variance (sum v^2 a(1-a)), thin-data minute share,
   roster age, projected-bad (tank) flag (+0.44 LL, t 1.1 - closest), |projection|. Residual sd by star-value quartile
   8.5/8.1/7.8/8.1 wins (flat), although an independent-injury simulation would imply 2.9-8.4 wins of injury sd across
   teams -> mechanistic star-injury sims would OVERSTATE differences. Star injuries are for in-season updates.
ROADMAP 1-6 COMPLETE. Active work: DEFENSE ROADMAP (#1 applied, #2 rejected; continue at #3/#5, see section below). Also pending: GitHub nightly
pipeline + phone app (see below), before opener ~2026-10-20.


## In-season update backtest (code/inseason_bt.py, inseason_report.py)  [2026-09-19, session 4]  -> DESIGN CHOSEN
Access test from GitHub runners (Actions, 2026-09-19): NBA cdn liveData/staticData (pbp, box, scheduleLeagueV2) work
ONLY with curl_cffi impersonate="chrome" (plain/curl get Akamai 403); S3 mirror
nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/{playbyplay,boxscore}/..._{gameid}.json works plainly
(backup). stats.nba.com times out (cloud-IP ban; lose hustle/shotdetail). ESPN site.api (injuries, scoreboard w/ odds)
needs Chrome imitation; site.web.api.espn.com works plainly. official.nba.com injury page + report PDFs work plainly.
fixturedownload JSON feed works. shufinskiy mirror updates only ~monthly -> backfill only.
parse_pbp.norm_cdn already reads cdn action fields (mirror CSV = flattened cdn JSON actions); zones must come from
cdn 'area' field instead of shotdetail (to verify on real games).
Backtest: seasons 2014-25, checkpoints at league-median 5..70 games; update pre_P ratings with games before the
checkpoint, predict all later games with ACTUAL minutes (rating quality only). Chosen method (M4):
  1) box update: current-season features (features.py recipe on season-to-date) added to the player's decayed
     N-2..N window at weight 1.0 per possession; rating += SLOPE*(new_box - hist_box). No-history players: prior
     counts as 2000 poss.
  2) RAPM toward prior: ridge on season-to-date stint NET ratings (luck-adjusted, garbage dropped), deltas shrunk to 0,
     LAM=6000 (lower lam hurts early, higher hurts late). Within-season recency decay: no gain -> none.
  3) team offsets: ridge on past-game residuals, lam_t=80 games (helps most 2020-25).
Game RMSE gain vs preseason-only (actual minutes): -0.05 @5 games, -0.10 @10, -0.18 @20, -0.25 @30, -0.30 @41, -0.38 @55, -0.49 @70.
Team rest-of-season margin RMSE @41: 3.47 -> 2.83. Same sign/size both halves. Box update alone is weak; RAPM delta
does most of the work.
IN-SEASON MINUTES (code/inseason_minutes.py, inseason_minutes_report.py; data/ism_{S}.pkl results, ism_r_{S}.pkl cached
M4 ratings per checkpoint)  [2026-09-19, session 5] -> CHOSEN, saved in data/inseason.json:
  mpg = w*(0.5*last-10 + 0.5*season-to-date) + (1-w)*preseason mpg, w = n/(n+5) (n = team games since joining current
  team, DNPs = 0); rescale team to 240/game. Injured per report -> 0; fallback w/o report: absent 5+ straight -> x0.5.
  Team rest-of-season margin RMSE (avg over cp 5..70, excl 2019-20): oracle actual minutes 3.03 | chosen ~3.67 |
  season-to-date only 3.72 | preseason minutes only 3.93. Optimum flat for k 3-10, recency 0.5-0.7. Preseason minutes
  prior is worth only ~5 team games. Absence-streak rule gains ~0.01 (mixed by half) -> real injury reports matter.
  Game RMSE: oracle 13.18 vs projected minutes 13.70 -> for TONIGHT's odds, same-day injury/lineup info can recover
  much of the 0.5 gap. Rest-of-season sims must add minutes uncertainty (future injuries/trades): projected-minutes
  team error ~3.4-3.6 pts/g vs 2.8-3.2 oracle; fit in-season TAU from these residuals when building the sim.
Data: stints/player_games/games 2012-2025 rebuilt (not in zip; rebuild.sh 2012 2025, ~15 s/season, foreground, mkdir raw).

## NEXT (as of session 5): in-season design complete (ratings M4 + minutes). Remaining before opener ~2026-10-20:
GitHub pipeline + phone app (below); fit in-season rest-of-season TAU; optional ideas: college prior for 2nd-round/
undrafted rookies and 2nd/3rd-year players; talent 3P% in stint luck adjustment. Maluach still missing from PHX roster.

## Blend constrained + forecast breakdown  [2026-09-19, session 5]
- User asked why OKC forecast (65.1) exceeded model (60.4) and line (62.5): free blend weights summed to 1.11 (se 0.065,
  not significant) + youth term. Switched to model+Vegas weights summing to 1 (refit_blend.py, blend.json constrained=true):
  w_model 0.351, w_vegas 0.649, w_age -0.798, tau_blend 2.429. LOSO 8.161 (free 8.141). OKC now 62.6. A team can still
  land outside both inputs via the roster-age term (e.g. BOS 50.9, old roster -1.4). sim + dashboard rebuilt.
- Standings columns (user's choice, waterfall tried and removed): #, Team, win strip, Forecast, Model, Line, Age adj
  (roster-age term incl. uniform re-centering ~-0.2 so league = 1,230), Model vs line (model - line), Over line,
  Playoffs, Title, Net rtg. Legend states the blend formula. Weights fixed (no slider). Fixed mobile page overflow
  (main grid minmax(0,1fr)). Visual check via playwright (pip playwright available; chromium works).

## Player view  [2026-09-20, session 5]
Tap a player (team rotation in Standings, or any row in Players tab) -> full-page player view (#pv; Back button,
browser back via history.pushState, Esc; sidebar hidden on narrow screens). Shows: rating/off/def/minutes/adds-to-team
cards; 'How the 2026-27 rating is built' (box 3-season window + lineup x weight + age bin + returner + re-centering =
model rating; override shown); skills sorted by size; rookies: college profile (per-100 rates, college score percentile,
baseline formula from rookie_value.json) or pick+age note; career chart by season (SVG, stacked components / offense vs
defense / single component selector; dashed bar = 2026-27 projection incl. lineup+age; <250 min faded) + season table.
History data (build_dashboard.py 'hist'): features.parquet per season x 0.5*W_prod (= fit_window W exactly), centered on
each season's possession-weighted league mean, x1000 ints; JS applies slider multipliers and slope. Season 3PT uses raw
single-season efficiency (projection uses shooting.py talent version). 'rook' = 2026 first-round college profiles.
Stacked 'All components' and 'Offense vs defense' views include each past season's OWN single-season lineup effect
(dot = season total = box + lineup); table columns Box / Lineup / Total; projection row uses the 3-season averages.
Chart modes also include 'Lineup effect' (bars = single-season lineup residual, gold line = 3-season weighted window the
next projection used; both x lineup slider x slope) and 'Age adjustment' (aging.csv bin at that season's age x slope;
league curve, not player-measured). Data: code/lineup_history.py -> data/lineup_hist.parquet (per season 1996-2025:
one_o/one_d single-season fit, win_o/win_d = production window; re-centered like production.py; window 2025 reproduces
ratings_2025 exactly; 3 s/season). Needs stints 1996-2025 (all rebuilt session 5; not in zip).
Note: the 3-season joint fit is not a simple average of single seasons (can exceed all three, e.g. LeBron 2016-17).
LeBron 2026-27 (user asked): box window +1.85, lineup -0.71 (single seasons x0.6: +0.75, -0.59, -1.04), age 36+ -1.22,
rating ~0.0; drop vs 2025-26 projection mostly lineup swing (~-1.4). Offered (not run): test whether sharp lineup
declines in aging stars persist or rebound.
AGE-CURVE SLIDER (S.age, default 1.0, range 0-2, sidebar 'Age curve'): scales every aging bin (rawRating ag*S.age,
also the player-view age chart). Engine treats missing S.age as 1 (test_dashboard.js compatibility).
Payload 2.0 MB. Player view re-renders on slider/override changes (renderAll -> renderPV). Tested via playwright.

## Age-curve validation (code/age_check.py, code/age_se.py)  [2026-09-20]  -> under-21 penalty NOT supported
User challenged the <=21 bin (-0.57) using Dylan Harper (SAS, age 20.9; box +0.41, lineup +1.29, age -0.55 -> +1.27).
- age_se.py = aging.py's game-level design + season-block bootstrap SEs (13,206 games): <=21 -0.52 (se 0.53, t -1.0),
  22-23 +0.13 (0.21), 24-25 +0.21 (0.18), 28-29 -0.12, 30-31 -0.04 (0.41), 32-33 -0.27, 34-35 -0.56 (0.35),
  36+ -1.21 (0.36, t -3.4). Only 36+ is well identified; <=21 is collinear with the rookie/unrated minute shares.
- age_check.py = player-level: window rating (decayed box + 0.6 window lineup, x slope) vs next season's measured
  rating (box alone, or box + single-season lineup), 1996-2024, 500+ min both seasons, minutes-weighted, season SEs.
  Gap by bin: <=21 +0.91 (0.08), 22-23 +0.59, 24-25 +0.23, 26-27 +0.03, 28-29 -0.13, 30-31 -0.34, 32-33 -0.50,
  34-35 -0.46, 36+ -0.80. Monotone; model's <=21 has the WRONG SIGN, 30-33 too gentle, 36+ maybe too harsh.
  Stable to min-minute threshold (200/1000) and across halves (+0.87 / +0.94).
  Much of the young gain is mean reversion: gap +1.22 when window rating < -1, +0.37/+0.52 when > 0. Regressing
  act = a + b*window by bin: <=21 a +0.69, b 0.76 -> implied adjustment at a +1.7 window rating (Harper) about +0.3,
  vs the model's -0.57 (so Harper is docked ~0.9). Caveats: survivorship (must play next season), target is measured
  rating not wins, b attenuated by noise in the window rating.
- NOT changed in the model yet. Proposed next step: refit the age curve (player-level shape, then score at game level
  and on the preseason team backtest), or at least re-estimate <=21/22-23 with rookie-share collinearity handled.

## AGE CURVE REFIT  [2026-09-20]  -> APPLIED (replaces aging.py's bins; old files kept as data/*_old.*)
Why: the old <=21 bin was -0.57 while young players' own ratings rise. Evidence (see 'Age-curve validation'): the
game-level design cannot identify the shape (freely refit bins LOSO 13.4502 vs NO curve 13.4453; the old curve only
looks good in-sample because it was fit on the same games). So: take the SHAPE from the player-level evidence and
check it on the team-level preseason backtest.
- New curve (data/aging.csv; old = aging_old.csv): <=21 +0.88, 22-23 +0.56, 24-25 +0.20, 26-27 0, 28-29 -0.16,
  30-31 -0.37, 32-33 -0.53, 34-35 -0.49, 36+ -0.83  (= age_check.py gaps, 26-27 centred; scale 1.0).
- Scale test on the preseason backtest (code/age_eval.py TAG; preseason_backtest.py now honours env AGING_CSV,
  SLOPE_OVR, BT_TAG): model RMSE (wins) none 8.829 | old curve 8.735 | shape x0.32 8.753 | x0.5 8.719 | x1.0 8.674
  | x1.25 8.683 | x1.5 8.719. Blend LOSO falls monotonically to x1.5 (8.139) only because the blend's roster-age term
  shrinks (w_age -0.80 -> -0.46): the model now does that work itself. Chose scale 1.0 (best standalone, no tuning).
- Joint refit at game level with the new curve fixed (code/rookie_refit.py, design code/age_design2.py ->
  age_design2.parquet, which adds college-rookie/cscore/non-college/unrated minute-share columns): slope 0.970 -> 1.070.
  Pre-slope values rescaled by the same fit's old->new ratios: calib.csv unrated -4.685 -> -4.139; rookie_value.json
  col_a -3.10 -> -2.97, col_b 3.78 -> 3.53, rk/rk_lp/rk_age x0.92. College rookie mapping barely moved (good sign).
- Result: preseason model RMSE 8.735 -> 8.674, slope on actual 1.01, blend LOSO 8.161 -> 8.157; blend.json refit
  w_model 0.358, w_vegas 0.642, w_age -0.622, tau_blend 2.427. All ratings ~10% wider (slope), team wins zero-sum.
- Movers: Flagg -0.48 -> +1.02, D. Harper +1.15 -> +2.82, LeBron ~0.0 -> +0.37, Curry +2.4 -> +2.95, Wemby +9.45.
  2026 wins: OKC 61.1 model / 62.1 blend, SAS 54.0/57.2, BOS 54.2/51.2, CLE 52.8/48.5; SAC 27.4/24.1 lowest.
- Age-curve slider note in the dashboard updated. Dashboard backtest box (game RMSE) unchanged: it scores component
  weights only, not the age curve.
- Not redone under the new curve: NCAA stage-B fit (ncaa_model/ncaa_gametest), aging.py itself (superseded; keep for
  reference), non-college rookie scale (game fit says x1.16-1.26 of the formula; left at 1.0 as before).

## Split lineup weight: offense vs defense  [2026-09-20, session 6]  -> APPLIED: w_off 0.6, w_def 0.8 (dashboard republished)
Motivation: user noted defensive leaderboard is all bigs (Amen/Ausar Thompson missing). Diagnosis (ratings_2025, min>800):
perimeter_def sd 0.35 (max ~1.1) vs dreb 0.61 / rim_prot 0.49; res_d sd 1.27. Mean def: bigs +1.59, wings +0.17, guards
-0.47; ~1.4 of that gap is box prior, ~0.4 lineup residual. Ausar 3rd in perimeter_def (+1.04); Amen +0.22, res_d -0.18
(box-invisible AND collinear with HOU teammates).
Test (code/bt_split.py -> data/bt_split.parquet: res_o/res_d game diffs, same windows as bt5; code/bt_split_score.py):
- Game LOSO RMSE (11 seasons 2015-25): (0.6,0.6) 13.4947 | best (w_off 0.4, w_def 0.8) 13.4792 | (0.6,0.8) 13.4860.
  Free coefs relative to box: w_off 0.37 (t 6.0), w_def 0.78 (t 7.7), difference t 3.4; halves 2015-19 0.36/0.79,
  2020-25 0.40/0.76. Passes the bar. Box captures offense better, so offensive residual deserves less weight.
- Preseason team backtest (preseason_backtest.py now reads env LW_O/LW_D, default 0.6; scored with age_eval.py TAG):
  model RMSE 0.6/0.6 8.674 | 0.6/0.8 8.673 | 0.4/0.8 8.683 | 0.6/1.0 8.724 | 0.2/0.8 8.757; blend LOSO 8.157 | 8.145 |
  8.150 | 8.138 | 8.161. Team level can't distinguish (noise).
- Leaderboard effect (w_def 0.6->0.8): bigs 13 of top 20 either way (their res_d is big too). Anunoby 31->26, White 34->27,
  H. Jones 79->60, Camara 72->58, Holiday 81->66, Caruso 10->9; Amen 107->112 (residual ~0: needs roadmap idea 2).
APPLIED (user OK): project.py LW_O 0.6 / LW_D 0.8; preseason_backtest.py defaults LW_O 0.6 / LW_D 0.8 (env override);
pre_T/pre_P regenerated -> score_preseason -> minutes_age_fix -> refit_blend: blend.json w_model 0.365, w_vegas 0.635,
w_age -0.674, tau_blend 2.422 (backup data/blend_pre_lwsplit.json). Model RMSE 8.673, blend LOSO 8.145.
Dashboard: lineup slider split into lineup_o / lineup_d (meta.lw_o/lw_d); backtest box uses bt_split res_o/res_d
(13.487 -> 13.479); career history arrays l1o/l1d/lwo/lwd; saved settings with old 'lineup' key migrate on load
(0.6 -> new defaults, any other value -> both sliders). test_dashboard.js passes (max 0.001 wins); jsdom UI smoke test
ui_test.js. Pre-change copies: code/dash_v2_pre_lwsplit.html, build_dashboard_pre_lwsplit.py, data/*_pre_lwsplit.csv.
Team wins moved <=1.7 (OKC +1.7, BOS +1.4, SAS +1.1, TOR +1.0; UTA -1.3, NOP/PHI -1.2, MEM/IND -1.0).
NOT refit: calibration slope 1.070 (game-level scale check: 0.6/0.6 -> 0.6/0.8 lowers fitted slope ~4%, 1.0875 -> 1.045;
team-level slope on actual 1.01 -> 0.98, within noise). Other scripts still use 0.6*lineup_impact for research
(aging.py, young_prior.py, age_*.py, rookie_refit.py, shooting_bt.py) - update if those fits are rerun.
Stints/player_games 2011-2025 rebuilt this session (not in zip).

## DEFENSE ROADMAP (agreed 2026-09-20; #1 DONE, #2 REJECTED, #5 APPLIED (dreb 0.5x), next #3 (charges/fouls) or #6)  <- START HERE for defensive work
Goal: credit elite wing/guard defenders the model misses (Amen/Ausar Thompson, Dort, Daniels, H. Jones) without
breaking what works (center tilt is real, t 5.1). Same bar as always: game LOSO RMSE (backtest_components + bt_split),
then preseason team backtest (preseason_backtest.py + age_eval.py TAG); require t>2.5 and same sign 2015-19 vs 2020-25.
Realistic target: good wings from ~+1 to ~+2 pts/100, not to the top of the list.
1. DONE: split lineup weight, w_off 0.6 / w_def 0.8 (section above).
2. DONE/REJECTED (see results section below) - TEAMMATE-ADJUSTED DEFENSIVE SUB-OUTCOMES. Problem: d_opprim and d_oppxq (features.py) are raw on-court
   attributions, so a wing next to Wemby inherits his rim suppression; Amen's box line is ordinary and his res_d ~0
   because HOU credit is split five ways. Plan: for each window, stint-level ridge (same lineup_matrix/solve machinery
   as impact.fit_window, poss-weighted, season FE, home) of DEFENSIVE sub-outcomes on the 5 defenders (+5 offensive
   players as controls): opp 3PA share of FGA (za4-6), opp rim share (za1), opp FTA per poss, opp TOV per poss,
   opp OREB rate, luck-adjusted opp 3P% (fg3m - x3, per 3PA; expect weak/noisy), opp rim FG% (rim_m/za1).
   All computable from existing stint columns (no re-parse; stints not in zip: rebuild.sh 2011 2025, ~5 min).
   Tune lambda per outcome by split-half persistence (these stabilise much faster than points). Then either
   (a) replace d_opprim/d_oppxq with the adjusted versions as new DEF_FEATS and refit W (fit_weights, 1996-2013 for
   backtest, all years for prod), or (b) cheaper first pass: add them as extra game-level diff columns in the
   bt_split style and test incremental value over current ratings. Watch leakage: sub-outcome ratings must come from
   the same N-2..N window used to predict N+1. Check Amen/Ausar/Dort/Daniels/H. Jones/Camara before and after.
3. Re-parse for defensive events currently discarded (parse_pbp.py; ~25 s/season): team TOV forced while on court
   beyond own steals; fouls split shooting / non-shooting / offensive fouls drawn (charges; check v2 and cdn formats);
   blocks by zone (non-rim blocks vs rim); opponent transition possessions after live-ball TOV / DREB. Adds stint or
   player_games columns -> new features. Do alongside #2 if re-parsing anyway.
4. Height- or position-interacted defensive weights (stl/blk/pf value differs for guards vs bigs). Game backtest test.
5. Retest def_rebounding weight (0.5x looked best earlier, n.s.) under current model; dreb inflates bigs' defense.
   Related: diminishing returns on DREB found in diminishing.py.
6. Display only: Players tab toggle for defensive value vs positional average, and split shown defense into
   rebounding vs shot prevention. Independent of modelling; can be done any time.
7. Later (needs GitHub-runner pipeline, stats.nba.com): hustle stats (deflections, contests, charges) 2016+,
   matchup data 2017+, tracking rim DFG%.
Diagnostic numbers for reference (ratings_2025, min>800): perimeter_def sd 0.35 vs dreb 0.61, rim 0.49, res_d 1.27;
mean def bigs +1.59 / wings +0.17 / guards -0.47 (~1.4 of the gap from box prior). Ausar perimeter_def +1.04 (3rd);
Amen +0.22, res_d -0.18.

## DEFENSE #2 RESULTS: teammate-adjusted defensive sub-outcomes  [2026-09-20, session 7]  -> REJECTED (nothing applied)
Code: code/defsub.py (stint-side rows, ridge of each sub-outcome on 5 defenders + 5 offensive controls, season FE, home,
decay .3/.6/1), defsub_tune.py (lambda by split-half CV, windows 2016-18 and 2022-24), defsub_bt.py / defsub_bt2.py
(windows N=2014..2024 -> actual-minute game diffs in N+1, files data/defsub_{N}, defsub2_{N} (+ _rat_ ratings)),
defsub_score.py / defsub_score2.py (LOSO over 11 seasons, season-clustered t, halves 2015-19 / 2020-25),
defsub_names.py -> data/defsub_2025_players.parquet (window 2023-25 player values + ranks, 800+ min).
Outcomes: opp 3PA, rim FGA, mid FGA, FTA, TOV per 100 poss; opp OREB%; opp 3P% minus shooter-expected (x3); opp rim FG%;
opp luck-adj pts (= plain RAPM-D reference). CV lambda (joint) ~500-1000 (3P% luck: no signal at any lambda).
Defender-only lambdas 4k/16k/64k also tried (offense controls at CV lambda).
Game LOSO vs current best (box + 0.6 res_o + 0.8 res_d) = 13.4860:
  best single additions: opp 3PA 13.4840-13.4847 (t -1.9 to -2.1; 2015-19 t -2.4, 2020-25 t -0.5 to -1.3) | forced TOV
  13.4862 (t 1.4-1.5) | OREB% 13.4861 (t 1.5) | FTA 13.4860 (t 1.5 at 64k, 2020-25 ~0) | rim FGA, mid FGA, rim FG%, 3P% luck,
  plain RAPM-D: worse. All sub-outcomes together 13.4941-13.5000 (worse). None meets t>2.5 + same sign both halves.
  REPLACING raw d_opprim/d_oppxq by adjusted versions: dropping the raw ones costs a lot (13.4860 -> 13.5429) and the
  adjusted ones recover only part (13.504-13.514). The raw on-court opp shot-profile features carry real predictive
  information (team scheme / teammates persist for most players) that teammate adjustment removes -> keep them.
Face validity is fine (forced-TOV leaders 2023-25: Caruso, J. Green, C. Wallace, D. Daniels, Dunn, Suggs) but that
information is already in d_stl. Named wings (window 2023-25, rank of 327 with 800+ min, model def rank first):
Ausar def 29 / adj pts-allowed 21; Daniels 43/23; H. Jones 56/31; Camara 57/32; Amen 130/121; Dort 227/187 (rim FG% 4).
Amen is not hidden by HOU teammates: every adjusted outcome is ordinary for him. Dort's poor res_d is also the adjusted view.
Takeaway: the box + lineup model is not systematically missing these wings once teammates are controlled; remaining
upside is data we don't have (tracking/hustle, roadmap #7) or new events (#3: charges/offensive fouls drawn, non-rim
blocks; forced TOV beyond steals now looks redundant with d_stl). Suggested order: #5 (quick dreb weight retest), #3
(charges/fouls only), #6 display.

## DEFENSE #5 RESULTS: def_rebounding weight retest  [2026-09-20, session 7]  -> APPLIED 0.5x (user OK; see 'Applied' at end of section)
Code: code/dreb_retest.py (post-hoc scaling of f:d_dreb, res fixed), code/dreb_refit.py (proper: fit_window refit with
Wd['d_dreb'] x m so res_d re-absorbs; files data/dreb_refit_{N}.parquet, m in .25/.5/1), dreb_refit_score.py,
dreb_preview.py (2026 leaderboard preview). preseason_backtest.py now honours env DREB_MULT (default 1.0).
- Post-hoc: x0 13.4711 | x.25 13.4684 | x.5 13.4701 | x1 13.4860 (base). Free coef relative to rest 0.28 (0.37 / 0.21
  by half); extra-dreb coef t -3.9 (2015-19 -1.9, 2020-25 -4.0).
- Refit (the honest version): x.25 13.4736 | x.5 13.4759 | x1 13.4860. x0.5 better in 9 of 11 seasons, season-paired
  t -2.2 (x.25: t -1.8); both halves improve. Slightly below the t 2.5 bar, but consistent with earlier 0.5x finding and
  with diminishing returns on DREB (diminishing.py). Under x0.5 the lineup split optimum is still ~(0.4-0.6, 0.8).
- Preseason team backtest (tags d05_, d025_): model(minfix) 8.673 -> 8.664 (x.5) / 8.670 (x.25); blend LOSO 8.145 ->
  8.144 / 8.145; slope 0.98 -> 0.99. Neutral-to-slightly-better.
- 2026 preview (window 2023-25, def = comps + 0.8 res_d, 800+ min): big-guard gap 2.04 -> 1.64; wings -0.29 -> -0.25.
  Ranks x1 -> x.5: White 30->17, Suggs 21->13, Ausar 24->18, Anunoby 29->20, Daniels 45->34, H. Jones 67->53, Camara 65->56;
  Zubac 13->22, Drummond 40->55, KAT 90->125; Wemby/Gobert unchanged 1/2; Amen 125->126, Dort 218->209.
- TO APPLY (if user OKs): production.py Wd['d_dreb'] x0.5 (prod W itself unchanged; scale inside fit_window call) ->
  ratings_2025; preseason_backtest.py DREB_MULT default 0.5 -> regenerate pre_T/pre_P -> score_preseason ->
  minutes_age_fix -> refit_blend -> project.py -> simulate.py -> build_dashboard.py (backtest box: dashboard rebounding
  slider default 0.5 or bake into weights; check test_dashboard.js) -> republish same link.
- APPLIED 2026-09-20 (session 7). HOW: baked into the weight files, not a runtime multiplier:
  weights_prod.json and weights_bt.json Wd['d_dreb'] halved (0.529->0.264, 0.596->0.298; originals *_pre_dreb.json;
  'note' key added). production.py no longer refits W unless env REFIT_W=1 (then re-halves dreb automatically).
  preseason_backtest.py DREB_MULT env default 1.0 = the new baked weight (do NOT set 0.5 again).
  Caches updated: backtest_components.parquet f:d_dreb x0.5 and lineup_impact = res_o+res_d; bt_split.parquet res_o/res_d
  = refit values at 0.5 (from dreb_refit_{N}). Backups: data/*_pre_dreb.*, data/pre_dreb/ (pre_T/pre_P), dashboard_pre_dreb.html.
  Reran: production -> preseason_backtest -> score_preseason -> minutes_age_fix -> refit_blend (w_model 0.366, w_vegas 0.634,
  w_age -0.656, tau_blend 2.422; model RMSE 8.664, blend LOSO 8.144) -> project -> simulate -> lineup_history 1996-2025
  (stints 1996-2025 all rebuilt; window 2025 reproduces ratings exactly) -> build_dashboard (backtest box 13.479 -> 13.469)
  -> test_dashboard.js (max 0.0009 wins) -> republished same link. Saved dashboard settings were all defaults (dreb slider 1
  = new weight). 2026 forecast moves <=0.3 wins (model <=0.9: LAC +0.9, PHI +0.6; SAC -0.8, ORL -0.6, HOU -0.5).
  Other research scripts reading weights_*.json now get the halved dreb automatically.

## Dashboard bug fix: player overrides not saving  [2026-09-20, session 7]
Cause: hosted runtime (contract 0.2.52) returns db snapshots whose data() is deep-FROZEN; load() assigned OV=st.players
directly, so after the db load every Save/remove silently failed (sliders worked: they were copied). Invisible in local
tests (no db). Fix in dash_v2.html load(): st=JSON.parse(JSON.stringify(st)). New tests: code/ui_edit_test.py (plain,
iPhone emulation) and code/ui_edit_test_db.py (mock db with frozen snapshots) - run both after dashboard changes
(playwright chromium at /opt/pw-browsers/chromium-1194/chrome-linux/chrome). Republished same link.

## TANK / GAP-YEAR TEST  [2026-09-21, session 8]  -> REJECTED (nothing applied)
User question: IND 2025-26 (Finals G7 -> Haliburton Achilles -> 19-63 tank) should last season count fully?
Code: code/tank_labels.py -> data/tank_team_seasons.parquet (wpct, margin, first-55 vs late margin, drop);
code/tank_player.py -> data/tank_player.parquet (age_check framework, prod weights, lineup 0.6/0.8, player_games per team
for minute shares; needs player_games 1996-2025 = rebuild.sh 1996 2025); code/tank_team.py -> data/tank_team.parquet;
code/tank_gap.py (teammates of a star who missed season N, star returns N+1); code/tank_cf_ind.py (stakes).
Tank = wpct <= .33 OR (late margin <= -9 and late-early drop <= -4): 165 team-seasons 1996-2025.
- Player level (7,834 player-seasons, 500+ min both years, clustered by team-season): off tank team -0.08 (t -1.5), i.e.
  NOT underrated; lineup part next year -0.09 (t -2.8). Stayers -0.13 (t -2.1), movers ~+0.12 (move t -4.7 general).
  Tank-season deviation from window over-weighted? box x T -0.10 (se .17), lineup x T +0.04 (se .10): halving the tank
  season's weight would need ~-0.3 -> ruled out for lineup. Same picture for 'bad' and 'late' definitions; halves mixed.
- Star-missed-season teammates (15 team-seasons incl. IND 2014 PG, GSW 2019, MEM 2023): -0.06 (t -0.4).
- Team level (preseason backtest 2014-25, 62 teams off a tank year): actual-model -1.0 wins (t -0.9); Vegas under-
  rates them +1.7 (t 1.6); blend LOSO 8.144 -> 8.146 (worse). Gap (tank + returning star): 3 cases, nothing.
- Stakes (approx): 2025-26 weight 0.5 -> IND +3.0 wins; 0 -> IND +10.6 (model ~32 -> ~43 vs line 44.5); others <=2 / 6.5.
  IND drivers: Nembhard lineup +2.10 (2024) -> -2.14 (2025); Walker bad in both years (-1.65, -2.99).
  IND model 32.3 vs line 44.5 (edge -12.7), blend 39.1. Conclusion: tank season is as informative as any; use player
  overrides if the user wants; in-season RAPM update corrects within ~20 games.

## Model-vs-Vegas slider  [2026-09-21, session 8]  (dashboard republished same link)
Sidebar section 'Forecast: model vs Vegas': S.bm = model share (0-1, step 0.01, default blend.json w_model 0.366);
forecast = 41 + bm*(model-41) + (1-bm)*(mkt-41) + w_age*age_c (age term kept at every mix, then re-centred to 1,230).
data/blend_curve.json (built inline, see chat): in-sample RMSE of that formula on 330 preseasons at bm 0..1 step .05
(0%: 8.29, best 8.065 at .35-.40, 100%: 8.72) and tau(bm) = tau_blend * sd(bm)/sd(default), sd = sqrt(rmse^2-18.9)
(2.42 at best, 2.69 at 100%). Sim uses exact tau_blend at default, else curve. Note under slider shows accuracy at mix.
Engine treats missing S.bm as default (test_dashboard.js unchanged, passes). Pre-change copies: code/dash_v2_pre_bm.html,
code/build_dashboard_pre_bm.py. Saved settings without bm load the default. If blend.json is refit, regenerate
blend_curve.json (same recipe as refit_blend.py data prep).

## GitHub + phone app pipeline (PLANNED; user not ready yet, will start later, before opener ~2026-10-20)
Agreed design (hybrid): one PUBLIC GitHub repo (free Pages needs public; private ~$4/mo) holding code, data, app.
- GitHub Actions nightly cron: (1) ingest last night's pbp from NBA.com live feeds (runners CAN reach NBA.com/cdn, the
  sandbox can't), final scores, dated schedule, injuries/transactions, optional game lines; (2) rosters/availability +
  user overrides; (3) parse new games, refit ratings with current season (weight grows with games; rate to be set by an
  in-season backtest: project Oct + first N games, score the rest); (4) minutes per data/inseason.json (blend toward usage), injured
  zeroed until return; (5) simulate REMAINING games on the real schedule with actual wins locked; (6) write data JSON +
  dated snapshot (for odds-over-time charts).
- GitHub Pages serves the dashboard as a PWA (manifest, icons, service worker, offline last data). Add to Home Screen via
  Safari/Chrome Share. Phone layout: bottom tabs Standings / Tonight's games (probabilities vs lines) / Players /
  Settings (sliders move here); per-team odds-over-time chart; sim in a Web Worker at 5,000 seasons.
- Overrides: app writes overrides.json to the repo via GitHub API using a fine-grained token scoped to that one repo,
  entered once in app Settings; applied instantly in-app and in the next nightly run. No token -> local only.
  Existing claude.ai dashboard keeps working but can't fetch external data (CSP): republish manually from repo results.
  Its saved overrides live in db doc settings/current (read via Artifact read_db) -> migrate into overrides.json.
- Push notifications deferred (need an extra service).
- User setup (~15 min, on a computer): GitHub account + repo, upload the folder Claude provides; or give Claude a
  temporary repo-scoped token to push, then revoke it. Claude writes all code and step-by-step instructions.
- First steps when building: confirm reliable sources for dated schedule + injury reports from runners; run the
  in-season weighting backtest; test the nightly job on the final weeks of 2025-26.

## BACKLOG: future optimization (not scheduled; user-approved list)
1. [DONE 2026-09-21, see 'SLOPE REFIT' at end] CALIBRATION SLOPE REFIT under split lineup weights (added 2026-09-20). Production slope 1.070 was fit jointly with the
   age curve and rookie/unrated values using 0.6*lineup_impact. With w_off 0.6 / w_def 0.8 a game-level check lowers the
   fitted slope ~4% (1.0875 -> 1.045); team-level slope on actual 0.98. To do properly: add res_o/res_d game diffs to
   age_design2.parquet (as in bt_split.py), rerun rookie_refit.py with base = comps + 0.6*res_o + 0.8*res_d, rescale
   pre-slope rookie/unrated values by the old->new ratio (as in the age-curve refit), then preseason_backtest.py ->
   score_preseason.py -> minutes_age_fix.py -> refit_blend.py -> project.py -> simulate.py -> build_dashboard.py.
   Expected effect: team projections pulled slightly toward the middle; small.

## Status of earlier next steps
1. DONE: win totals + season sim. Remaining: dated schedule + game lines (pipeline).
2. DONE: preseason backtest; team-direction tested (only roster age kept); contract year rejected.
3. DONE: Dashboard v2. Ideas: dated schedule + game list, forecast history.
4. Daily update path: decided = GitHub hybrid above (not started).

Style notes: user knows basketball well; test their intuitions against data and report honestly, including when they're
right. User can't run code; Claude runs everything. Keep chat answers concise (user often on phone).

## SHIFT-SIGN MODEL TEST (user idea)  [2026-09-21, session 9]  -> REJECTED as a model (nothing applied)
Idea: player shift (check-in -> check-out) scores +1/-1/0 by sign of team margin during the shift; regress vs outcomes.
code/shifts.py -> data/shifts_{y}.parquet (2011-25; ~31-36 shifts/team-game, median ~7 min, 8% ties; not in zip, 10 s/season).
code/shift_bt.py, shift_bt2.py -> data/shift_bt.parquet; code/shift_score.py. Same backtest as JIMBO (13,206 games 2015-25,
ratings from T-3..T-1 decayed .3/.6/1, actual minutes, season LOSO). Adjusted versions = ridge at shift level with
teammate/opponent exposure weights (+home).
RMSE: home only 14.708 | raw sign rate 13.654 | raw +/- 13.632 | adjusted sign (lam 1000) 13.618 | adjusted +/- (lam 10000)
13.595 | JIMBO 13.435. Adding either to JIMBO: nil (13.438 / 13.436, t 0.3 / 1.3). Same ordering both halves.
Descriptive 2025-26: net shift score corr 0.95 with raw +/- (1500+ min); same-game team shift-sign diff corr 0.90 with margin.
Conclusion: sign is a noisier +/-; needs teammate adjustment to be usable, then ~equals plain RAPM, below JIMBO; no added info.

## PIPELINE BUILD  [2026-09-21, session 10]  <- START HERE for pipeline work
User's GitHub repo: github.com/jvector26/JIMBO (public, branch main). User gave a 7-day fine-grained token 2026-09-21
(stored only in sandbox /home/claude/.ghtoken, never committed; ask again in a new chat). Clone at /home/claude/gh.
Repo code is on GitHub (clone it); zip also has a copy as repo_snapshot/ (without live/ data history? no: includes live/ as of 2026-09-21).
Done this session:
- parse_pbp.run(season, cdn_df=None, meta=None, xy_zones=False, out=None): accepts live cdn action frames, a meta
  table (game,date,htm,vtm) instead of shotdetail, and coordinate zones. Pre-change copy code/parse_pbp_pre_live.py.
- code/cdn_zone.py: zone from xLegacy/yLegacy (RA r<=40, paint |x|<80 & y<142.5, corner 3 |x|>=220 & y<87.5,
  backcourt y>417.5). 99.6% agreement with shotdetail zones on all 219k 2025-26 shots (cdn 'area' field only 97.0%:
  labels distance-0 shots Mid-Range, RA/paint boundary differs).
- Validation on full 2025-26 (mirror CSV = same fields as live JSON actions): live path gives identical lineups,
  stints, points, FG/FT/REB/TOV, x3/xft, minutes; only zone counts differ (<0.5% of shots).
- repo/pipeline/nba_fetch.py (runs on Actions): schedule (scheduleLeagueV2 -> live/{S}/schedule.csv), games (final
  games' pbp -> live/{S}/pbp/{gameId}.csv.gz compact 29 cols ~17 KB/game, box -> live/{S}/box/, games_meta.csv;
  --idrange for past seasons), injuries (ESPN -> injuries.csv + injuries_hist/), odds (ESPN scoreboard -> odds/).
  curl_cffi chrome impersonation with plain-requests and S3-mirror fallbacks.
- repo/pipeline/parse_live.py: live files -> parse_pbp.run(... xy_zones=True). Tested offline on 130 games: exact.
- repo/.github/workflows/fetch-test.yml: rehearsal (2026-27 schedule + 2025-26 games 1101-1230 + injuries + odds, commit).
Next: get repo access (token or manual upload), run fetch-test, validate real JSON vs mirror stints; then build
state (preseason ratings/minutes/windows for M4 update), nightly update step, remaining-season sim on real schedule,
in-season TAU fit, then the PWA.
REHEARSAL RESULT (run 35654207009, 2026-09-21): all sources fetched from Actions. Real NBA.com JSON for 2025-26 games
1101-1230 parsed via parse_live.py = mirror stints EXACTLY (lineups, all non-zone columns, dates, home/away).
2026-27 schedule feed: 1206 regular-season games dated (2026-10-20 .. 2027-04-11), 80 per team, 40 home each; the
remaining 2 per team are NBA Cup knockout-week slots (TBD until December) -> sim must fill them (e.g. random same-conf
pairings) until the feed adds them. Opening night: BOS@DET, PHI@NYK, OKC@SAS. 67 preseason games. ESPN injuries: 76
rows with status/detail/return_date. Odds: no games yet (test once preseason starts).

## DEFENSE #3 RESULTS: charges / offensive fouls drawn, foul types, blocks by zone  [2026-09-21, session 10] -> REJECTED
Code: code/defev.py (per player-season counts from raw pbp: offd = offensive fouls drawn [v2 msgtype 6 action 4/26,
drawer PLAYER2 (100% present, other team); cdn subType offensive + foulDrawnPersonId], chg = charges (subset), sfl/nsfl
= shooting / non-shooting personal fouls committed, blk_rim/paint/jump = blocks by zone of blocked shot [v2 PLAYER3 on
misses + shotdetail; cdn blockPersonId + coordinate zones]) -> data/defev_{2012..2025}.parquet;
defev_rebuild.sh (download+extract+parse; also rebuilt player_games/stints 2015-2024, not in zip);
code/defev_bt.py (window N-2..N decay .3/.6/1, per-season centered rate shrunk poss/(poss+k), k 300/1000/3000;
game diffs N+1 with actual minutes -> data/defev_bt.parquet; data/defev_window_2025.parquet = 2023-25 player values);
code/defev_score.py (LOSO vs base box + 0.6 res_o + 0.8 res_d = 13.4759, season-clustered t, halves).
Results (added one at a time; coef in pts per 1/100-poss rate x full game):
  offd (all off fouls drawn) k1000: dLOSO -0.0001, coef 0.82 (t 1.5; halves 0.8 / 1.5)  <- right sign both halves,
     size ~ theoretical value of a possession (~1 pt), but below the t 2.5 bar.
  chg only: worse (t 1.0; 2.1 / 0.5). non-charge off fouls: t 1.5-1.7, sign flips by half.
  sfl / nsfl / total fouls (vs existing d_pf): nothing (|t| <= 1.4 or flips).
  blk_non (paint+jump blocks, given d_blk): coef negative, t -1.1 to -2.4 but 2015-19 ~0, 2020-25 -2.3 to -3.4 (fails).
  blk_rim: nothing.
Face validity good (offd leaders 2023-25: Camara +1.23/100, Shead, Dort +0.76, Brunson, NAW, McConnell) but player sd
only 0.21 pts/100 -> team effect negligible. Named wings: Amen -0.06, Ausar +0.18, Daniels 0, H. Jones +0.13.
Not applied. Optional (user call): credit offd at ~0.8 as a theory-consistent small term (LOSO neutral).
Remaining defense items: #4 (position-interacted def weights), #6 (display), #7 (tracking; pipeline).

## POSITIONS dataset  [2026-09-21, session 10]  (user wanted positions regardless of the test)
Before: only bio.json listed position (coarse G/G-F/F-G/F/F-C/C-F/C, one per career), missing 2.1% of minutes.
code/positions.py -> data/positions.parquet (every player-season 1996-2025): pos_listed, height_in, weight_lb,
size_score (min-weighted regression of listed position on height, weight, season-centred per-36 ast/oreb/dreb/blk/stl/
3PA/FTA/pf + rim share: R2 0.815; stats-only fallback R2 0.661 when height missing), pos_est = Basketball-Reference-
style lineup rank (each stint's 5 players ranked by size_score, 1 = PG .. 5 = C, time-weighted mean), pct_1..pct_5,
pos5 (cuts 1.5/2.5/3.5/4.5; ~20% of minutes each), pos3 (G < 2.5, W 2.5-3.9, B >= 3.9: 40/26/33% of minutes).
Corr with listed position 0.91. Spot checks sensible (Curry/Brunson/CP3 PG; SGA, Amen, Daniels, Klay SG; Ausar, Giddey SF;
LeBron, KD, Draymond, Barnes, Flagg PF; Jokic-type, Mobley, Wemby, Giannis(4.3) PF/C). Needs stints 1996-2025
(all rebuilt this session; not in zip). Not yet in project.py / dashboard (2026 no-history players need listed pos +
height from bio/draft data).

## DEFENSE #4 RESULTS: position-interacted defensive weights  [2026-09-21, session 10] -> REJECTED
code/defpos_bt.py (window feature x (pos_est-3), x guard dummy, x big dummy; wings reference) -> data/defpos_bt.parquet;
code/defpos_score.py. 18 tests vs base 13.4759. None passes (t>2.5 + same sign both halves). Closest: guard defensive
rebounds worth less: d_dreb x G coef -0.20 vs Wd 0.30 (t -1.9; halves -0.4 / -2.2), dLOSO -0.0028. Blocks, steals,
fouls, opp rim/shot quality: no consistent position dependence (several t~3 in one half, ~0 or opposite in the other).
Note: rank-based position sums to ~15 per lineup, so a pure position offset is not identifiable from lineup data;
this test only asks whether each box stat's VALUE differs by position.
Defense roadmap now: #6 display, #7 tracking (pipeline). Backlog: calibration slope refit.

## SLOPE REFIT under split lineup weights (backlog #1)  [2026-09-21, session 10] -> APPLIED, dashboard republished
code/slope_refit.py (rookie_refit.py design + bt_split res_o/res_d; current caches incl. dreb 0.5x + shooting prior):
old base (0.6 single lineup) slope 1.101, LOSO 13.4053 | production base (0.6 off / 0.8 def) slope 1.050, LOSO 13.3951.
Pre-slope old->new ratios: col_a 1.018, col_b 1.048, non-college 1.022, unrated 1.042 (data/slope_refit.json).
Applied: calib.csv slope 1.070 -> 1.0501, unrated -4.139 -> -4.313; rookie_value.json col_a -2.97 -> -3.02,
col_b 3.53 -> 3.70, rk/rk_lp/rk_age x1.022. Backups data/pre_slope/ (calib, rookie_value, blend, curve, proj, sim, pre_T/P).
Chain rerun: preseason_backtest -> score_preseason -> minutes_age_fix -> refit_blend (w_model 0.367, w_vegas 0.633,
w_age -0.641, tau_blend 2.422; model RMSE 8.664 -> 8.661, blend LOSO 8.144) -> blend_curve.json regenerated
(recipe now inline in chat: in-sample RMSE at bm 0..1 step .05 with w_age fixed; tau = tau_blend*sd(bm)/sd(default);
rmse0 8.065, best at .35) -> blend.json tau_model 2.72 -> 2.68 (score_preseason) -> project -> simulate ->
build_dashboard -> test_dashboard.js (0.0009) + ui_edit_test_db.py (pass) -> republished same link.
Team moves <= 0.3 wins model, <= 0.1 forecast (spread 9.56 -> 9.42). Dashboard backtest box 13.469 unchanged.

## POSITIONS v2 + DEFENSE #6 (dashboard) + #7 source check  [2026-09-21, session 10] -> dashboard republished
Positions v2 (code/positions.py; supersedes the v1 description above): pure lineup rank was fragile when guards are
near-equal in size (SGA came out PF in OKC's small lineups; bio weights also have errors, e.g. Ajay Mitchell 235 ->
weight dropped from the size score, R2 0.798). Final pos_est = mean(lineup rank pos_rank, size_score clipped 1-5,
listed-position map) minus a ball-handling tilt for guards/wings: 0.12 x season-centred ast/36 (clip -3..6) x
clip(3.2 - mix, 0, 1). Cuts: PG <1.5, SG <2.5, SF <3.45, PF <4.4, C (minute shares 22/18/20/23/17%); G <2.5,
W <3.9, B (40/32/28%). Spot checks: Curry/Brunson/Cade/Haliburton/Harden/LaMelo PG; SGA, Booker, Ant, Daniels,
Caruso, Klay, Amen, Giddey SG; Ausar, Kawhi, Barnes, OG, Draymond, Tatum SF; LeBron, KD, Giannis PF; Bam, Jokic, Wemby C.
2026 players: code/newcomer_pos.py -> data/pos_2026.csv (NBA: 2023-25 minute/decay-weighted pos_est; 72 college
newcomers: regression of rookie pos_est on last college season per-40 ast/oreb/dreb/blk/stl, 3PA share, ESPN G/F/C,
fit on 2024-25 + 2025-26 rookies, R2 0.88, rmse 0.45 (ESPN player_box 2024-26 in data/ncaa/); 7 internationals set
by hand from listed positions: Diop, Biberovic, De Larrea, Diallo, Toohey, K. Lopez, Markovic).
Dashboard (dash_v2.html; pre-change copy dash_v2_pre_pos.html; build_dashboard adds "ps" per player):
- Position shown next to team ("OKC · SG"); position filter (Guards/Wings/Bigs/PG..C).
- Defense toggle "Defense vs league / vs position": positional baseline = minutes-weighted mean of def, D-reb,
  Stops among rated rostered 500+ min players per pos5 bucket, linearly interpolated on the 1-5 scale; recomputed
  live under sliders/overrides. Def column now visible on phones.
- Defense split: new sortable columns D-reb (def_rebounding) and Stops (rim protection + perimeter + lineup
  defense; hidden on phones); detail row shows position, split, and split vs position. def = D-reb + Stops + aging/2
  (+ returner/2). Fixed-value rookies/newcomers show '-' for the split.
- Tests: test_dashboard.js 0.0009, ui_edit_test_db.py pass, no page errors (desktop + 390px).
Defense vs position leaders: wings Ausar +2.8, Ajay Mitchell, OG, Hugo Gonzalez, Draymond, Camara, H. Jones;
guards Suggs +3.6, Caruso, Thybulle, D. White, Daniels +2.6, Kris Dunn, Wallace.

#7 SOURCE CHECK: a public mirror exists. github.com/gabriel1200/site_Data (same repo as our contract data; git clone
works, blob:none partial clone + checkout of single files is fast). Year label = season END year.
- hustle.csv (root): 2017-18 .. 2025-26 (years 2018-2026), ~550 players/season: CONTESTED_SHOTS (2PT/3PT),
  DEFLECTIONS, CHARGES_DRAWN, loose balls, box outs (off/def), screen assists, MIN, distance/speed. Updated DAILY
  during 2025-26 (commits every day Mar-Apr 2026) -> usable for in-season updates too (pipeline can pull it).
- {Y}/defense/dfg.csv, rimdfg.csv, rimfreq.csv, rim_acc.csv: 2013-14 .. 2024-25 (no 2025-26 yet): defended FGM/FGA,
  DFG% vs expected (DIFF%), rim DFG% and frequency. {Y}/hustle/hustle.csv older per-year copies 2016-17..2023-24.
- Also player_tracking/ (touches, drives, pullup, cs, passing, post_up, paint, elbow), opp_shooting/ by defender
  distance (2026 folder exists for those). No matchup data there; shufinskiy/nba_data has 'matchups' 2017-18..2024-25.
Not yet tested in the model. Test plan (same game-level LOSO framework as defense #3/#4): per-100 deflections,
contests (2PT/3PT), rim DFG DIFF (shrunk by DFGA), windows N-2..N; hustle only from 2017-18 -> backtest 2020-21..2025-26
(fewer seasons; halves 2020-22 vs 2023-25).

## DEFENSE #7 RESULTS: tracking / hustle stats  [2026-09-21, session 10] -> PARTLY APPLIED (hustle term)
Data: data/tracking/ (hustle.csv root file from gabriel1200/site_Data, seasons 2017-2025 (year-1); rimdfg_/dfg_{S}.csv
2013-2024, older files have no PLAYER_ID -> matched by normalized name+season, 16 unmatched rows).
code/track_bt.py (features per 100 def poss, season-centred, shrunk k 1000/3000; windows N-2..N over seasons with data;
game diffs N+1 actual minutes (no pace factor, ~1%) -> data/track_bt.parquet, track_window_2025.parquet);
code/track_score.py (hustle tests on games 2018-2025 only, halves 2018-21 / 2022-25; DFG tests 2015-2025).
Hustle-era base LOSO 13.8675. Single terms (dLOSO, coef, t, t halves):
  DEF loose balls recovered k1000: -0.0131, 2.03, t 4.3 (3.9 / 3.3)  <- clear pass
  contested 3PT k3000: -0.0062, 0.37, t 3.2 (2.7 / 1.9)               <- passes (overall t>2.5, same sign)
  deflections: t 2.1 (fails); all contests / 2PT contests / def box outs: nothing.
  rim DFG% points saved and overall DFG% saved: nothing (t 0.1-0.5; sign flips) - defended FG% is noise, as in public work.
  rim DFGA volume (shots defended at rim, 'presence'): t 2.6 (4.0 / 1.9) - passes narrowly but NOT applied: the mirror
  has no 2025-26 data and no in-season updates for these tables.
Joint: lbr+cont3 -0.0146; +rimdfga -0.0174; lbr correlates with d_stl 0.47 (defl 0.44), cont3 with d_blk 0.50.
APPLIED: code/hustle.py -> data/hustle_2026.csv (+ hustle.json cfg): hustle = 1.74*lbr_k1000 + 0.253*cont3_k3000
(joint coefs), post-slope rating pts/100 added to def; project.py column 'hustle' (0 for no-history players);
dashboard: payload 'hu', part 'hustle' (color --c-hustle), included in Def and Stops; player-view projection bar.
sd 0.19 (range -0.47..+0.81). Top 2023-25: Josh Hart +0.68, McConnell, Ausar +0.66, KPJ, Caruso +0.62, Daniels +0.53,
Eason, K. George, Flagg; bottom: Beasley, Gobert -0.47, J. Green, Trent, Kuzma, Trae. Amen +0.06, Camara -0.10.
Team moves <= ~1 win. NOT in preseason_backtest / blend refit / dashboard backtest box (hustle only 2017+;
blend weights fit without it - effect tiny). In-season: hustle.csv updates daily -> pipeline can refresh the window.
DEFENSE ROADMAP CLOSED for now (user 2026-09-21).

## DASHBOARD 90s THEME  [2026-09-21, session 10] -> republished (same link)
User asked to make it "wild: 90s theme, bright colors, fun fonts". All styling is one appended CSS block in
code/dash_v2.html ("90s THEME"); austere original kept as code/dash_v2_austere.html (swap templates to revert).
Fonts (Google): Bungee Shade (JIMBO title, NBA Jam style: yellow, black stroke, magenta+purple extrude, tilted, one
pop-in animation, reduced-motion respected), Bungee (headings, tabs, team names, table headers), VT323 (all numbers,
via --cond), Comic Neue (body). Palette: Hornets teal bg #00A3AD with Memphis confetti SVG pattern, purple #4B1F8C,
magenta #FF1F8E, yellow #FFE500; dark mode = deep purple #1A0B3B with neon tokens (--hd, --mark, --ring switch).
Settings aside = Windows 95 window (title bar 'jimbo_settings.exe', bevel buttons); accuracy box = green LCD.
Tabs = Trapper Keeper dividers (yellow/pink/cyan); panels/tables thick black borders + hard offset shadows;
conference headers = tilted magenta stickers. Phones: Players table fits (team/pos under name), smaller shadows.
Screenshot review: fonts from raw.githubusercontent.com/google/fonts (sandbox can't reach fonts.googleapis) served via
Playwright route (/tmp/shots90.py recipe in chat). Tests pass (engine 0.0008, db edit test).

## NIGHTLY PIPELINE LIVE (preseason mode)  [2026-09-21, session 10]
Repo github.com/jvector26/JIMBO (clone /home/claude/gh). Workflow .github/workflows/nightly.yml: cron 11:00 and 21:00
UTC + manual (input asof). Steps: nba_fetch schedule/games/injuries/odds -> pipeline/nightly.py -> commit live/,
site/data/ (latest.json, history.json). First run 2026-09-22 00:34Z SUCCESS (bot commit "nightly ..."). fetch-test.yml
removed. requirements.txt. Actions logs are NOT readable from the sandbox (log blob host blocked) -> debug by
reproducing locally; step conclusions are readable via the jobs API.
Code:
- model/jimbo_update.py: production M4 (partial_features, stint_rows, rapm_delta, team_offsets, update_ratings);
  weights = 0.5 x weights_prod (fit_window convention), lam 6000, dS 1, H0 2000, lam_t 80, min 15 games.
- model/jimbo_sim.py: rest-of-season sim on the real schedule with results locked; TBD NBA-Cup slots filled with random
  same-conference games (fill_tbd) until each team has 82; postseason() as simulate.py.
- model/parse_pbp.py: paths from env JIMBO_RAW / JIMBO_DATA; shooter_pcts(season, out) -> league-average first pass
  when no totals exist; pipeline/parse_live.py runs two passes on a fresh output dir (state/season_parse, gitignored,
  rebuilt every run from live/{S}/pbp).
- pipeline/build_state.py (SANDBOX ONLY): state/2026/ = players.parquet (roster + FA + all rated players not on a
  roster at box-only history rating; key, rating_pre, mpg_pre = min_model/(82*avail_pre), avail_pre = exp_games/82,
  pos_est, hustle...), hist_box.parquet (window 2023-25 feature sums, wo/wd), teams_pre.csv (model & blend preseason
  margins/wins, line, mkt_mean), config.json. `2025 --replay` builds state/2025_replay from pre_P_2025/pre_T_2025.
  REBUILD state/2026 + push whenever projections change (rosters, overrides of model, refits).
- pipeline/nightly.py: schedule (live/{S}/schedule.csv, kind reg) -> results locked (W/L from schedule scores) ->
  parse -> ratings (M4) -> pace (season stints once 60+ games) -> team offsets -> rosters (team = last team played
  for; overrides t) -> minutes: mpg_est = (10*mpg_pre + gp*mpg_season)/(10+gp); avail = (20*avail_pre + gp)/(20 + team
  games); ESPN injuries: Out -> games until return_date (or rest of season if comment says so, else 10), DTD 0.5;
  overrides (g of 82 scaled to remaining, mpg; r rating) from overrides.json; allocate 240*G_rem per team (trim bottom /
  scale, cap 3000/82 per game) -> model margin (zero-sum) + offsets; forecast = model + (blend_pre - model_pre) *
  20/(20+team games) -> tau interpolated 2.42 @0 games, 2.0 @41, 1.8 @70 -> sims (forecast + model) -> latest.json
  {teams: w,l,proj_w,p10,p90,model_w,pre_w,line,playoffs,playin,top6,seed1,r2,cf,finals,title,over,margin,offset;
  players: k,n,t,pos,r,r_pre,gp,mpg,min_rem,status; tonight: game, time, home, away, margin, p_home, line, total}.
  history.json keyed by asof date: team -> [proj_w, playoffs, title, w, l].
- overrides.json at repo root (migrated from dashboard db settings/current 2026-09-21: empty, sliders default).
Validation: day-0 parity with dashboard (no injuries): team model margins match project.py within 0.005; forecast
wins within 0.7 (real vs formula schedule, sim noise). 2025-26 replay (state/2025_replay, sandbox-parsed data):
final-wins RMSE by date: preseason 9.5 | Dec 15 (25 g) 5.66 (win-pace 7.83) | Feb 1 (49 g) 4.27 (4.36) | Mar 15 (67 g)
2.11 (2.06). p10-p90 coverage 63-83% (one season). PROVISIONAL params: gap_k 20, mpg_k 10, avail_k 20, tau curve.
NEXT: (1) multi-season replay 2014-2025 to tune gap_k / tau / minutes params (build_state --replay S for each S;
nightly.py --parsed /home/claude/data --schedule-from-parsed; score vs final records + coverage);
(2) phone app (PWA) on GitHub Pages reading site/data; (3) hustle refresh in-season; (4) token expires ~2026-09-28.

## DASHBOARD: toned-down light 90s look + settings drawer  [2026-09-21, session 11] -> republished (same link)
User: keep 90s vibe but light mode only, more readable fonts, sliders behind a dropdown at the bottom, forecasts first.
- code/dash_v2.html: dark-mode blocks removed (color-scheme light; stays light on dark-mode devices / viewer theme).
  Theme block "90s LIGHT THEME": cream bg #FBF8F1 + faint pastel confetti, 2px ink borders, soft lavender/cyan offset
  shadows, pastel Trapper Keeper tabs, magenta conference stickers (-1deg), purple table headers (white Rubik).
  Fonts: Rubik (body, team names), Barlow Semi Condensed (all numbers), Bungee (tabs, stickers, drawer title),
  Bungee Shade (JIMBO logo only, smaller), VT323 only for the green LCD accuracy numbers.
  Loud version kept as code/dash_v2_90s_loud.html; austere original code/dash_v2_austere.html.
- Layout: single column. Former sidebar is now <details id="settings"> at the bottom of main (closed on load), styled
  as a Win95 title bar; summary line #setSum shows "N sliders changed · N player changes" (renderSetSum in renderAll).
  Inside: left = accuracy LCD + player-changes list; right = sliders in CSS columns (2 x 260px). Phone lede hidden.
- ui_edit_test.py / ui_edit_test_db.py now open the drawer before reading #chgList (db test takes dashboard path arg).
  All tests pass (engine 0.0008, db edit/remove OK, no page errors). Screenshot recipe: /tmp/shots.py (fonts from
  raw.githubusercontent.com/google/fonts routed via Playwright).

## PHONE APP (PWA on GitHub Pages)  [2026-09-21, session 11] -> BUILT, TESTED, PUSHED (commit 9d76dbd)
URL: https://jvector26.github.io/JIMBO/ . Pages NOT yet enabled: the token can't (POST /repos/.../pages -> 403, needs
Administration). User sets repo Settings > Pages > Source = GitHub Actions; then dispatch pages.yml (or wait for next nightly).
Token (7-day, Contents/Workflows/Pages) stored only in /home/claude/.ghtoken (never committed); expires ~2026-09-28.
Nightly dispatched after push: success with the new fields (commit 3a686f2).
- One template for both: code/dash_v2.html. claude.ai artifact = inline payload (preseason only, unchanged behaviour);
  app = code/build_app.py -> site/index.html (+ code/appsrc/app.js loader, sw.js, manifest, icons) and site/data/dash.json
  (payload copied from dashboard.html; versioned ?v=hash). Rebuild: build_dashboard.py -> build_app.py -> commit site/.
  Main code is jimboBoot(D, SZ, APP); SZ = latest.json (nightly), APP = hooks {nsim, hist, remote, save, afterBoot}.
  window.__jimbo exposes P/SIM/S/OV for tests.
- In-season layer (season block in template): rating = slider-built preseason rating + (nightly r - r_pre) (part 'season');
  team = override t || nightly t; minutes = nightly formula in JS (want = (g_rem - miss)*avail*mpg_est; overrides
  g/82*g_rem*mpg locked; allocate 240*g_rem, cap 3000/82*g_rem); net/(48*g_rem) zero-summed; model margin = net*pace/100 +
  offset; forecast = model + gap*dec, gap = preseason-engine (bmargin - margin) at CURRENT sliders + overrides (nightly uses
  fixed teams_pre gap0: same at defaults; with overrides nightly moves forecast 1:1 with model, app ~37%).
  Expected wins = W + sum Phi(d/sqrt(SIG^2+2tau^2)) over remaining + TBD-filled games; sim = E.simulate(..., {sched, w0})
  (no re-centering, as nightly). TBD NBA-Cup slots filled in JS with its own seed.
  Day-0 parity vs nightly: margins <=0.008, wins <=0.6 (TBD fill + MC noise), playoffs <=0.03.
- Tonight tab: nightly `tonight` (today, or next game day = new field game_day); win prob recomputed in-app from
  mpg_est x status (Out 0, DTD 0.5) + edits; market line from ESPN odds when posted (parsed 'TEAM -x.5').
- Team detail: projected-wins/playoff history chart (history.json). Record column once games are played.
- Phone (<760px, app only): bottom nav Standings/Tonight/Players/Matchups/Settings; Settings = drawer as a page; compact
  header, forecast legend hidden. Sim 5,000 on phones (main thread), 10,000 desktop. Web Worker not done.
- Sync: edits -> localStorage always; with a token saved in app Settings (localStorage 'jimbo-gh-token') -> PUT overrides.json
  via contents API ({_note, players, sliders, updated}). Load: remote (API with token, else raw.githubusercontent) wins if
  its `updated` is newer than local. Nightly applies players only (sliders are app-side).
- Service worker: shell + versioned dash.json cache-first; index + nightly JSON network-first, offline fallback.
- nightly.py: teams += gp, gap, dec; players += mw, me, av, miss and now ALL rostered players (injured were dropped);
  top-level game_day, cfg {gap_k,sig,hca}, rem ["HOM-AWY", ...]. Injury rule: explicit return_date beats comment text
  (stale Feb-2026 note had Furphy out all 2026-27; now 32 games). ESPN list still has many offseason DTD tags (0.5 g each).
- .github/workflows/pages.yml: deploy site/ on workflow_run(nightly), push to site/**, manual.
- Tests: /tmp/apptest.py (local server, parity vs latest.json, screenshots), /tmp/apptest2.py (synthetic mid-season +
  mocked GitHub API: token save, edit -> PUT, new device pulls remote). Artifact tests still pass. Recreate from chat if needed.
- Pre-app template: code/dash_v2_pre_app.html. NBA feed tip times look like placeholders (e.g. 19:00Z) until set.

## CROWDED-ROSTER MINUTES RULE TEST (code/alloc_test.py)  [2026-09-21, session 11] -> NO CHANGE (current rule kept)
User Q: Dieng/Sims (MIL) projected 0 min (MIL wants 24,447 > 19,680; bottom-trim zeroes smallest first; they miss by ~70).
Test = minutes_age_fix framework (pre_P min_model x LOSO age mults -> allocation -> team wins vs actual, 2014-25 excl 2019;
44% of team-seasons are over 19,680). Team RMSE (wins) / player-minute RMSE:
  bottom (current) 8.661 / 577 | proportional 8.691 / 558 | power g=1.5..5 8.700..8.665 / 564..572 |
  floor f (keep >= f x projection, rest bottom-trimmed) 0.25: 8.665 / 574, 0.4: 8.673, 0.5: 8.681. All vs bottom |t| < 1.1.
Zeroed players with 600+ wanted (68 cases) actually played median 512 min (51% >= 500) -> the cliff is wrong for individual
minutes, but spreading minutes evenly takes them from starters (MIL proportional: top player 1,899 -> 1,529), which costs
team accuracy. Proportional would move 2026 model wins up to -4.6 (SAS), unsupported. Floor 0.25 = tie on teams, slightly
better player minutes: offered to user as optional.

## IN-SEASON MINUTES RULE IN NIGHTLY + FULL ROSTER VIEW  [2026-09-22, session 11] -> PUSHED (2693ba1), nightly OK on Actions
User: Sims/Dieng (MIL) at 0 min; wants all players in team view; asked whether in-season minutes adapt quickly.
- Old nightly minutes (mpg_est k=10 on games played, avail k=20, bottom-trim) was NOT the backtested session-5 rule.
  Old rule: Sims at 20 mpg -> 0 after 5 games, ~15 mpg after 10.
- 9-season nightly REPLAY (2015-18, 2021-25; build_state S --replay, nightly at Nov 5/Nov 20/Dec 15/Feb 1/Mar 15;
  scripts /tmp/replay_season.sh, /tmp/replay_var.sh TAG key=val, /tmp/cmp_all.py - recreate if needed; stints/pg/games
  2015-2025 parsed into data/ this session, not in zip). Player rest-of-season min RMSE | team rest-of-season margin RMSE
  (season-clustered t vs old) by checkpoint:
    old            526 474 406 273 149 | 3.34 3.21 3.38 4.04 5.04
    k5 + streak    542 508 434 269 126 | 3.33 3.22 3.41 3.89 4.67 (Feb t -2.2, Mar -3.2)
    k5 no streak   518 483 409 256 123 | 3.34 3.22 3.40 3.95 4.72 (Feb -1.7, Mar -3.9)  <- CHOSEN ("final")
    trim variant   576 528 439 269 130 | 3.34 3.21 3.39 3.96 4.68 (prior = pre-trim wants, bottom-trim allocation)
  k=10 ~ k=5. Absence-streak x0.5 fallback hurt player minutes (replay has no injury reports; live has them) -> off.
- nightly.py now: m_tg = w*(0.5*l10 + 0.5*std) + (1-w)*pre, w=n/(n+5), n = team games since joining current team,
  DNPs 0; pre = min_proj/82 (preseason ALLOCATED), FA (state team FA) pre = mpg_pre*avail_pre; FA zeroed after team
  overrides; rescale per team to 240 (config min_alloc 'rescale'); want = (g_rem - miss)*m_tg; allocate() as before
  (locked overrides; trim only if overrides overfill). Tonight strength uses m_tg x status. config.json: min_k 5,
  min_rec .5, min_streak 999 (off), mpg_k/avail_k removed (code defaults 10/20 kept for display mpg_est/avail and
  override defaults). latest.json players += mg (rescaled), mr (raw). Day-0 changes: injured players' minutes now spread
  proportionally (WAS Watkins out: Middleton stays ~200 instead of next-man-up 1,040; WAS margin +0.3); untested either way.
- App/dashboard (dash_v2.html): season layer uses z.mr (fallback p.mm/82), rescales per post-override team, want =
  (g_rem - miss)*mtg; tonight uses mtg. Team view lists ALL roster players: rotation, then "No projected minutes (N)"
  divider row + zero-minute players (dashes, Edit). Phone: roster sticky to screen width, compact, 'Adds to team' hidden.
  Tests pass: engine 0.0008; app day-0 margins 0.006; sync/db tests OK. claude.ai artifact NOT republished with the roster
  change yet (publish dashboard.html to the artifact URL if wanted).
- Repo: .gitignore excludes test replay states (2025_replay tracked as before).

## BACKLOG: app features (model-optimization backlog is the earlier "BACKLOG: future optimization" section)
### B1. Game detail page in the phone app (requested 2026-09-22) -> BUILT + PUSHED 2026-09-22 (30cc1cb), see 'GAME DETAIL PAGE' at end
Tap a game on Tonight -> full page (same pattern as the player view: history push, Back, bottom nav stays).
Everything below except C/D/E can be computed in-page from data the app already has (tonightGames() in dash_v2.html).
A. Header: teams, records, tip time, win % each side, JIMBO spread, market line + total (ESPN), gap vs market.
B. "How the odds are built" waterfall (home minus away, in pts per game), each term from tonightGames():
   lineup strength (sum rating x expected-min share, per 100 poss) -> x pace/100 -> + team offsets (in-season residual,
   ridge lam_t 80) -> + blend adjustment (market/roster-age gap x 20/(20+games played)) -> + home court 2.6 -> spread ->
   win % = Phi(spread/12.8). Refactor tonightGames() to return these parts instead of only sp/ph.
C. Expected rotations, both teams: every roster player with tonight's expected minutes (m_tg x status: Out 0, DTD 0.5,
   rescaled to 240), rating, points contributed; injured/out players listed at 0 with the injury note; user edits
   flagged; tap -> player view, Edit button -> editor (odds update live). Label honestly: these are expected minute
   SHARES, not a starting lineup. True starters are only known ~30 min before tip (would need a pre-tip pipeline run;
   the nightly runs at 7 AM / 5 PM ET; the 5 PM run could refresh injuries).
D. Pace: the model uses ONE league pace (cfg pace, from season stints once 60+ games); team-specific pace is not
   modeled, so it only scales the margin. Show it plainly. Adding team pace + a projected total is a MODEL item
   (backtest vs ESPN totals) -> separate backlog entry if wanted.
E. Context (display only, not in the model): last-10 results, rest days / back-to-back, earlier meetings this season.
   Rest/travel are NOT in the model (roadmap: pipeline-dependent signals) -> label as "not used by JIMBO".
F. Uncertainty: margin distribution (normal, sd 12.8) with the market spread marked; P(win by 10+).
Data changes (nightly.py): latest.json += per-team recent results (date, opp, home/away, score) and schedule dates
for remaining games (rem has no dates) for rest days; optional daily odds snapshots for line movement (live/odds/{date}.csv
already saved per day -> small history array per game).
Effort: ~1 session (UI ~70%, small nightly additions, tests: parity of waterfall sum = spread, phone screenshots).
### B2. Jericho Sims "3PT drag" check (user, 2026-09-22) -> checked cheaply; follow-ups backlogged
- Sims three_pt -1.33 = o_3vol -1.31 (takes ~no 3s vs league 3PA rate) + o_3eff -0.02 (accuracy term ~0 because his
  volume ~0). So the drag is 3PT VOLUME (spacing), not poor shooting. Every non-shooting big gets ~-1.4 (Gobert, Zubac,
  Allen, Duren, Poeltl...); bottom 10% of volume avg -1.28.
- Is it real? Game backtest (backtest_components + bt_split, lineup .6/.8): o_3vol multiplier 0 -> RMSE 13.504,
  0.5 13.469, 0.75 13.465, 1.0 13.469, 1.5 13.499; same shape 2014-19 and 2020-25 (optimum 0.5-1.0). Free coefficient
  0.88 +- 0.14 (t ~6). Removing it clearly hurts -> spacing penalty is supported by data.
- Side finding (free coefs, collinear, treat as hint): o_rimvol 2.0 +- 0.4 and o_oreb 2.0 +- 0.5 look UNDER-weighted
  (big-man strengths), d_dreb 0.43 +- 0.32 over-weighted (matches earlier rebounding 0.5x result).
Backlog:
  a. UI: split the "Three-point shooting" bar/label into "3PT volume (spacing)" and "3PT accuracy" in player view +
     sliders tooltip (cheap; offered to user).
  b. Position-specific spacing: does a non-shooter cost less as a dunker-spot big than as a guard/wing? Test in
     diminishing.py framework (stints 2012-25 lineup sums; split o_3vol sum by big/non-big; stints 2015-25 were rebuilt
     this session, not in zip). If bigs' penalty is smaller -> position-scaled o_3vol.
  c. Rim/OREB volume under-weight: refit W with o_rimvol/o_oreb freed per position or check via backtest multipliers
     (like the 3PT shooting 1.5x study). Could offset part of bigs' spacing penalty.

### B3. COACHING EFFECT (user, 2026-09-22; backlog only, not scheduled)
Question: does the coach move team results beyond the players (e.g. MIL: Doc Rivers -> Taylor Jenkins for 2026-27)?
Plan when picked up:
- Data: head coach per team-season (and mid-season changes with dates) 2004-2026. Source to find: a GitHub dataset
  (Basketball-Reference scrape) since the sandbox can't reach B-Ref; 2026-27 hires from user/news.
- Outcome = team residual the player model misses: actual margin minus player-based expectation with ACTUAL minutes
  (backtest_components / stint data), per team-season (and per half-season around mid-season firings).
- Tests: (1) coach fixed effects (ridge-shrunk) on residuals; (2) MOVERS design: does a coach's residual follow him to
  a new team (like player RAPM persistence)? This is the key test; year-to-year team persistence alone is confounded by
  front office/arena/culture; (3) new-coach dummy and "coach quality" (prior-team residual) in the LOSO preseason blend
  (test_direction.py method; bar t > 2.5, same sign both halves). Also look at MINUTES: does a new coach redistribute
  minutes more than the minutes model expects (coach-driven rotation changes)?
- Caveats: ~60-80 coach changes 2014-25, residual noise ~2-3 pts/season -> only a large effect (sd >~1 pt/100) is
  detectable; firings follow bad luck (regression to the mean looks like a new-coach bump -> compare with teams that
  had equally bad luck and kept the coach). In season, team offsets (lam_t 80) already absorb a coach effect after
  ~20-30 games; the value would be the PRESEASON prior.
- Expectation: plausible small signal; worth one session.

### B4. DONOVAN CLINGAN OVER-RATING (user, 2026-09-22; backlog, not scheduled) - suspected SYSTEMIC
JIMBO has Clingan ~#5 in the league (+5.9 projected, ~Luka's +6.1); public models (EPM/DARKO/LEBRON) put him ~50-100.
User does not accept "lineup data says so" as an answer; the goal is to find what is over-crediting him and whether it
hits a whole class of players. Do NOT start from the conclusion that it's real.
Quick look (ratings_2025, window through 2025-26, before slope/aging): box parts: OREB +1.10, rim protection +1.23,
DREB +0.70, rim finishing +0.33, ball security +0.32, mid +0.15 vs 3PT volume -0.59, playmaking -0.57, foul drawing
-0.17 (box ~ +2.3); lineup residual res_o +1.30 / res_d +2.27 (weighted 0.6/0.8 -> ~+2.6, ~45% of rating);
window only 6,824 poss (Luka 13,360, Gobert 14,293); age 21.9 -> young aging bump. Suspects:
  1. LINEUP RESIDUAL on a small sample: ridge lam is fixed per possession, maybe too weak for ~7k-poss players; test
     persistence of res_o/res_d by sample-size bin (next-window r vs this-window) and whether small-sample high
     residuals regress more than the shrinkage assumes -> sample-dependent extra shrinkage.
  2. BIG-MAN STACKING: rebounding (OREB+DREB), blocks/rim protection AND lineup_d may double count the same
     defensive/rebounding value (DREB looks over-weighted: free coef 0.43; earlier rebounding 0.5x best). Also the
     diminishing-returns finding (team DREB saturates) is not in individual ratings.
  3. CONTEXT: Portland teammates / backup center quality (on-off vs a bad backup inflates lineup residual).
  4. AGING bump for <=22 applied on top of an already rising young player.
Tests: (a) systemic comparison: pull a public model (EPM/DARKO/LEBRON; find a source reachable from sandbox or GitHub
runner) and compare ranks by position, age, window sample size, lineup share, component; find the group JIMBO
over-rates. (b) Out-of-sample arbiter: for players where JIMBO and the public model disagree most, whose rating
predicts NEXT-season lineup results/team margins better (needs historical public ratings; else use JIMBO's own
next-window check). (c) If a fix is found: refit + game backtest + preseason team backtest as usual (bar: better
out of sample in both halves), and re-rank Clingan/bigs.
Related backlog: B2 (spacing penalty for non-shooting bigs, rim/OREB weights) - same component set, run together.

## ROOKIE COLLEGE STATS DISPLAY FIX  [2026-09-22, session 11] -> PUSHED (c99a606) + artifact republished
User: Nate Ament shown 33.4 "points" (he averaged 16.7). Not a data/model bug: college stats (ncaa_scores x_*) are per
100 on-court possessions (584 pts / 1,750 poss); the page showed them unlabeled. Affects every college rookie (per-100 /
per-game ratio 1.5-3.5, e.g. Dybantsa 41.2 vs 25.5 ppg). Model inputs unchanged (pace-neutral rates are intended).
Fix: build_dashboard.py adds st.pg = pon/g/100 and st.g from ncaa_draft_match (2026); player view shows counting stats
per game with the per-100 rate in small print; formula constants rounded (were printed to 16 decimals).

## B4/B2 BIGS INVESTIGATION RESULTS  [2026-09-22, session 12] -> NOTHING APPLIED (no bigs-specific bias found)
User: Clingan (#5, +5.9) looks over-rated, Sims (-2.7) possibly under-rated; asked for tests 2+3 only (NO public-model
benchmark: user does not want to train off other models). Also asked: is plus-minus controlled for teammates? (yes: ridge
on 10-man stints; additive, no player-pair chemistry; thin-sample players shrink toward box prior).
Code: code/bigs_windows.py (windows 2014-2025, weights_bt; PROD=1 -> weights_prod) -> data/bigs_win_{N}[_prod].parquet
(res_o/res_d, f:*, dposs decayed, thin=2000/(dposs+2000), ctx = backup-thinness when OFF minus teammates when ON, pos, age);
code/bigs_bt.py -> data/bigs_bt.parquet (game diffs N+1, actual minutes; production o_3eff via shooting.talent_feature;
reproduces bt_split res_o/res_d exactly); code/bigs_score.py cols[,cols] (base box+.6res_o+.8res_d = 13.4759, dLOSO, rel coef
vs base, season-clustered t, halves 2015-19/2020-25); code/bigs_lamo.py -> bigs_lamo_{N} (separate offensive ridge);
data/bigs_spike_bt.parquet (latest-season vs window, from lineup_hist one_* = prod W, small leak).
impact.fit_window gained optional lam_o (default None = unchanged; impact_pre_lamo.py = old copy).
Results (rel = extra weight on the term relative to base; bar t>2.5 + same sign both halves):
- Residual weight for bigs: res_o_B rel -0.12 t -0.6, res_d_B -0.13 t -0.9; guards nil. Bigs' total rating x B -0.12 t -1.5.
  Persistence across NON-overlapping windows (N vs N+3, 2240 players): res_d slope 0.48 all, bigs -0.04 (t -0.6) -> bigs'
  defensive residual is as real as anyone's. Bigs carry a persistent +0.55 res_d intercept (t 7): box under-credits bigs.
- Young bigs (<=23): young_B +0.73 (t 0.9); persistence res_d:B:young -0.19 (t -1.4). Hint only.
- Backup-thinness leak (ctx): opposite sign to the hypothesis (+2.6, t 3.8, but halves 0.7 / 7.8) -> not the Clingan cause;
  odd, unexplained, not pursued.
- Small-sample offense: post-hoc res_o x thin rel -0.7 t -2.8, persistence t -2.9, BUT proper refit with heavier offensive
  ridge (lam_o 3000/4500/7000) gains only 0.0025 (t -1.2) -> REJECTED.
- Spacing for bigs (Sims): o_3vol_B rel -0.45 t -1.4 (direction = smaller penalty for bigs, not significant); o_rimvol_B,
  o_oreb_B nil. o_ftvol_B -0.73 t -3.1 alone (halves -1.1/-3.7) but -1.9..-2.5 jointly -> not robust. o_3eff_B with the
  production shooting prior: nil (the old-feature version looked significant - artifact).
- Latest-season spike (relevant: Clingan/Queta residuals come mostly from 2025-26: Clingan one_d 1.09 -> 2.21, one_o
  0.15 -> 1.21): adding single-season N residuals jointly: one_o -0.18 t -2.9 (-0.9/-3.5), one_d -0.13 t -2.6 (-1.0/-2.6),
  dLOSO -0.0045; no bigs interaction. Borderline; suggests a flatter window decay than (.3,.6,1). NEXT candidate test:
  refit windows with decay e.g. (.4,.7,1)/(.5,.8,1) -> game LOSO + preseason backtest. Would trim Clingan ~0.5.
- Side finding (not bigs): team-level concavity in game MARGINS (team_kink2 rel -0.65, t -8.8, both halves; player-level
  kink vanishes once included). Wins barely affected (logloss -0.0017; decile win% calibrated; preseason top sextile
  model 53.6 vs actual 52.4). Likely score effects/coasting + aging not in backtest base. Backlog, not applied.
Conclusion for user: data does not show bigs as a class (or young/small-sample bigs) over-rated; Clingan's rating is his
2025-26 lineup defense+offense, which the data says to trust about as much as anyone's. Sims low = own lineup results
(one_d -0.94 in 2025-26) + spacing penalty that data does not say to shrink for bigs.

## WINDOW DECAY TEST  [2026-09-22, session 12] -> box decay (.1,.4,1) APPLIED (user OK; see 'APPLIED' below)
Motivation: spike test above (latest season over-weighted?). Code: code/decay_bt.py (full refits, 5 decays for box+residual
together) -> data/decay_bt_{N}; code/decay_score.py; code/boxdecay_bt.py (box-only grid, no refit needed: fit_window's
prior uses per-season features, so the residual does NOT depend on the box decay) -> data/boxdecay_bt.parquet.
impact.fit_window gained box_decay=None (None = same as decay); preseason_backtest.py honours env BOX_DECAY="a,b,1".
- Joint decay (box+res): (.1,.4,1) 13.4694 (-0.0065, t -1.6) | (.4,.7,1) +0.0061 t +3.7 | (.5,.8,1) +0.0128 | (.7,.9,1)
  +0.0231 -> flatter is clearly WORSE; spike hypothesis REJECTED (latest season is not over-trusted).
- Residual decay alone: (.5,.8,1) +0.0007, (.1,.4,1) +0.0054 -> current (.3,.6,1) fine.
- BOX decay alone: (.1,.4,1) -0.0126, 9/11 seasons, t -2.8, halves -0.008/-0.016 (PASSES); (.05,.3,1) -0.0165 t -2.6;
  (0,.2,1) -0.0164 t -1.8; (.2,.5,1) -0.0064 t -3.1. Both off and def box prefer steeper (off gains more).
- Preseason team backtest (age_eval.py TAG): current model(minfix) 8.661, blend LOSO 8.144 | bd1_ (.1,.4,1) 8.643, 8.139 |
  bd01_ (.05,.3,1) 8.644, 8.139. Slope 1.00 all.
- Also seen: lineup weights re-optimised at any decay -> best w_off 0.4, w_def 0.7 (13.4666 vs 13.4759 at 0.6/0.8);
  known from session 6 (0.6 kept on purpose); not re-litigated.
- 2026 preview (box parts only, x slope 1.0501, re-centred): Clingan +0.11 (stays #5; his box improved in 2025-26),
  Wemby +0.44, Duren +0.37 (#23 -> #14), P. Reed +0.47, Cade +0.37, Avdija +0.37; AD -0.35, Embiid -0.35, Sabonis -0.29,
  Gobert -0.11. Approx model wins: DET +1.4, POR +1.0, SAS +0.7; HOU -1.0, IND -0.9, SAC -0.8 (blend moves ~1/3).
- TO APPLY: production.py fit_window(..., box_decay=(.1,.4,1)); preseason_backtest.py default BOX_DECAY .1,.4,1 ->
  regenerate pre_T/pre_P -> score_preseason -> minutes_age_fix -> refit_blend -> blend_curve/tau_model -> project ->
  simulate -> build_dashboard (check hist 'win' box arrays use the same decay) -> test_dashboard.js + ui_edit_test_db.py
  -> republish same link. Calibration slope/aging not refit (team slope stays 1.00).
Bigs conclusion unchanged: no test found a reason to lower Clingan; the one lead (one-season spike) failed properly.

APPLIED 2026-09-22 (session 12), box decay (.1,.4,1); lineup residual decay unchanged (.3,.6,1):
- production.py BOX_DECAY -> ratings_2025; preseason_backtest.py env BOX_DECAY default "0.1,0.4,1" -> pre_T/pre_P (identical
  to the bd1_ test run) -> score_preseason -> minutes_age_fix -> refit_blend: w_model 0.373, w_vegas 0.627, w_age -0.595,
  tau_blend 2.420; model RMSE 8.643, blend LOSO 8.139; tau_model 2.67 (score_preseason Model projection sd);
  blend_curve.json regenerated (recipe: in-sample RMSE at wm 0..1 step .05, w_age fixed; tau_i = tau_blend *
  sqrt(rmse_i^2-18.9)/sqrt(rmse0^2-18.9); rmse0 8.059, best .35).
- NOTE: run order matters: score_preseason REWRITES preseason_backtest.parquet (drops model_wpct_agefix) -> always
  follow with minutes_age_fix then refit_blend.
- project.py + simulate.py. min_bm/min_ba.npy (minutes3.py) and rookie_min_coef.npy (rookies.py; needs draftrepo clone
  + data/player_games_all.parquet, rebuilt from player_games 1996-2025 with build_seasons' first block only - do NOT run
  build_seasons.py whole, it rewrites player_seasons without bio columns) were missing from the zip; regenerated and
  VERIFIED: old ratings reproduce old players/teams projections exactly (max diff 0.0). Now in zip.
- backtest_components.parquet f:* (except f:o_3eff) rewritten for the new decay by code/bc_boxdecay.py (check mode with
  .3,.6,1 reproduces the old file to 1e-15). Base game LOSO 13.4759 -> 13.4633; dashboard backtest box 13.469 -> 13.457.
- build_dashboard (meta.box_decay added) -> test_dashboard.js (0.0011 wins) + ui_edit_test_db.py (pass) -> REPUBLISHED
  same link. Backups of everything pre-change: data/pre_bd/.
- 2026: Clingan +0.10 (#5 unchanged), Wemby +0.42, Duren +0.36 (#23 -> #14), Reed #24 -> #12, Cade +0.37; AD -0.36,
  Embiid -0.34, Gobert -0.12. Model wins: DET +1.3, POR +1.0, UTA +0.6; HOU -0.9, DEN/IND -0.8. Forecast moves <= 0.4.
- REPO (github.com/jvector26/JIMBO): pipeline/build_state.py now uses BOX_DECAY (.1,.4,1) for hist_box and KEEPS
  hand-tuned keys of an existing config.json (it used to reset min_k/min_rec/min_streak/min_alloc/notes to old defaults
  - would have silently reverted the session-11 minutes rule). state/2026 rebuilt; build_app.py -> site/. Local nightly
  dry run OK (pre_w vs sim_2026 blend within 0.9 = schedule/MC). PUSHED 2026-09-22 (a6ac631, user token, 7-day, Contents/Actions/
  Workflows RW, stored only in /home/claude/.ghtoken); nightly dispatched -> success (6039c5f), pages deployed; latest.json
  pre_w matches new sim within 0.9. (Patch kept for reference.) repo_push_box_decay.patch (binary git format-patch) is in the zip root; to push: clone repo,
  `git am repo_push_box_decay.patch` (or cherry-pick), push with a token, then dispatch nightly/pages. repo_snapshot/
  updated with the same files. In-season box update now blends against a steeper-decayed history (untested either
  way; box update is a small part of M4).

## GAME DETAIL PAGE (B1)  [2026-09-22, session 12] -> BUILT, TESTED, PUSHED (30cc1cb; nightly 7b6ae66 has the new fields)
Tap a game card on Tonight -> #gv page (app only; SEASON layer). Template code/dash_v2.html (pre-change copy
code/dash_v2_pre_gv.html). Navigation: history push like the player view; popstate/Esc close pv first, then gv; player
view opens over gv (body.pvopen hides #gv) and Back returns to gv; bottom nav closes both directly.
- tonightGames() refactored: gameStrength() (expected rotation per team: m_tg x status, edits, rescaled to 240) and per
  side parts {lin (per 100 vs league), pts = lin x pace/100, off (team offset), bl = gap x dec, m, rot, x}; sp = H.m - A.m
  + hca; spm/phm = model alone (no blend). Same numbers as before (card % = nightly p_home at day 0).
- Page: header (win %, records once played, tip), cards (JIMBO spread + model alone, market line/total, gap), margin
  distribution SVG (Normal(sp, 12.8); market marker; P(10+ either way), within 5, 80% range), waterfall table (edges sum
  to the spread; test checks), expected rotations both teams (regulars 8+ min, 'spot minutes' group, 'not expected to
  play' with injury status: detail, expected return / 'your edit' / 'outside the rotation'; tap -> player view, Edit ->
  editor, odds update live), context (rest days/back-to-back, earlier meetings, last-10 chips; labelled not used by
  JIMBO), pace note (one league pace; no JIMBO total).
- nightly.py: tonight += rest_h, rest_a (days since previous game; None = first game), meet [[date, home, away, hp, ap]];
  teams += l10 [[date, opp, H/A, pts, opp_pts]] (newest last); players with a status += inj (ESPN detail + side),
  ret (return date). Only factual fields (no ESPN comment text) since the site is public. Odds history not added.
- Tests: /tmp/gvtest.py (recreate: local http.server on a copy of gh/site with a test latest.json; playwright mobile
  390px + desktop; open Tonight -> game -> player -> back -> edit -> back; checks page errors and waterfall sum) with
  day-0 data (injected lines/injury/l10) and a 2025-26 replay as of 2026-01-15 (nightly --season 2025 --state
  state/2025_replay --parsed data --schedule-from-parsed). test_dashboard.js + ui_edit_test_db.py pass.
- claude.ai artifact not republished (no Tonight tab there; nothing visible changes).
Ideas left: line-movement history per game (live/odds/{date}.csv snapshots), true starters via a pre-tip run.

## CATEGORY WEIGHT REVIEW  [2026-09-23, session 13] -> NOTHING CHANGED (weights confirmed); eff-slider removal offered
User: dislikes the separate 'Shooting efficiency emphasis' slider; skeptical of ball_security weight; wants weights confirmed.
Code: code/wcal.py (loader/LOSO/clustered-t helpers on backtest_components + bt_split), code/tov_bt.py -> data/tov_bt.parquet
(TOV x usage/assist interactions from bigs_win windows, game diffs N+1; needs player_games 2015-2025 + stints 2012-2025:
rebuild.sh 2012 2025, ~2 min). Base = box + .6 res_o + .8 res_d with unrated multiplier free (backtest's fixed newcomer
values are stale, mult 2.05) = LOSO 13.4174.
- One category freed at a time (best mult, se, halves 15-19/20-25, dLOSO free | dLOSO if removed):
  three_pt .90 (.09) +.0003|+.054; rim .93 (.27) +.002|+.014; mid 1.53 (.49) +.001|+.009; foul .83 (.18) +.002|+.024;
  self_creation .99 (.59) +.006|+.007; playmaking 1.01 (.11) +.003|+.058; ball_security 1.03 (.18; 1.26/.83) +.002|+.042;
  oreb 1.33 (.53) +.002|+.006; dreb .47 (.35) +.0001|-.0003; rim_prot 1.25 (.27) +.003|+.042; perim 1.44 (.28; 1.59/1.35)
  +.0003|+.047; shooting-eff (4 accuracy terms together) 1.18 (.20; 1.76/.74) +.0025|+.039.
  No single freed weight improves out of sample; all 11 freed 13.4283 (worse), ridge toward current best 13.4211 (worse).
- Ball security: effective charge -0.74 pts/100 per extra TOV per 100 poss (x slope 1.05), below face value of a possession
  (~1.1+). Usage interaction nil (top-15% usage x TOV rel +0.20 t 0.7; continuous t 1.3); 'excess' TOV beyond usage vs
  usage-explained TOV: same value (rel 0.00). TOV x assist-rate z: rel +0.23 t 3.2 (3.4/1.8) = elite passers' TOVs cost MORE,
  but collinear with playmaking diminishing returns (ast x az t -2.9; jointly neither significant). Not applied.
  Most-penalized 2026: Trae -2.15, Cade -2.04, Ja -1.99, Luka -1.91, Harden -1.83 (offset by playmaking/self-creation/fouls).
- Efficiency is already inside three_pt / rim_finishing / pullup_mid / foul_drawing; the eff slider only re-scales it and
  data says 1.0 (t 0.9, halves disagree). Recommended: remove the slider (UI only; ratings unchanged at default).
APPLIED 2026-09-23: efficiency slider REMOVED (user: it was an early example, never wanted). dash_v2.html (pre-change copy
code/dash_v2_pre_noeff.html): slider + note gone, featMult ignores S.eff, DEF.eff removed, load() deletes saved sl.eff.
meta.eff still emitted by build_dashboard.py (unused). test_dashboard.js DEF.eff removed. Tests: engine 0.0011, db edit
test pass, 15 sliders, no page errors. Artifact republished (same link). App: build_app.py -> gh/site committed locally
(07feadc on top of nightly 99790c1) but NOT pushed (no token this session): repo_push_noeff.patch in zip root ->
clone repo, `git am repo_push_noeff.patch` (or just rerun build_dashboard.py + build_app.py), push with token, dispatch pages.

## MINUTES REVIEW: durability, calibration, game-level minutes  [2026-09-23, session 13] -> APPLIED + PUSHED
User: top guys projected ~2,400-2,500 min / 65-70 games looked too low; asked for (1) durability, (2) calibration of the
current minutes model, (3) a game-level minutes model (tonight's minutes were season averages with missed games baked in).
Data: player_games + stints 2012-2025 rebuilt this session (rebuild.sh 2012 2025, ~3 min; not in zip).
(1)+(2) code/minutes_cal.py -> data/minutes_cal.parquet: replays the CURRENT preseason minutes model each October 2014-25
  (mA/mB/aA/aB refit on seasons < S, age mult, allocation) incl. full-season absentees (pre_P lacks them). Scored excl 2019-20.
  - Top-30 projected per season: 2,559 proj vs 2,219 actual (games 71 vs 66; mpg 33.6 vs 33.2 = fine). Total-minutes
    calibration slope 0.79 (both halves); availability slope for 25+ mpg regulars 0.69 (2021-25: 0.57). The model was
    TOO HIGH at the top, not too low.
  - Durability (regulars, groups by worst of last 3 regular seasons): iron (all 90%+) proj 72 / act 68 games (2021-25: 71/63),
    51% reach 75; solid 68/65; some misses 63/61; fragile 55/58. History over-trusted, not under. Load-management era.
  - code/minutes_recal.py (variants, LOSO): piecewise-linear recal of min_model (incl. age mult) -> player min RMSE 597.7 ->
    583.3, top-30 2356 proj vs 2232 act, team RMSE 8.642 -> 8.639 (neutral). Availability-only recal 8.611 (t -1.1, n.s.).
    Thin-roster 'replacement fill' instead of scaling stars: team RMSE worse (8.71-9.08) -> rejected.
  APPLIED: code/min_recal_fit.py -> data/min_recal.json (c = 75.62, 0.932, -0.0066, -0.2964; knots 1000/2000; 2000 -> 1933,
  2800 -> 2436). project.py: min_model_raw kept; min_model = recal(min_model_raw); exp_games = min_model / mpg_when_playing
  (cap 82) so the age multiplier AND the recal sit in GAMES (dashboard mpg = raw mpg when playing; e.g. Edgecombe 68 g x 35.4
  instead of 71 x 39.1). Rotation players now avg 62 games (sd 5). Model wins moved HOU -2.1, OKC -1.0, SAS -0.9; UTA +1.3,
  BKN +1.2; forecast <= 0.7. NOT redone: preseason_backtest.py / blend refit (team RMSE neutral; blend weights kept).
(3) Game-level minutes: code/game_minutes_bt.py (oracle 'who played') and code/game_minutes_active.py ->
  data/game_minutes_active.parquet (every team game x rostered player; roster = appeared before and after; absence = 'out' if
  run of 3+ games or regular (played-only mpg 20+), else active DNP); code/game_minutes_active_score.py. Ratings = pre_P
  preseason, pace from pre_T, hca 2.6. Results 2014-25 (min RMSE incl DNPs | game margin RMSE, halves):
    oracle actual minutes 0.74 | 13.2612 (worse than predicted rotations: garbage time)
    current nightly rule (season avg per team game incl. zeros, x status, prop. rescale) 7.53 | 13.2649 (12.5407/13.9494)
    conditional k5 a.5, prior x age mult, bottom-trim 7.08 | 13.2247 (t -3.0; 12.5354/13.8777)
    conditional k5 a.5, prior RAW mpg_wp (no age mult), trim 7.07 | 13.2236 (t -3.3; 12.5316/13.8790)  <- CHOSEN
    (prop rescale 6.77 | 13.2584; 50/50 prop+trim 6.62 | 13.2353, but prop is sensitive to roster size in production)
  Age mult on the prior over-predicted young / under-predicted old early-season minutes; raw prior calibrated by age.
  PRODUCTION (pushed d0b20af): nightly.py m_c = w*(0.5*last5 played + 0.5*season played, current team) + (1-w)*mpg_wp,
  w = np/(np+game_k=5); overrides: g=0 -> 0, mpg -> locked; tonight_minutes(): out/suspension 0, each DTD plays with p .5
  (average over 2^n scenarios, n<=8, so a DTD star isn't trimmed as a 17-min bench guy), trim240 (bottom-trim, locks kept,
  scale up if short, cap 48). latest.json players += mc. build_state.py: players.parquet += mpg_wp (= mpg_when_playing, rookies
  min_proj/70). state config += game_k. App (dash_v2.html; pre-change code/dash_v2_pre_gamemin.html): trim240 + gameStrength
  same logic; game page shows 'N min if he plays' for DTD; notes updated. Tests: app cards = nightly p_home exactly (390px +
  desktop), no page errors; 2025 replay (asof 2026-01-15) nightly OK (Jokic back from injury: old 16 -> new 34 min).
  Day-0 examples: Maxey 30 -> 37, SGA 29 -> 33, Jokic 26 -> 34; BOS@DET DET 55% -> 46%, PHI@NYK 59 -> 56%.
  Season-sim minutes (m_tg rule, expected minutes incl. absences) unchanged - correct for win totals.
Repo: pushed d0b20af (efficiency slider removal + tonight minutes) and 75aadb9 (recal, state/app rebuilt). New token
(user, 2026-09-23) in /home/claude/.ghtoken only. Artifact republished (recal). Nightly dispatched after each push.
Note: nightly proj_w vs sim_2026 blend differs up to 1.75 (WAS: ESPN injury 'out' games) - pre-existing, not from this change.

## ROSTER-ROLE AUDIT + DEPTH-CHART MINUTES TEST  [2026-09-23, session 14] -> TESTED, NOTHING APPLIED (awaiting user choice)
User worry: structural role misses going into the season (Bruce Brown NYK, Harrison Barnes SAS). Nothing in production changed.
Knicks (web, 2026-09-23): Brown, Konchar, Agbaji, Eubanks, Wiseman all on NON-GUARANTEED Exhibit 9 deals; NYK has 13
guaranteed and ~$3M under 2nd apron -> realistically ONE of the five makes it. Reporting favors Wiseman (Fischer: "real chance",
Mike Brown fan; Stein: need a 3rd big; Robinson + Hukporti gone). JIMBO gives the five ~3,700 min (Brown 1,378 at -4.3 rating,
lineup residual -1.5). Keep-one scenarios (project.py allocation): all five 43.4 model wins | keep Wiseman 49.3 | Eubanks 49.0 |
Konchar 49.9 | Brown 46.2 (blend moves ~0.45x). Also: Brunson wrist + Diawara finger 'questionable for start' (CBS/RotoWire).
Offered (not done): move 4 invitees to FA (keep Wiseman placeholder) + audit other teams' camp invitees. Final cuts ~Oct 17-19.
Barnes: SAS signed Tobias Harris; beat coverage says Barnes drops out of the rotation. JIMBO 1,619 min. SAS had no RealGM chart
on 9-18 (role '?'). BUT today's RealGM chart lists Barnes as 1st backup SF (Rotation) -> RealGM would NOT fix Barnes or Brown.
Audit of JIMBO vs RealGM tiers (roles in rosters_2026.csv are used ONLY for no-history players in project.py):
- Ty Jerome (MEM starter, +2.8) projected 0 min: smallest-first bottom trim on a 21-man roster, his raw projection small after a
  38-game year. Lim-PT rookies get 1,000-1,800 (Boozer 1,835, Mikel Brown Jr. 1,712 @-4.1, Burries 1,332 @-4.3).
  SAS roster also carries Ja'Kobi Gillespie / Maliq Brown as -4.5 newcomers at 728 min each.
Data built:
- research/wayback_depth.py + .github/workflows/wayback-depth.yml (dispatch input `script`, default wayback_depth.py): CDX +
  fetch of archived RealGM league depth pages (and ESPN captures, unparsed) -> repo research/depth_wayback/{captures.csv,html/}.
  RealGM parsed 2015-2025 (no 2014): 30 snapshots, all 30 teams (2019-10-07 partial, dropped). Format stable (h2 "YYYY-YYYY Team
  Depth Chart", tr class depth_starters/rotation/limpt, td cells in PG..C order; player /player/Slug/Summary/rgmid).
- code/depth_parse.py -> data/depth_realgm.parquet (season, snap, team, player NBA id, tier S/R/L, row, col, depth_rank);
  name match 95.1% (norm = ascii letters only, suffixes dropped; team+last-name fallback; ALIAS dict). Unmatched = mostly camp
  bodies who never played.
- code/depth_test.py [early|late] -> data/depth_test_{snap}.parquet: minutes_cal.parquet (2015-25) baseline m0 = min_recal of
  min_model_age; tier N = not on his (opening) team's chart. LOSO by season, players scored excl 2019/2020, teams excl 2019.
    player-min RMSE | team-win RMSE (season-clustered t vs baseline)
    late snap (last pre-opener):  baseline 583.0 | 8.619;  trim-by-tier only 572.5 (t -5.6) | 8.651 (t +0.5);
      tier model 564.3 (t -2.6, both halves) | 8.618;  tier model+trim 565.1 | 8.669;  tier+rank+trim 562.1 | 8.660
    early snap (~4 wk out):  tier model 573.8 (t -3.3) | 8.595 (t -0.7); tier+rank+trim 572.1 | 8.618
  Big misses: 170 cases m0>=1200 & tier L/N: base 1,674, tier 1,313, actual 1,292. 48 starters with m0<800: 582 -> 970 vs 1,049.
  Tier means (late): S m0 1726 act 1838 | R 1106/1114 | L 737/641 | N 740/324. Team accuracy neutral (as with all minutes work).
  LIMITATION: backtest rosters = players who actually appeared, so camp-cut cases (Brown) cannot be tested.
- code/depth_fit.py -> data/depth_tier.json: want = a[t] + b[t]*m0 (pooled early+late): S 352+0.845m, R 188+0.842m,
  L 26+0.900m, N -97+0.630m (clip 0..3000), then the usual allocation.
- pipeline/depth_fetch.py + .github/workflows/depth-fetch.yml (MANUAL dispatch only, not in nightly): RealGM league page ->
  live/2026/depth/realgm_YYYYMMDD.html.gz + fetch_log.txt. RealGM is behind Cloudflare: runner IPs get 403 "Just a moment"
  intermittently (1st run all clients 403, 2nd run curl_cffi chrome 200) -> Wayback fallback (CDX latest; Save Page Now if
  >30h old). Sandbox cannot reach realgm or archive.org (web_fetch tool can read realgm page).
- code/depth_2026.py -> data/depth_2026_preview.csv: tier model on current projections with today's chart. Model-win moves:
  MEM +3.8 (Jerome 0 -> 1,045), ATL +3.0, GSW/LAC +1, DET -2.7, NYK -1.2, TOR/NOP -1.0. Barnes 1,619 -> 1,551, Brown
  1,378 -> 1,348, Boozer 1,835 -> 1,677, Mikel Brown 1,712 -> 1,567. Artifacts to fix before applying:
  * 'N' mixes roster staleness with role: RealGM already has today's trade Hield + Ryan Nembhard -> CHA (Dorian Finney-Smith ->
    ATL), Bez Mbeng on MIA -> treat "on another team's chart" as a ROSTER-UPDATE flag (move player), not a minutes cut.
  * Duren (DET) absent because unsigned RFA (Pistons offered 5y/$200M) -> exempt.
  * Name misses: "Herb Jones" (NOP), others -> extend ALIAS / match on RealGM id.
  * Diawara (NYK, L, finger) 768 -> 0 through bottom trim on the crowded (camp-inflated) roster.
- research/cdx_probe.py -> research/depth_wayback/probe.csv: archived preseason minutes-PROJECTION sources (Sep 15-Oct 25):
  FantasyPros /nba/projections/ captures 2015 (28), 2018-2023 (13-130/yr), none 2016-17/2024-25; Hashtag Basketball 0-3/yr;
  Basketball Monster, numberFire, RotoWire projections: none. RotoWire depth charts are paywalled (CBS depth page is RotoWire-fed
  but positional, players repeated).
DECIDED session 15: user chose (A), see 'DEPTH-CHART MINUTES APPLIED'. Was: (A) apply RealGM tier model now (fix name matching; 'other team' -> roster update; exempt unsigned;
nightly fetch + apply to preseason want and to the in-season prior), or (B) first test FantasyPros projected minutes (human-
curated, might catch Barnes-type role changes) with the same framework, ~7 seasons. Knicks camp-invitee cleanup also pending.
Token: user gave a new 7-day repo token 2026-09-23 (sandbox only, not in zip) -> ask again; remind user to revoke.

## DEPTH-CHART MINUTES APPLIED (option 1A)  [2026-09-23, session 15] -> APPLIED, PUSHED (7d4b7f6, 1e6d4c3), artifact republished
User chose A (apply RealGM tier model now); B (FantasyPros test) moved to backlog.
Sandbox pipeline (new order): build_rosters.py -> data/rosters_2026_BASE.csv (raw txt not in zip; base = old rosters_2026.csv)
-> code/depth_apply.py (latest gh/live/2026/depth/realgm_*.html.gz; config data/depth_config.json) -> data/rosters_2026.csv
(+ columns tier, depth_date; role = tier) + data/depth_2026_log.csv -> project.py -> newcomer_pos.py -> simulate.py ->
build_dashboard.py -> test_dashboard.js + ui_edit_test_db.py /home/claude/dashboard.html (absolute path!) -> gh: build_state.py
2026 -> build_app.py -> push. Pre-change copy: code/project_pre_depth.py.
- Shared module gh/model/depth_chart.py: parse (RealGM league page), load_chart (dedupe by RealGM id, highest listing),
  match (normalised name + ALIAS dict, then unique last name on same team; 555/567 matched, the 12 misses = no-history camp
  players, added as newcomers), tiers, want, allocate (= project.py rule).
- Roster rules: on another team's chart -> moved; NBA-history player on a chart but not rostered -> added; rostered but on
  NO chart -> dropped to FA (carry-over teams POR/SAC/SAS/UTA/WAS: 18 stale 2025-26 players; elsewhere Will Baker);
  exempt keys (depth_config.json: Duren, unsigned RFA) keep team and minutes (tier X); team with <13 chart players untouched.
  Today: moved Nembhard + Hield ATL->CHA, Mbeng CHA->MIA (listed MIA starter), Finney-Smith CHA->ATL; added Maluach, Spencer,
  Huntley (PHX), Payton (SAC), Josh Green (UTA) + 12 no-history L players.
- project.py: after min_recal, min_model_nodepth kept; min_model = a[tier] + b[tier]*m0 (depth_tier.json) for S/R/L/N;
  note += "depth chart: starter/rotation/limited minutes"; effect sits in mpg (dashboard g = exp_games unchanged).
  No-history newcomers: tier S/R -> m0 = backtest convention recal(400 x age mult) = 488 -> want ~764 / ~598 (was role
  heuristic 2000/1200); tier L -> old 250 kept, NOT tiered (backtest L newcomers act ~414 but only those who appeared;
  production L newcomers are mostly camp invitees). This L rule is a judgment call, untested.
- build_dashboard.py: rookie pick parsed by regex (note text now has a suffix).
- Effect: model wins MEM +3.8 (Jerome 0 -> 1,045), SAS +2.3 (camp newcomers 728 -> 0, Reed/Peter up), ATL +1.9, UTA/GSW/
  LAC/SAC ~+1; PHX -2.6 (thin roster: added Spencer/Peat take scaled-up minutes), MIA -2.0 (Mbeng S), NYK/LAL -1.2.
  Forecast (blend) moves <=1.1: MEM +1.1, ATL +0.9, SAS +0.8; PHX -0.7, CHA -0.6, NYK -0.5.
  Barnes 1,619 -> 1,551; Harris 1,597 -> 1,702; Boozer 1,835 -> 1,677; Diawara 768 -> 0 (L, crowded NYK); Duren unchanged.
  Tests: engine 0.0010 wins, sim within 0.37; db edit test pass; app smoke (390px) no page errors.
- State/nightly: players.parquet += m0, tier, newc; config.depth {coef, exempt, min_team_players 13, newc_L = recal(250),
  state_chart}. nightly.py (block before ratings) re-applies the latest chart dated <= min(asof, opener-1) with the same
  rules (moves, FA->team adds, no chart -> FA, tiers, allocation) and writes latest.json "depth" {chart, moved, added,
  dropped}. Parity: state chart reproduces state min_proj to 1e-12; synthetic chart test (Barnes removed, Love added) OK.
  Nightly minutes then flow through the existing in-season prior (pre_tg = min_proj/82). Chart-driven model changes move
  the nightly forecast 1:1 (same convention as overrides; see backlog). Tonight's game minutes (mpg_wp) NOT tier-adjusted.
- nightly.yml: step "Depth chart" = depth_fetch.py --until 2026-10-19 (continue-on-error). depth_fetch now 3 rounds 60 s
  apart: RealGM Cloudflare 403 on the first nightly (all clients), Wayback had no capture (Save Page Now 500). Two runs/day;
  missing days just keep the last chart. Check live/2026/depth/fetch_log.txt. If fetching keeps failing, Claude's web_fetch
  tool can read the RealGM page -> save as live/2026/depth/realgm_YYYYMMDD.html.gz by hand.
- First Actions nightly with the change (run 35908880170, e0ffce9): success; latest.json depth chart 20260923, no changes.

## MASTER BACKLOG  [consolidated 2026-09-23, session 15; user: "capture the rest in the backlog"]  <- START HERE
Time-sensitive (before opener 2026-10-20):
 1. CAMP-INVITEE CLEANUP (final cuts ~Oct 17-19). NYK: Brown/Konchar/Agbaji/Eubanks/Wiseman on Exhibit 9, ~1 makes it
    (keep-one scenarios in session 14 section). Audit other teams' non-guaranteed / Exhibit 10 players. The daily chart now
    drops players who vanish from RealGM, which may handle cuts automatically -> check after cuts before overriding by hand.
 2. Watch depth fetch reliability (fetch_log.txt); manual fallback above. Remove Duren from exempt once he signs (or leave:
    harmless while he is on DET's chart).
 3. IN-SEASON TUNING: multi-season nightly replay 2014-25 to tune gap_k (forecast decay toward model), tau curve, coverage
    (session 10 NEXT item 1; minutes rule already tuned in session 11).
 4. OPENING-WEEK CHECK: first real games parse, results lock, TBD NBA-Cup fill, tonight cards vs ESPN lines.
Depth-chart follow-ups:
 5. FANTASYPROS minutes test (option B): archived preseason projections 2015, 2018-23 (research/depth_wayback/probe.csv);
    same framework as depth_test.py; could catch Barnes-type role changes RealGM misses.
 6. Tier for tonight's minutes (mpg_wp prior) - untested; and L-tier newcomer value (250) - untested judgment call.
 7. Forecast response to chart/override changes: nightly moves forecast 1:1 with the model; the blend would say ~w_model
    (0.37) if Vegas had also seen the news. Test or pick a rule.
 8. Thin-roster scale-up gives camp players real minutes (PHX Spencer 751, Peat 589). alloc_test.py floor 0.25 option also
    still offered (tie on teams, better player minutes).
App features:
 9. Line-movement history per game (live/{S}/odds/{date}.csv snapshots).  10. Pre-tip run (~30 min before tip) for late
    injuries / starters.  11. Sim in a Web Worker.  12. In-season hustle refresh.  13. Push notifications (deferred).
Model research (bar: game LOSO + preseason team backtest, t>2.5, same sign both halves):
 14. B3 COACHING EFFECT (plan in its section).  15. B2a UI: split 3PT bar into volume (spacing) vs accuracy (cheap).
 16. College prior for 2nd-round / undrafted rookies and 2nd/3rd-year players.  17. Talent 3P% in stint luck adjustment.
 18. Team-level margin concavity (bigs section side finding).  19. B2b/c position-specific spacing, rim/OREB weights.
 20. NCAA stage-A + stage-B refit under new age curve / current features (NOTE session 16: rerunning ncaa_model.py alone
     already shifts cscores up to 0.66 - refit A and B together, never A alone); non-college rookie scale (x1.16-1.26 hint).
 21. (session 16) Young x rating age-bump interaction: right sign everywhere, t -1.7 game / -1.2 team -> retest after 2026-27.
 22. (session 16) App: role edits + team move in the same edit - nightly applies tier before the team override (edge case).
Pipeline-dependent signals (need NBA.com from runners): rest/travel/altitude for game odds; better same-day lineups.
Token: user gave a new token 2026-09-24 (session 16; sandbox only /home/claude/.ghtoken, used for d65f69e/35073db/cbe208e).
Ask again next chat; remind user to revoke.

## LINEUP PORTABILITY CHECK (Jakucionis question)  [2026-09-23, session 16] -> REJECTED at game level (see GAME TEST below)
Jakucionis (MIL, 20.7): rating +3.12 = box -0.13 + hustle +0.50 + lineup +1.83 (raw res_o 1.08/res_d 1.36 on ~1,980 poss)
+ age +0.92. Highest on MIL (KPJ +1.76). Note Herro, Jaquez, Ware were also his MIA teammates.
code/lineup_persist.py -> data/lineup_persist.parquet: next-season single-season residual (0.6o+0.8d) on window residual,
1998-2024, min>=300 in N, >=500 in N+1, weighted by N+1 min, season-bootstrap SE. Slope all 0.37; window poss <2.5k 0.24,
7k+ 0.39; team changers 0.26 vs stayers 0.39; thin+moved 0.085 (n 129, se 0.09). Joint: x 0.40, x*thin -0.09,
x*moved -0.11, same sign both halves (1998-2012 / 2013-24). => ridge under-shrinks thin samples and team changers;
thin+moved carries ~half the usual. Also the flat <=21 age bump overstates for already-high window ratings (age_check:
implied ~+0.2 at his window vs +0.88). Rough fair value ~+1.5 to +2.0. Candidate model change: lineup weight x
f(window poss, moved) -> needs game LOSO + preseason team backtest. Offered user an override meanwhile.
GAME TEST (user asked to run it): code/lineup_port_bt.py (G = player-game rows 2015-25 from game_minutes + bt_split_ratings
window residuals; reproduces bt_split res_o r .9999, res_d r .993; P = decayed (.3,.6,1) on_poss; moved = game team != last
team through N), lineup_port_fit.py, lineup_port_boost.py. m = (1 - a*moved) * P/(P+c) normalised at 8,000 poss.
Base (fixed .6/.8) LOSO 13.4577. EVERY discount worse: moved a .2/.35/.5 +0.001/+0.005/+0.010; c 1000/2000/4000 +0.008/
+0.012/+0.018; free-weight versions same. Free group coefs (relative to box, cut 2,500): stay-thick 0.70, stay-thin 1.05,
moved-thick 0.79, moved-thin 1.58 (t 2.6) -> thin/moved residuals are worth AT LEAST full weight at team level; 4-way split
LOSO 13.4588 vs pooled 13.4545. Up-weighting thin x1.2-1.4: -0.0015, sign flips by half -> not applied either.
Why player-level disagreed: single-season RAPM target splits credit with new teammates; not a target-shrinkage artifact
(gap holds for N+1 min 1500+). Team-level preseason test NOT run (game level failed the bar). Jakucionis stays +3.12.
Remaining lever for young high-window players: age bump interacting with window rating (age_check: <=21 act = .69 + .76*win).

## YOUNG x RATING AGE-BUMP INTERACTION  [2026-09-23, session 16] -> FAILS BAR (suggestive), NOTHING APPLIED
code/young_int.py [game | team k agemax TAG]: vets aged <= agemax get rating += SLOPE*k*(win - c), win = pre-age window
rating backed out of pre_P ((rating - returner)/SLOPE - bin), c = season's minutes-weighted mean win of those young vets
(mean-preserving: good young players lose, bad young players gain). Young vets 2014-25: n 344, win mean -1.56, sd 1.70.
- Player level (age_check.py rerun): act = a + b*win has b 0.73 for <=21 vs 0.69-0.79 every other bin -> no young-specific
  extra regression; the old '+0.3 implied at +1.7' was general regression the game-level slope (1.05) already rejects.
- Game LOSO (base incl. age curve 13.4740): <=21 k -.1/-.25/-.4: -0.0027/-0.0051/-0.0058, both halves same sign, but
  only 5-6/11 seasons better; free coef -0.38 (t -1.7; halves -0.47 / -0.35). <=23: nothing (free coef 0.00) or worse.
- Team preseason (age_eval.py TAG yi01_/yi025_/yi04_): model(minfix) 8.643 -> 8.628 / 8.612 / 8.602; blend LOSO 8.139 ->
  8.130 (k -.4); 7/11 seasons better, season-clustered t -1.1 to -1.3, halves same sign.
- Verdict: right direction everywhere, not significant (bar t>2.5). Jakucionis would be ~+2.3 (k -.25) / ~+1.8 (k -.4).
  Backlog: retest after 2026-27 adds a season. TAG files yi*_pre_P/T_* left in data/ (disposable).

## ROLE (DEPTH-TIER) OVERRIDE IN THE EDIT SCREEN  [2026-09-23, session 16] -> BUILT, TESTED; artifact republished;
## repo commit d65f69e PUSHED 2026-09-24; nightly dispatched (success, latest.json has dt/dc), pages deployed
User: wants to override a player's depth-chart role himself in Edit player (NOT a Q&A with Claude - don't quiz him).
- overrides.json players[key].d = S/R/L/N (Starter / Rotation / Limited minutes / Out of rotation); '' = follow chart.
- Page (code/dash_v2.html; pre-change copy dash_v2_pre_role.html): Edit has a Role select ("Depth chart: <tier>" default).
  Engine wantOf(p,o) = a[d] + b[d]*p.m0 (clip 0..3000; L newcomer -> newc_L), used in project() allocation; exported
  wantOf + allocate. Typed games/minutes still lock and win. Changing Role/Team previews games/minutes in the editor.
  Season layer: nightly mr already contains the nightly's tier (z.dt); if app role (o.d || z.dc) differs, every teammate's
  preseason share moves by (1-w)*(alloc with app roles - alloc with nightly tiers)/82, w = n/(n+5), n = team games;
  roster sorted by mraw first so trim ties follow the nightly's outcome. Changes list shows "role: starter"; minutes cells
  highlighted for role edits.
- build_dashboard.py (pre-change copy build_dashboard_pre_role.py): players += m0 (newcomers: recal(400 x age mult)),
  dt (chart tier; L newcomers stored as X in project.py shown as L), nc; meta.depth {coef, newc_L}.
- nightly.py: loads overrides d before the depth block; user tiers replace chart tier (also exempt X) before want/allocation;
  PL tier_chart kept; latest.json players += dt (tier used), dc (chart tier); depth.user_tiers list.
- Tests: engine vs project.py 0.0010 (unchanged); role overrides in-page vs project.py (depth_apply tier_override) wins 0.0010,
  player minutes 0.22; nightly parity without overrides exact; nightly with overrides = project.py (Jakucionis S 1,325,
  Porzingis L 915); app day-0 vs nightly: no edits / edits not yet in nightly / edits applied / edits removed but nightly
  still has them: all 0.006 (= baseline parity). UI (iPhone 13): select, preview, save, reopen, reset OK, no page errors.
  ui_edit_test_db.py pass. Test scripts /tmp/approle.py, /tmp/test_role.js (recreate if needed).
- Sandbox depth_apply.py also supports depth_config.json tier_override (key -> {tier}); left EMPTY (app/overrides.json is the
  single source). The 3 Q&A answers (Jakucionis R, Jerome S, Edey S) matched the chart and were discarded.
- (done) push + nightly. Note: before a nightly with the new code ran, the app ignored role edits in the season layer; since the new nightly
  runs, the app's role edits are ignored in the season layer (no z.dt) but still move the preseason engine/gap.
- FIX 2026-09-24 (35073db, pushed, pages deployed): user saw role edits change GAMES (sometimes >82), not mpg. Cause: season
  gmv() held mpg = nightly mpg_est and put every minutes change into games (g = gp + min/me) - affected all minute shifts in
  the app, not just roles. Now: games left played = (team games left - miss) x av (or o.g/82 x gr), mpg = min / that,
  g = min(82, gp + games left); fallback min/me if availability ~0. Pre-change template code/dash_v2_pre_gmv.html.
  Tests: Jakucionis R->S 57 g x 20.3 -> 57 x 23.3; Porzingis S->L 40 x 30.7 -> 40 x 23.7; mpg lock round trip OK;
  app/nightly parity tests unchanged (0.006). Preseason-only artifact already used g = exp games, mpg = min/g (unchanged).

## SORTABLE STANDINGS  [2026-09-24, session 16]  (cbe208e pushed; artifact republished; pre-change template dash_v2_pre_stsort.html)
- Standings controls row (#sttools): Conferences | All teams; "Sort by <column>" menu; direction button (↓/↑; label text on
  desktop). Column headers are buttons (data-stsort): tap sorts high->low, tap again reverses. Ties -> forecast.
- Columns (ST_COLS): Record (in season), Forecast, Model, Line, Age adj (pa+padj), Model vs line, Over line, Playoffs, Title,
  Net rtg, NEW Off rtg, Def rtg, Avg age (min-weighted) - new ones hide-m, but the sorted column is always shown; on phones
  (<=640px) the sorted column moves right after Team. All teams view tags each team E/W. Playoff/play-in cut lines only in
  Conferences + Forecast high->low. Choice saved in localStorage 'jimbo-standings-sort' {view,k,dir}. Team row expand works.
- Tests: /tmp/sttest.py (phone app + desktop artifact: sort, reverse, header tap, expand, persistence, no page errors);
  engine parity 0.0010, ui_edit_test_db, app/nightly role parity 0.006 unchanged.

## DYBANTSA vs BOOZER EXPLAINED  [2026-09-24, session 16] -> no change
Ratings: Boozer +0.71 (cscore 0.31, AT THE CAP: col_mean -0.994 + 1), Dybantsa -2.65 (cscore -0.86). Gap 3.4 pts.
Decomposition (ncaa_model.explain, fresh refit, cscore gap 0.91): ORB +0.30, AST +0.17, STL +0.16, TS +0.14, BLK +0.10,
DRB +0.07, age (18 vs 19) +0.07, TOV +0.05; offsets 3P% -0.10, PF/MPG/PTS -0.07. Points/100 and usage have ~0 weight.
Calibration of cscore within top picks holds (y on cscore: all 0.99, picks 1-10 1.16, 1-5 1.06 (se .23)).
Comps (usg 32+, low stocks): Trae, Ivey, Morrison, Edwards, Barrett, Cade, Harper, Mayo, Russell, J. Brown.
WARNING: rerunning code/ncaa_model.py today changes ncaa_scores.parquet (max |d cscore| 0.66; Boozer 0.31->0.09, Dybantsa
-0.86->-0.82) - target (features/weights) moved since it was fit. Production file restored from backup. Refit stage A+B
together (backlog 20) before touching it.


## SESSION 16 SUMMARY  [2026-09-23/24]  <- READ FIRST, then MASTER BACKLOG
Setup: unzip to /home/claude; the live repo is github.com/jvector26/JIMBO (public): `git clone` it to /home/claude/gh
(depth_apply.py and build_app.py expect /home/claude/gh; repo_snapshot/ in the zip is a copy at commit cbe208e, no .git).
Push needs a fine-grained token (Contents + Workflows write) from the user; `git pull --rebase` first (nightly commits).
Session 16 work (details in the sections above):
 - Jakucionis question: lineup portability (sample size / team change) REJECTED at game level; young x rating age bump
   suggestive but fails bar. Model unchanged; he stays ~+3.1 (MIL).
 - User does NOT want Claude to quiz him on player roles; he edits in the app. Built Role override (Edit player: Starter /
   Rotation / Limited / Out of rotation), overrides.json players[key].d, nightly applies it, latest.json has dt/dc.
 - Fixed season-mode games/mpg display (minutes changes now show in mpg; games <= 82).
 - Sortable standings (headers + Sort by menu, reverse, Conferences / All teams, Off/Def/Avg age columns, remembered).
 - Explained Dybantsa (-2.65) vs Boozer (+0.71, capped): activity stats (ORB/STL/BLK/AST) + efficiency, not scoring volume.
Deployed: repo main at cbe208e (+ nightly commits); Pages live; claude.ai artifact republished (same link).
Tests to rerun after template changes: node code/test_dashboard.js; python3 code/ui_edit_test_db.py dashboard.html;
code/tests/approle.py (needs /tmp/n1,/tmp/n2 nightly outputs: see file header), code/tests/sttest.py, code/tests/test_role.js.

## JRUE HOLIDAY / RIM PROTECTION FOR SMALL PLAYERS  [2026-09-24, session 17] -> NOTHING CHANGED (display option offered)
User: Jrue looks like an average defender (good perimeter, offset by a rim-protection penalty); why grade guards on rim protection?
Jrue (ratings_2025, window 2023-25): rim_protection -0.18 = blocks -0.30 + opp rim FG% (d_opprim) +0.12 (helps him);
perimeter +0.19; dreb -0.16 -> box def ~-0.15. Lineup defense res_d +1.90 (x0.8 = +1.52); win_d >= +0.7 every window since
2011, +2.0..2.4 at peaks. Projected def +1.45 pre-age (14th of 127 guards, 63rd of 290 with 1000+ proj min), +1.01 after age 36.6
(23rd of guards). Positional averages 2023-25 (800+ min): rim_prot PG -0.24 / SG -0.17 / SF -0.13 / PF +0.11 / C +0.62;
perimeter PG +0.10 .. C -0.15; opp-rim part ~0 at every position (blocks make the gap). Jrue is slightly above PG average.
Test: code/guard_rim_bt.py (bt_split_ratings windows x window pos_est from positions.parquet, game_minutes N+1, scaled to
backtest units: corr with f:d_blk+f:d_opprim 0.996) -> data/guard_rim_bt.parquet; code/guard_rim_score.py. Base LOSO 13.4633.
Extra weight relative to base (se; halves 15-19/20-25):
- guards' rim_protection -0.22 (0.43; +0.26/-0.65); forcing 0 for guards: dLOSO +0.0035 (worse).
- within-guard deviation from guard average +0.21 (0.44); dropping it +0.0086 -> shot-blocking guards' extra value is real.
- positional re-centering of rim_protection: -0.20 (0.72) = unidentifiable (lineups balance positions); full re-center +0.0028.
- re-centering ALL box defense by position: +0.0256 (coef -0.06, se 0.29) -> big-vs-guard defensive gap is real (center tilt).
- bigs' perimeter_def / guards' perimeter_def: nil. (Defense #4 earlier: per-stat values don't differ by position.)
Conclusion: guard 'rim penalty' is a positional baseline (model is position-blind, centred on all players); team forecasts are
essentially invariant to it; keep. Offered: display-only 'vs position' baselines for the player-view component bars
(Players tab already has Defense vs position for Def/D-reb/Stops).

## LINEUP IMPACT SPLIT INTO OFFENSE / DEFENSE (display)  [2026-09-24, session 17] -> PUSHED (e3b01c5), pages deployed, artifact republished
User wanted lineup ratings shown as offense and defense separately everywhere (table Off/Def columns already included them).
dash_v2.html (pre-change copy code/dash_v2_pre_lusplit.html): rawRating parts.lineup -> parts.lineup_o / parts.lineup_d
(no parts.lineup any more); COMPS/PARTS use lineup_o, lineup_d (bar, detail row, 'By skill'); colors --c-lineup_o
(#A3AEC4 base, #A99BD6 90s) / --c-lineup_d (= old lineup color). Rating build table: two lineup rows. histSeasons adds
l1o/l1d/lwo/lwd. Career chart: 'all' and 'od' stack lineup O and D (od = off box, off lineup, def box(+hustle on proj),
def lineup, age/return); 'lineup' mode stacks O/D per season with net dot + gold 3-season total line. Career table via
headCols/projCells/rowCells: od = Off(incl lineup) | L off | Def(incl lineup) | L def | Total; lineup = Box | L off | L def |
3-season off | 3-season def; all = Box | L off | L def | Total. Phones: Age/Games/Min hidden, short labels (Yr, 'YY, Proj,
L off/L def, 3y off/3y def) via .ctab .lg/.sh spans.
Tests: test_dashboard.js 0.0010; ui_edit_test_db.py pass (needs ABSOLUTE path arg); /tmp/lutest.py, /tmp/lushot.py (player view
modes desktop + iPhone 13, no page errors).
SCHEDULE PIN: build_dashboard's SIM.make_schedule(default_rng(2026)) gave a DIFFERENT schedule in this sandbox (348 of 2460
entries) -> would have shifted preseason sim numbers. Published schedule saved to data/dash_sched_2026.json; build_dashboard.py
now loads it if present. Keep that file.
Token: user gave a new token 2026-09-24 (session 17; /home/claude/.ghtoken only). Remind user to revoke.

## LINEUP SLIDERS SHOWN RELATIVE TO DEFAULTS  [2026-09-25, session 18] -> artifact republished; app commit 51bceeb LOCAL (not pushed: no token)
User disliked seeing 0.6 / 0.8 as slider defaults. UI-only change in dash_v2.html (pre-change copy code/dash_v2_pre_lwrel.html):
lineup_o/lineup_d sliders display x(S/M.lw_*) with range 0-2 (effective 0-1.2 off, 0-1.6 def; old raw range 0-1.5); S, saved
settings (db settings/current, localStorage, overrides.json) and the engine keep EFFECTIVE weights -> no migration, old saves
at 0.6/0.8 show x1.00. Rating-build rows, lineup-mode note, slider note, How-it-works text now say x / 60% / 80%.
Model numbers unchanged (backtest 13.457). Tests: test_dashboard.js, ui_edit_test_db.py pass; /tmp/lwtest.py (slider x1.5 def
-> S 1.2). To deploy the app: clone repo to /home/claude/gh, `git am` outputs/0001-lineup-sliders-...patch (or rerun
build_dashboard.py + build_app.py), pull --rebase, push with a user token.

## SORTED SKILLS + WIN DISTRIBUTION  [2026-09-28, session 19] -> artifact republished; app commit c63765d PUSHED, pages deployed
User asked: (1) player-view skill list ordered best to worst (was by absolute size) + visual cleanup; (2) team view histogram of
simulated wins. Choices (user): list includes EVERYTHING (12 skills + hustle + lineup off/def + age/returner/season);
histogram shows forecast AND model-alone outline; ship to artifact + app.
- dash_v2.html (pre-change copy code/dash_v2_pre_hist.html): skillBars(pa) replaces the 'By skill (largest first)' .rk2 grid:
  "Strengths and weaknesses", sorted by value desc, diverging bars around a zero line (zero placed by max pos/neg), component
  colors, dashed divider between positives and negatives; phones stack label+value over a full-width bar. Hustle shortened in
  this list only ('Hustle (loose balls, contests)').
- Engine simulate(): returns per team hist (counts 0..82 of final wins) and n; opt.noPost skips play-in/playoffs.
  runSim: after SIMB, a second sim SIMM = model margins (x.margin), tau = M.blend.tau_model (in season SZ.tau*tau_model/tau),
  regular season only; re-renders standings if a team is open.
- winsHist(x) at top of the expanded team view: bars = forecast sim (magenta over the line, pink under, grey push),
  purple step outline = model alone, dashed line at the betting line, dot/ring at the standings forecast/model values
  (sim means differ ~0.3 by MC/nonlinearity), readout = middle 80%, most common, P(over); hover/tap a bar -> share of seasons,
  model-alone share, cumulative. HTML-positioned labels over a preserveAspectRatio=none SVG (non-scaling strokes).
- Built by injecting the published payload (data/_dash_payload.json, identical to repo site/data/dash.json) into the new
  template, NOT by rerunning build_dashboard.py. Tests: test_dashboard.js 0.0010 (unchanged), ui_edit_test_db.py pass,
  screenshots desktop + iPhone (/tmp/shots_h.py, /tmp/apph.py), app mode with nightly latest.json OK, no page errors.
- To deploy app: cd /home/claude/gh, git pull --rebase, push c63765d with a user token (Pages deploys on push to site/**).
Token: user gave a new token 2026-09-28 (session 19; /home/claude/.ghtoken only). Remind user to revoke.

## CHEMISTRY TESTS 1+2  [2026-09-28, session 19] -> NOTHING APPLIED; one candidate found (NEW-TEAM PENALTY), awaiting user
User: "teams are not just the sum of their parts" - asked for (1) does the team leftover follow the core, (2) familiarity.
Code code/chem/: chem_team.py, chem_team2.py (team level; data/chem_team*.parquet), chem_stints.py (-> data/chem_stints.parquet,
665k stint sides 2015-25 with familiarity built from stints 2011-25; stints rebuilt this session via rebuild.sh 2011 2025, ~12 s/season,
not in zip), chem_fit.py (WLS poss-weighted, cluster offense team-season; base = box/lineup-residual sums of window N-1 from
bigs_win_{N-1}, unrated counts, age, log window poss, home, season FE; option team-season FE both sides), chem_movers.py
(-> data/chem_stints_movers.parquet).
TEST 1 (team residual = actual - additive prediction w/ actual minutes, backtest_components+bt_split, 300 team-seasons 2016-25):
- Persistence of team residual N -> N+1: 0.16 (t 3.5). Split franchise vs stayers vs movers' carried residual: unidentifiable
  (movers term se ~0.5). Continuity main effect +5.4 pts/game per unit share (t 3.0).
- Decomposed vs stayers (full share): established movers (>=500 min elsewhere) -2.9 (t -2.3), first half -2.35 (t -2.6),
  halves 2016-20 -2.35 / 2021-25 -1.87. Rookies -7.3, other newcomers -13: backtest uses a flat 'unrated' value (production
  uses college/pick priors) + tanking -> NOT chemistry. Second-half gap somewhat smaller (-3.3 -> -2.3, ns).
TEST 2 (stints): prior-2-season pair familiarity alone +0.19 off / -0.15 def per sd (t ~1.8); beyond players' own tenure with the
  team: nil. Prior tenure alone +0.25/-0.23 per sd (t 2.1/2.9) but defense sign flips by half. In-season familiarity/tenure
  measures are contaminated by coaches' playing-time choices (in-season minutes reveal quality) + season trend -> unusable
  (offense tenure +1.4/sd, defense pair -0.9/sd, wrong-sign partners).
- NEW-TEAM PENALTY (stint level): each established mover in a lineup -0.24 off (t -2.7) and +0.23 allowed (t +3.7) = ~-0.47
  pts/100 per mover; within team-season FE -0.27/+0.26; offense same both halves, defense only 2020-25 (+0.01 / +0.60).
  Matches test 1 (~-0.5 to -0.6 per player). Fade with calendar time on new team: +0.29/-0.26 per season-share (t 1.4/-1.3),
  none with team FE -> not established. Consistent with lineup-portability test (residual WEIGHT for movers fine; this is a LEVEL shift).
  ~0.8 wins per full-time mover (2,400 min). Not separated: chemistry/system learning vs adverse selection (teams let go of
  players they know are slipping) vs old-team context in carried ratings.
Offered next: (a) year-2 check (movers' 2nd season with the new team vs 1st: rebound = familiarity; no rebound = selection);
(b) standard bar: game LOSO + preseason team backtest + blend with a mover term (MIA/PHI/HOU/TOR/POR/WAS heavy movers in 2026).
FOLLOW-UP (same session, user asked for both):
(a) YEAR-TWO CHECK (code/chem/chem_year2.py -> data/chem_stints_yr2.parquet; yr2 = joined T during S-1 with 300+ poss in S-2):
  familiarity predicts yr2 BEATS its (year-1-depressed) rating (~+0.25); selection/old-context predict still below (~-0.25).
  Stint: net per player yr1 -0.56, yr2 -0.32 (offense part persists -0.30 t -2.7; defense part gone +0.02); team FE -0.56/-0.17;
  halves yr2 -0.33 / -0.28. Game level yr2 k2 +0.03 (t 0.1). => NO rebound: not 'settling in'; drop looks real (selection)
  or partly noise. Only hint of learning: defensive part vanishes in year 2.
(b) BAR (code/chem/mover_bar.py -> data/mover_games.parquet): game-level k1 -0.46 per mover full-game (season-cl t -2.9),
  matches stint -0.47. LOSO RMSE base 13.4633 -> 13.4556, but 2015-19 +0.0045 worse / 2020-25 -0.0169; half coefs -0.15/-0.73.
  Preseason (age_eval.py TAG, pre_P rating += d for vet movers on opening team): d -.25 model(minfix) 8.643->8.637, blend
  8.139->8.143; -.5 8.646/8.149; -.75 8.670/8.156; +yr2 -.5/-.5 8.636/8.137 (noise). FAILS BAR, NOT APPLIED.
  2026 illustration at -0.46 (zero-summed): LAL -1.6, PHI -1.5, CHA -1.2, MIA -1.0, MIL -0.8 ... OKC +1.4, SAC/CLE/ORL +0.8.
  Backlog: retest after 2026-27 (heavy-movement year); in-season RAPM update absorbs it after ~20 games anyway.

## MOVER CONTEXT TEST: does raw opp shot profile carry over to a new team?  [2026-10-06]  -> REJECTED (nothing applied)
Prompt: Peyton Watson (DEN -> CLE) perimeter_def -0.41, of which d_oppxq -0.33; d_oppxq is ~half team context (DEN players
all negative; DET/ORL all positive). DEFENSE #2 kept raw d_oppxq because context persists for stayers; question here was
whether it should be discounted for players who change teams.
Code: code/mover_context.py -> data/mover_context.parquet. Splits each game's f:d_oppxq / f:d_opprim diff into a mover part
by (1 - share of the player's season-N minutes with his N+1 team) (no season-N minutes = stayer); reconstruction of f: cols
exact (R2 1.0000). Movers = 26.7% of minutes. Base (box + .6 res_o + .8 res_d) game LOSO 13.4633.
Free mover term (coef relative to base; -1 = movers' value worthless, 0 = carries over fully): oppxq +0.10 (t 0.1; halves
-0.09/+0.28), opprim +0.39 (t 0.5), both +0.24 (t 0.5). Fixed discounts all worse: drop 50% oppxq 13.4640, 100% 13.4657.
=> movers' on-court opponent shot profile predicts as well as stayers'. Low power (se of ratio ~1), but point estimate
says no discount. Watson stays as rated; even a full discount would be ~+0.33 pts/100 for him (~0.35 CLE wins).

## SESSION 20 SUMMARY  [2026-10-06]
- Peyton Watson (DEN -> CLE) question: why is his perimeter defense bad? perimeter_def -0.41 = steals -0.07, fouls -0.01,
  d_oppxq -0.33 (raw on-court opponent shot profile; ~half team context: every DEN player negative, DET/ORL positive).
  His defense shows up as blocks (29th of 327) under rim_protection +0.50, hustle +0.26; total defense ~+0.35 (32nd of 109
  wings). Mover-discount test above (MOVER CONTEXT TEST) rejected -> rating unchanged; user can override.
- ZIP SLIMMED: 148 MB / 900 files -> 21 MB / ~280 files. Core keep-list traced at runtime (audit hook on open()) over the
  production chain + build_state + build_app + bt_split_score + age_eval; parquet rewritten with zstd 19 (same data);
  game_minutes and ncaa player_box slimmed to needed columns. Verified: chain on the slim tree vs the full 09-28 tree gives
  byte-identical rosters/projections/state/positions/depth log and identical dashboard page code + payload except sim
  results; sim differences are the same size as two runs of the same tree (simulate.py not fully deterministic).
  bt_split_score / age_eval / mover_context output identical. Recipe: code/make_core.py. HANDOFF.md rewritten as a short
  current-state file; this log holds the history. Note: the scratch chain run used the newest RealGM chart (Oct 5); outputs
  in the zip are still the 09-28 published ones (nothing republished or pushed this session).

## IN-SEASON TUNING: 12-season nightly replay for gap_k + tau  [2026-10-06, session 21] -> gap_k 20 CONFIRMED; tau slightly narrow (awaiting user)
Data: github.com/shufinskiy/nba_data/raw downloads are blocked in this sandbox now; rebuild via a blobless clone of shufinskiy/nba_data
(code/replay/rebuild_git.sh, same as rebuild.sh but `git cat-file blob`; don't run `git ls-tree -l` on the blobless clone,
it fetches every blob). Replay harness in code/replay/ (copy of gh model/+pipeline/ in /home/claude/rp with EMPTY
overrides.json, so the live 2026 role overrides don't leak into replays): build_state S --replay -> nightly.py at league-avg
0/5/10/20/30/41/55/70 team games (checkpoints.json), seasons 2014-2025 incl. 2019 (bubble) and 2020 (72 g).
Note: nightly's fill_tbd pads short seasons to 82 games, so latest.json proj_w is wrong for 2019/2020 replays; the
analysis scripts use the real remaining schedule instead (no fill).
gap_k (code/replay/gapk.py; exact expected final wins, tau as current): final-wins RMSE all checkpoints
  k 0: 5.272 | 5: 5.153 | 10: 5.122 | 20: 5.106 | 40: 5.106 | 80: 5.117 | never decay: 5.155 | win pace far worse.
  Flat 20-40; 14-18 leans 10, 21-25 leans 40, nothing near t 2.5 -> KEEP 20.
tau (code/replay/tau_cal2.py; sims with common random numbers, randomized PIT for integer wins): best tau by CRPS ~2.75
  from preseason through 41 games, ~2.25 at 55, 1.0-1.5 at 70 (21-25 wants ~3.0, 14-18 ~2.5). Current curve (2.42 -> 2.0
  @41 -> 1.8 @70): p10-p90 covers 73-76% at 0-55 games (nominal 80%), 80% at 70; p25-p75 covers ~45%.
  tau_mid 2.0 -> 2.5 (pre 2.42, late 1.8 unchanged): coverage 76-79%, CRPS -0.004 (t -1.8, 9/12 seasons, both halves) =
  calibration fix, not accuracy. Replay has no injury reports (live does) so live forecasts are a bit sharper -> 2.5 rather
  than 2.75. Preseason tau 2.42 also covers only ~73% (in-sample blend; out-of-sample a bit worse): 2.75 -> 76%, 3.0 -> 78%;
  affects dashboard preseason odds -> separate decision.
APPLIED 2026-10-06 (user OK, "do it"): blend.json tau_cal 1.1365 (new key) -> tau_blend 2.4197 -> 2.75, tau_model
  2.67 -> 3.0345, blend_curve.json tau x1.1365 (backups data/blend_pre_taucal.json, blend_curve_pre_taucal.json; not in zip);
  refit_blend.py multiplies its fitted tau by tau_cal; build_state tau_mid 2.0 -> 2.5 (and tau_mid now refreshed, not kept
  from an old config); state/2026 config tau_pre 2.75 / mid 2.5 / late 1.8. Chain rerun (depth chart 2026-10-05, so team
  margins moved up to 1.2 vs the zip's 09-23-chart files; the live nightly already used that chart). Preseason p10-p90 width
  ~18 -> ~20 wins; OKC title 44.8% -> 40.2%. Dashboard artifact republished (v34). Repo commit 37ea1e7 NOT PUSHED: the session's
  git proxy refused jvector26/JIMBO (GitHub account not linked to this claude.ai session, token alone no longer enough).
  Patch saved as gh_unpushed_37ea1e7.patch (zip root): next session `cd gh && git am ../gh_unpushed_37ea1e7.patch`, push,
  dispatch nightly. Until then the app/nightly run with the old tau.

## TEAM HALO + THIN-SAMPLE LEVERAGE (Ajay Mitchell question)  [2026-10-06, session 22] -> BOTH REJECTED, NOTHING APPLIED
User: Ajay Mitchell (OKC, +4.21, #10; box ~+1.0, lineup ~+2.8 mostly defense, 98th pct) - does elite-team performance leak into
player lineup residuals ("put an average player on OKC for 3 years")? And: young thin-sample players with +3/+4 ratings make team
forecasts leverage-sensitive to their minutes. Repo state: OKC has the largest minutes-weighted non-box component (+1.15/player).
TEST A, halo (code/halo/halo_bt.py -> data/halo_players.parquet): halo = season-N teammates' minutes-weighted mean residual
(main team, excl. self), joined to lineup_port_bt G. Halo sd only 0.31 (o) / 0.28 (d); corr with own residual -0.06/-0.15.
Free coefs relative to model (-1 = fully spurious): movers' halo -0.54 (t -1.2; 2015-19 +0.24, 2020-25 -1.16 t -2.2);
stayers' halo -0.08 (2015-19 +0.43 t 2.6, 2020-25 -0.36 t -3.9). Subtract lam*halo from movers: best lam .5 LOSO 13.4560 vs
13.4577 (-0.0017), 5/11 seasons better, halves +0.003/-0.005 -> FAILS (sign flips). From everyone: worse at every lam.
Recent-era hint (2020-25 movers' halo t -2.2) -> retest after 2026-27.
TEST B, leverage (code/halo/leverage_pre.py -> data/leverage_pre_{players,teams}.parquet): preseason backtest 2014-25 (330
team-seasons, age_eval scoring). Team features in wins = sum(min_proj2 x (0.6 res_o + 0.8 res_d))/3936 x pace x wins/pt for
thin (<4,000 decayed window poss) players with lineup > +1.5 (n 86 player-seasons). Forecast error on it: model +0.02 (t 0.0),
blend +0.56 (t 1.6; teams BEAT the blend). All-thin lineup: model -0.45 (t -1.6; halves -0.12/-0.60), blend -0.09. Any lineup
> +2: model +0.09, blend +0.11. Minutes realised/projected: thin-high 1.11, thin other 1.09, thick 0.94 -> they play MORE.
Sensitivity 2026: +/-500 min = Ajay 1.0 model wins (0.38 forecast), Jakucionis 1.6 (0.61). Thin all-lineup -0.45 is the one
thing to watch (same sign both halves, t < 2).

## LINEUP SHRINK, JUDGMENT CALL (user, 2026-10-07, session 22) -> APPLIED (everyone, by possessions)
User: young/thin-sample players with big lineup numbers (Ajay Mitchell, Jakucionis) make team forecasts leverage-sensitive;
shrink them harder. Evidence going in was mixed (game test 09-23 said no; preseason thin-lineup -0.45 t -1.6 said maybe),
so this is a JUDGMENT override with its cost measured, not a research-bar pass.
Rule: res_o, res_d x m, m = 1 - A*clip(1 - P/P1, 0, 1), A = 1/3, P1 = 10,000, P = poss_window (raw 3-season offensive
possessions; a full-time starter ~5,000/season). Symmetric (negative residuals shrink toward 0 too). code/lineup_shrink.py,
applied right after ratings_2025 loads in project.py and build_dashboard.py (dashboard JS sees the shrunk lo/ld).
Env LINEUP_SHRINK_A=0 switches it off (comparisons only).
Cost grid (code/shrink/shrink_game.py, shrink_pre.py): game LOSO base 13.4577; P1 10k A .33 13.4635 (+0.006; 2015-19
-0.000, 2020-25 +0.010); all settings +0.002..+0.011. Preseason (pre_P ratings adjusted with bt_split_ratings residuals,
age_eval scoring): model 8.643 -> 8.631 (A .33), blend 8.139 -> 8.144; A .5-.75 at P1 8k best model 8.626, A 1.0 worse.
Young-only variant (<=3 NBA seasons, code/shrink/shrink_young.py): same game cost (13.4635), preseason model 8.640, blend
8.145 -> the preseason gain comes from thin-window veterans too; user chose everyone.
2026 effect (same rosters, A=0 vs 1/3): 201 of 609 players move >=0.1. Ajay Mitchell 4.21 -> 3.70 (#10 -> #14), Jakucionis
3.12 -> 2.63, Hugo Gonzalez -0.76, Edey -0.49, Harper -0.40; raises Drake Powell +0.97, Mbeng +0.64, Ace Bailey +0.50.
Blend wins: BOS -0.49, CHA -0.33, SAS -0.25, OKC -0.14; IND +0.34, BKN +0.30 (team net change sd 0.22).
Pushed 5ae9734 (state + app payload, meta.po kept from e55f28f; site/index.html kept from e55f28f with version bump),
nightly dispatched (success; latest.json Ajay r 3.70), dashboard artifact v36 (live page's playoff engine kept, payload
swapped). Retest after 2026-27: does the shrink help on 2026-27 preseason ratings?

## PLAYOFF MINUTES (session 21, 2026-10-07) -> APPLIED at half weight  [RECONSTRUCTED 2026-10-18: session 21's own log section was lost with its zip]
What is in the code (gh e55f28f; model/po.json holds the fit and its note): expected playoff minutes per team game = avail x g(mpg when
playing), g piecewise linear (knots 15/22/28/33) fit on 2014-25 playoff teams, team rescaled to 240. Playoff strength = model strength
+ sum rating x (m_po - m_rs)/48 x pace/100, per-team change clipped to the 2014-25 playoff-team range [0.03, 2.89], weight 0.5 (user
choice: the forecast is ~60% Vegas, which never had the projection's bench drag). Series home court 4.0 (play-in keeps 2.6). Used in
nightly.py / jimbo_sim.py, the page's poDelta() and (2026-10-18 back-port) code/simulate.py po_delta + build_dashboard meta.po.
The original test tables (fit quality, backtest of series results) are not in this log: if needed, rerun from player minutes 2014-25.
Convention: simulate.py's sim_2026.csv / meta.py_sim = series home court 4.0 but NO po_delta (that is the baseline test_dashboard.js
compares the page sim against, run without opt.pd); the page and nightly add po_delta on top.

## ROLES UI + ALWAYS FIVE STARTERS (session 23, 2026-10-09, user request) -> APPLIED, commit d6026af, artifact v37
Rule, display and nightly hook are described in HANDOFF.md. Changed nothing on 2026-10-09 (every team already at five starters).

## CLEANUP / CODE-TREE RECONCILIATION (session 25, 2026-10-18)
Problem: session 21 (playoffs) and session 22 (shrink) forked the code tree; session 23 patched only the BUILT pages. Fix: (1) page
template code/dash_v2.html = the live site/index.html run backwards through build_app.py (head additions, loading div, app script
removed; data slot restored), which carries playoffs + roles + the "How it works" text; (2) simulate.py/build_dashboard.py got the
playoff pieces back; (3) add-on applied (shrink hook); (4) duplicate halo section removed from this log.
Regression check (depth chart pinned to 20261007 via DEPTH_DATE): rebuilt dash payload = live dash.json with 0 player-field differences,
identical teams/sched/meta (py_sim within Monte Carlo noise: wins <= 0.06, playoffs <= 0.005, title <= 0.008); build_app.py output
identical to the live site/index.html, sw.js and manifest. code/test_dashboard.js and ui_edit_test_db.py pass on the current chart.
tools/patch_dash_roles.py is no longer needed (patch is in the template).
Duren (agreed to a 5-yr $200M deal 2026-10-01; on RealGM's chart since): removing him from depth_config exempt = tier S, 1,960 min,
DET 49.7 -> 49.8 wins. Not yet applied in the repo.

## SEPARATED-POSSESSIONS SHRINK (stapled teammates)  [2026-10-15, session 24] -> FAILS BAR (consistent but small), NOTHING APPLIED
User: lineups aren't randomized; if a core group always plays together, RAPM can't split credit among them, so the lineup
residual partly carries teammates' unexplained value. Proposal tested: base the thin-sample shrink on possessions that
actually separate a player from his most frequent teammates instead of total possessions.
Data: stints 2011-2025 rebuilt (rebuild_git.sh). code/sepposs/sep_poss.py -> data/sep_poss.parquet, per player x window:
P_raw; sh1 = share of poss shared with top teammate (median ~0.50 for >3k poss); sep2 = share with NEITHER top-2 teammate on
(median ~0.31); neff = ridge-posterior effective sample 1/[(X'WX+2000 I)^-1]_ii - 2000 (same design/decay/lam as the RAPM
fit; neff/naive median ~0.67). Most stapled (2025 window, neff/naive): Jokic .36, Brunson .40, SGA .40, W. Carter .41.
Variant shrink: P -> P x c / k (c = neff/naive per side, 1-sh1 or sep2; k = minutes-weighted mean, data/sep_k.json) so
the average level is roughly unchanged and only WHO gets shrunk changes (variants shrink slightly MORE on average:
mean 1-m .091-.115 vs .088, so gains are not 'less shrink').
GAME (code/sepposs/sep_game.py, sep_h2h.py; grid in data/sep_game_grid.csv): every variant beats total poss at every
matched (P1 8-15k, A .25-.5) setting, both halves. At production setting (10k, 1/3) vs current 13.4635: neff 13.4610
(-0.0026, 8/11 seasons, t 2.05, halves -.0015/-.0033), 1-top1 13.4597 (-0.0038, 7/11, t 2.01), sep2 13.4593 (-0.0043,
7/11, t 2.06). neff at P1 12-15k reaches t 2.7-3.8 (grid-picked). Direct test (code/sepposs/sep_diag.py, no shrink as
base, residual x (c-1) as extra regressor): neff coef +0.75 t 2.50 (2015-19 +0.12 t 0.3; 2020-25 +1.18 t 4.1) -> stapled
players' residuals carry over less, as the user suspected; 1-top1 +0.42 t 1.5; sep2 +0.20 t 1.0.
PRESEASON (code/sepposs/sep_pre.py, age_eval): current shrink model 8.631 / blend 8.144; neff 8.636/8.145, top1
8.633/8.145, sep2 8.628/8.141 (other P1/A similar) -> no team-level gain.
2026 effect (neff, 10k, 1/3): 15 players move >= 0.1, max 0.18 (Jokic -0.17, Caruso +0.14); sd 0.03. Too small to matter.
Verdict: direction confirmed, size small, t ~2 < 2.5, early half weak, preseason flat -> not applied. Retest neff version
after 2026-27 (adds a season; recent-era signal t 4.1 is the thing to watch).

## SKILL DRILL-DOWN (app feature, user request)  [2026-10-18, session 25] -> BUILT; artifact v38 published; app ships with next site push
Tap a skill bar in the player view -> table of the box stats behind it: projection contribution, 2025-26 contribution, 2025-26
actual stat (e.g. 9.9 AST/100, league 5.6; 36.9% 3P 195/528, league 36.0%). Single-stat skills (playmaking, self-creation, ball
security, both rebounding) show one row. Hustle splits into loose balls / 3PT contests. Payload +60 KB (hist[k].raw for 525
players with a 2025-26 season, meta.raw league rates, players[].hb). Display only, no model change: with DEPTH_DATE=20261007 the
payload equals the live dash.json except the added fields (0 player diffs, hist identical apart from raw). Tests: per-stat rows
sum to the bar value (Brunson, Gobert, Wembanyama: all 5 skills tried), open state survives re-render, no page errors on
iPhone 13 and desktop, in the artifact build and the app build with in-season latest.json; test_dashboard.js and
ui_edit_test_db.py pass. Pre-change copies: scratch only (template diff is self-contained between the two
'skill drill-down' comments).

## CLINGAN CONTESTED-3 HUSTLE CHECK  [2026-10-18, session 25] -> NOTHING APPLIED to the model (user: "Leave it."); display regroup DONE
(Copied from project doc claude/clingan_c3_check_2026-10-18.md; session 25's zip predates it.)
Question (user): Clingan gets +0.44 from 3PT contests in the hustle term; implausible for a center.
hustle = 1.74*lbr_k1000 + 0.253*cont3_k3000 (hustle.py). Clingan window c3 = +1.75 -> +0.444; loose-ball part -0.21, net +0.23. No calc bug
(reproduced from the raw NBA.com table). 2025-26: 278 contested 3s in 2,094 min = 6.5 per 100 def poss (league 3.3), #1 of 333 with 1,500+ poss;
4.8 per 36 vs 2.4 league (Gobert/Wemby/Adebayo 2.8-3.0). Persistent (2024-25 also top) and not a Portland scheme effect (Ayton, Reath,
R. Williams 2.7-3.4 on the same team). Largest c3 term in the model (next J. Williams .33, Wiggins .31). Rating 5.76 (#6), without c3 5.32. ~0.6 POR wins.
Test (code/clingan_c3_test.py, not in the 10-18 zip; game LOSO, hustle era 2018-2025, 9,516 games): base 13.8524; lbr+c3 13.8408 (c3 +0.227 t 1.8
joint; solo +0.337 t 2.8). Non-centers +0.274 (t 2.3); centers -0.026 (t -0.1). Difference -0.30, t -0.7; halves 2018-21 t -2.5, 2022-25 t -0.2.
pos3=B bigs +0.15 (t 0.5). Fails the bar. Options if revisited: zero c3 for centers (Clingan -0.44, M. Williams -0.31, Missi -0.29, Kessler -0.27,
Mobley -0.26) or halve/drop c3 for everyone; retest after 2026-27.
Display regroup (user "Yeah"): the 3PT-contest part of hustle (p.hb[1]) is added to the perimeter_def display part; the hustle bar is loose balls
only. Perimeter-defense drill-down gets a '3PT shots contested' row (projection only). Totals unchanged. App commit 6658eb2, artifact v39.

## FORECAST VS LINE COLUMN (UI, user request)  [2026-10-18, session 26] -> DONE, display only
User question: NOP forecast 31.5 (model 33.8, line 27.5) above CHI 30.6 (model 35.2, line 27.5). Answer: the blend's roster-age term
(-0.595 x (roster age - avg)) adds ~+1.65 to NOP, ~+0.2 to CHI; the 37/63 split alone gives NOP 29.9, CHI 30.4.
User floated baking the age term into the model column; advised against (the term corrects a bias Vegas and the model share; inside
the model it would need 1.6 wins/yr and would vanish at the all-Vegas slider end). User: "fine as is". Possible later test: model-only
roster-age term in the preseason backtest, then refit the blend.
Change: standings column 'edge' "Model vs line" (mw - line, colored >= 3) -> "Forecast vs line" (bw - line, colored >= 2).
First applied to the built pages only (app c2e5e0b, artifact v40), then ported to code/dash_v2.html along with the 6658eb2 regroup
(the 10-18 zip template lacked both). Check: DEPTH_DATE=20261007 chain -> test_dashboard.js (max 0.0011; wins 0.35, title 0.005),
ui_edit_test_db.py (no page errors), skill_drill_test.py pass; build_app.py = live site/index.html except the dash.json version hash.

## TEAM-VIEW POSTSEASON LINE (UI, user request)  [2026-10-18, session 26] -> DONE, display only, app 9631394, artifact v41
User asked what "Playoffs" means: the 8-team bracket (seeds 1-6 plus the two play-in winners), not reaching the play-in. He asked for
play-in odds in the team view. The page sim already counted top6 / pin (7th-10th) / seed1 / r2 / cf / fin; poPath() shows Top-6 seed,
Play-in, Makes playoffs, Title in the rsum style inside .roster (outside it, the line clipped on phones because .roster is the sticky,
viewport-wide block). Checked: test_dashboard.js, ui_edit_test_db.py, phone screenshots of NOP (artifact build and app build), no page errors.

## SESSIONS 27-33 (2026-10-19 .. 10-28) [added 2026-10-29, session 34, from the project docs written at the time]
- s27 COLLEGE BPM vs ROOKIE PRIOR -> NOTHING APPLIED. First-rounders 2008-25 (N 394, LOCO wRMSE): cscore 0.977, BPM+age+ht 0.998,
  cscore+BPM 0.983; BPM adds nothing over cscore (DBPM carries the signal). 2nd-round/UDFA (N 385): BPM+age 0.906 vs flat 0.927,
  t -2.55: lead for backlog "college prior for 2nd-round/undrafted"; not game-tested. Join on draft_year + pick (csv rookie_season = debut).
  Doc: claude/college_bpm_rookie_test_2026-10-19.md.
- s28 OPENING NIGHT: odds abbreviation + overwrite fixes; final pre-opener rebuild (forecast = model + FIXED preseason gap, so state
  changes move forecasts 1:1); opening-day chart. Doc: claude/opening_night_check_2026-10-20.md; HANDOFF SESSION 28.
- s29 "vs Vegas" app tab (gh 9ee6949): pregame.csv log of last pre-tip JIMBO margin + DK line; vegas.json; tiles, running avg miss,
  through-origin t. First 14 games: closer 8/14, miss 14.4 vs 15.2, t 1.1. ~600 games for t 2.5 if JIMBO deserves half weight.
- s30 team-view wins chart from 2026-10-20 only (gh f3ff01c).
- s31/32 player page This season | Projection | Career; season leaderboard (sr raw season rating, points added = sr x poss/100 vs
  average). USER RULE: This-season views pure 2026-27, no shrink. Rookie-note fix (nbaPast). gh 397bdf0, eedb84f, 16df89d, d8089f0.
- s33 TONIGHT'S MINUTES (gh df13c2b, user OK): healthy DNPs count as 0-min games, prior capped at preseason mpg-when-playing, trim
  order by minutes x chance he plays. Replay 10-20..10-27 (56 games): player-game RMSE 8.43 -> 7.13 (t -7.2); margin 15.06 -> 15.03.
  OWED: 2014-25 replay incl. non-appearing players. Doc: claude/tonight_minutes_fix_2026-10-28.md.
- s34 housekeeping: all seven built-page patches folded into code/dash_v2.html + appsrc/app.js; build reproduces live index.html exactly.

## MATCHUPS TAB REMOVED + PLAYER-CARD INJURY BOX (2026-10-29, session 35, user request) -> SHIPPED, display only
See HANDOFF session 35. Checks: phone + desktop playwright (Leonard, Doncic, Barlow, Veesaar, Brunson), no page errors;
skill_drill_test, test_dashboard.js (max 0.0010), ui_edit_test_db.py pass; local nightly ratings/mk identical to live.
