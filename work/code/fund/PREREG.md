# Four fundamentals tests: definitions and expected signs, written BEFORE any result was seen (2026-11-11, session 50)
Bar as usual: lower LOSO, season-clustered / season-paired t > 2.5, same sign 2015-19 and 2020-25. Every number gets reported, hits and misses.
One primary definition per test; anything else is labelled robustness.

## T1 within-season improvement of young players (in-season harness, 2014-25, as learnspeed)
Base = production in-season stack (pre + box update + RAPM lam 6000 + team offsets 80 + B2B). Add, per game (home minus away):
exposure E_g = sum of minute shares x 5 x pace/100 of players in group g, and E_g x prog, prog = mean of the two teams' game number / 82.
Groups (as of season start): R1 = no NBA history and age < 25; R2 = first NBA season was S-1 and age < 25.
Primary: the two drift terms (E_R1 x prog, E_R2 x prog) added jointly, level terms E_R1, E_R2 included in base and candidate.
EXPECTED SIGN: both drift coefficients > 0 (young players are better later in the season than the running estimate says).
Scored with actual (a) and pregame trailing-5 (t) minutes; the t version decides.

## T3 hole in the lineup (research bar: preseason ratings, 13,206 games)
Lead handler = playmaking component at or above the 80th percentile of rated players with 500+ minutes in season N (window N-2..N).
Rim protector = same on the rim_protection component. Coverage = min(sum of group minutes, 48) / 48 per team-game, home minus away.
Primary minutes: "active x trailing": a player who appeared tonight gets his season-to-date minutes per game played (team games 6+);
robustness: actual minutes; cut-offs 70th / 90th.
EXPECTED SIGN: both coverage coefficients > 0 (a team with no lead handler / rim protector does worse than the rating sum says).

## T6 transition (add-on features, defev protocol: window N-2..N decay .3/.6/1, rate vs league shrunk poss/(poss+k), primary k 1000)
Transition play = FGA, turnover or FT trip by a team within 7 s of gaining the ball on a player defensive rebound or a steal.
o_trvol = player's own transition plays per 100 on-court offensive possessions.  EXPECTED SIGN: > 0 (weak prior).
o_tron  = team transition points per 100 offensive possessions with the player on court.  EXPECTED: > 0.
d_tron  = opponent transition points per 100 defensive possessions with the player on court.  EXPECTED: < 0.
Robustness: k 300 / 3000.

## T8 garbage time out of the box stats (refit test, like dreb / FT)
Garbage = same rule as the stint fits (stint starting in Q4 with margin 20+, or in the last 6 minutes with 15+).
Box counts and on-court possessions recomputed without garbage stints for 2012-25; component weights W fixed; lineup residuals refit.
EXPECTED SIGN: LOSO lower without garbage time. Expected size small (under 0.01).
