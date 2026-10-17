# Position traits v1: per-position trait ratings and the grade

## What the model is

Every FBS player gets a set of named traits at his position and two numbers.

- The **grade** is a 20 to 80 scouting number built from his traits, which
  at most positions include his EPA credit. It has mean 50 and sd 10 among
  graded qualified FBS players at his position that week. It is built to
  predict how good he will be next season.
- The **impact** is player value v4 (`cfb_player_value_v4`, published in
  `cfb.player_values`): points above positional replacement from his
  credited plays, opponent adjusted. The trait table carries it as **wins
  above replacement** (WAR), the same points weighted by how much a point
  was worth in each game he played (see Wins above replacement). v4 is not
  refit here.

A trait is a per-game rate built from tagged plays or box score counts,
adjusted for the opponent where the data allows, and turned into a position
percentile among qualified FBS players that week. Traits are `counted` when
the feed tags the player directly and `inferred` when they come from unit
outcomes or teammate comparisons. Nothing from the week being rated enters
its own rating.

Kickers were removed from the trait table on 2026-10-16: field goal accuracy
has no measurable skill at college volumes (made over the distance
expectation has a split-half reliability of 0.00 at eight games and a
year-to-year Spearman of 0.08; research/kicker_punter_grade_2026-10-16.md),
and the one stable kicker trait, the kickoff leg, is not what a kicker grade
should mean. Punters went from one trait (punt distance, a single statistic
rescaled to 20 to 80) to four: `p.punt_average`, `p.net`,
`p.fair_catch_share` and `p.no_return_share`, built in
`backend/players/traits/punting.py`. The P rows in the tables below are from
the one-trait fit and the K rows from the fit that still carried kickers;
both are kept as a record, and the refit's P figures are in the section
"Punters, four traits" at the end.

The grade replaces the margin fit of the first version. That fit asked
whether a trait improved a team spread forecast beyond the team model, and
almost nothing passed. The amendment to the design withdrew that question:
the grade rates players individually, as prospects and for scouting, and
nothing is fitted to team margins.

Development seasons are 2020 through 2024. 2025 is the held-out check. 2026
was never used. The model is
`backend/data/processed/players/models/position_grade_2024.json`, the full
evaluation is
`backend/data/processed/players/position_traits/evaluation_2024.parquet`,
and every table below is copied from those two files and from the run log of
`position-traits-fit --dev-seasons 2020 2021 2022 2023 2024 --validate 2025`
on 2026-10-16 (`output/position-grade-fit-202610162323.log`). Player values
for 2020 through 2025 are all `cfb_player_value_v4`, so the EPA baseline is
one model.

## How the grade was chosen

Every choice below was made on the four development pairs, 2020 to 2021
through 2023 to 2024. The measure is the Spearman correlation of a player's
final-week grade in season t with his v4 `shrunk_per_game` at the final
week of season t + 1, per position, averaged over the four pairs, over
players graded at the final week of both seasons at the same position and
with a v4 value in both. EPA alone (v4 in t against v4 in t + 1, same
players) is the baseline. The 2024 to 2025 pair was computed once, after
every choice was fixed (`score_holdout=False` withholds it while
developing).

The starting grade weights each trait by evidence about the trait itself:

- **Reliability**, the split-half correlation at 8 games (a player's value
  over his first four odd games against his first four even games), with
  the five development seasons pooled. A reliability on fewer than 30
  players counts as unmeasured; 30 is a sample floor, not a tuned value.
- **Portability**, the transfer test: last season's final-week value
  regressed on this season's, once for players who stayed and once for
  matched transfers, over the development pairs. The portable share is the
  mover slope over the stayer slope, clipped to [0, 1]. When fewer than 20
  players moved or the stayer slope is not positive, the share is imputed
  as the mean measured share of the position's traits (or of every
  position's) and flagged `portability_imputed`.

`weight = max(reliability, 0) x portability`, normalized within the
position. Under this rule a trait is `weighted`; `unmeasured` when its
portability is imputed; `noise` under 0.10 reliability;
`unmeasured_reliability` when reliability cannot be measured (under 30
players, or no per-game values); and `team-driven` when its measured
portability is at or below zero. Orientation: a higher-is-better trait
enters as is, a lower-is-better one negated, and one with no registered
direction takes the sign of its Spearman with v4 `shrunk_per_game` on the
development seasons. Only `def.run_support` has none; its correlation is
0.200 over 4,795 player seasons, so it enters positive.

Three changes were then tried in order. Each was kept at a position only
where it raised the dev-pair mean there.

1. **EPA credit as a trait.** `epa.credit` is the player's v4
   `shrunk_per_game` through the week, z-scored among qualified FBS players
   at the position like any trait. Its reliability comes from per-game
   adjusted EPA in the credited games and its portability from the same
   portal study. It enters the reliability x portability weights.
2. **Fitted weights.** Per position, a ridge regression of next season's v4
   `shrunk_per_game` (z-scored within position and season) on the season-t
   final-week trait z-scores (available traits only, capped at plus or
   minus 3, missing as 0), over the players graded at t. Coefficients are
   bounded at zero below for higher-is-better traits; negative is allowed
   only where the direction is lower-is-better or unregistered. Only traits
   with a reliability on at least 30 players enter. The penalty is chosen
   from 61 log-spaced values between 0.01 and 10,000 by leave-one-pair-out
   squared error. For the evaluation each pair is graded with weights
   fitted on the other three pairs, so the fitted column is out of sample;
   the kept weights are the fit on all four pairs.
3. **Shrinkage by games.** The composite is multiplied by
   `games / (games + k_g)` before the 50 plus or minus 10 scaling, with
   `k_g` the games at which the composite's split-half reliability reaches
   0.5 (interpolated from r = 0 at zero games), capped at 12 when it never
   does.

Dev-pair means, and which change each position kept:

| Group | Current grade | +EPA trait | +fitted weights | +shrinkage | EPA alone | EPA kept | Fitted kept | Shrinkage kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QB | 0.307 | 0.404 | 0.429 | 0.432 | 0.422 | yes | yes | yes |
| RB | 0.277 | 0.340 | 0.360 | 0.359 | 0.364 | yes | yes |  |
| WR | 0.228 | 0.268 | 0.387 | 0.384 | 0.366 | yes | yes |  |
| TE | 0.253 | 0.311 | 0.342 | 0.343 | 0.353 | yes | yes | yes |
| DL | 0.250 | 0.252 | 0.268 | 0.266 | 0.136 | yes | yes |  |
| LB | 0.188 | 0.188 | 0.199 | 0.198 | 0.131 |  | yes |  |
| DB | 0.118 | 0.118 | 0.119 | 0.120 | 0.079 |  | yes | yes |
| OL |  |  |  |  |  |  |  |  |
| P | -0.090 | -0.090 |  | -0.094 |  |  |  |  |

Each column builds on what the position kept before it. OL has no v4
value, so it has no measure and keeps the starting grade. P has too few
punters with a v4 value to fit (a pair needs 30) and its correlations are
not meaningful. K has no grade.

Per pair:

| Group | Variant | 2020-21 | 2021-22 | 2022-23 | 2023-24 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| QB | current grade | 0.209 | 0.381 | 0.188 | 0.450 | 0.307 |
| QB | +EPA trait | 0.329 | 0.495 | 0.325 | 0.465 | 0.404 |
| QB | +fitted weights | 0.392 | 0.530 | 0.367 | 0.428 | 0.429 |
| QB | +shrinkage | 0.395 | 0.531 | 0.376 | 0.427 | 0.432 |
| QB | EPA alone | 0.374 | 0.518 | 0.408 | 0.390 | 0.422 |
| RB | current grade | 0.348 | 0.215 | 0.281 | 0.263 | 0.277 |
| RB | +EPA trait | 0.364 | 0.293 | 0.336 | 0.366 | 0.340 |
| RB | +fitted weights | 0.332 | 0.337 | 0.363 | 0.407 | 0.360 |
| RB | +shrinkage | 0.336 | 0.334 | 0.360 | 0.405 | 0.359 |
| RB | EPA alone | 0.328 | 0.340 | 0.373 | 0.417 | 0.364 |
| WR | current grade | 0.204 | 0.289 | 0.157 | 0.261 | 0.228 |
| WR | +EPA trait | 0.252 | 0.325 | 0.202 | 0.295 | 0.268 |
| WR | +fitted weights | 0.431 | 0.388 | 0.339 | 0.390 | 0.387 |
| WR | +shrinkage | 0.431 | 0.377 | 0.338 | 0.392 | 0.384 |
| WR | EPA alone | 0.395 | 0.383 | 0.316 | 0.372 | 0.366 |
| TE | current grade | 0.259 | 0.135 | 0.327 | 0.291 | 0.253 |
| TE | +EPA trait | 0.287 | 0.216 | 0.433 | 0.306 | 0.311 |
| TE | +fitted weights | 0.393 | 0.281 | 0.404 | 0.289 | 0.342 |
| TE | +shrinkage | 0.403 | 0.277 | 0.404 | 0.289 | 0.343 |
| TE | EPA alone | 0.329 | 0.337 | 0.473 | 0.275 | 0.353 |
| DL | current grade | 0.301 | 0.251 | 0.197 | 0.251 | 0.250 |
| DL | +EPA trait | 0.303 | 0.252 | 0.204 | 0.249 | 0.252 |
| DL | +fitted weights | 0.319 | 0.255 | 0.231 | 0.265 | 0.268 |
| DL | +shrinkage | 0.321 | 0.248 | 0.230 | 0.266 | 0.266 |
| DL | EPA alone | 0.142 | 0.122 | 0.129 | 0.153 | 0.136 |
| LB | current grade | 0.174 | 0.149 | 0.261 | 0.167 | 0.188 |
| LB | +EPA trait | 0.174 | 0.149 | 0.261 | 0.167 | 0.188 |
| LB | +fitted weights | 0.180 | 0.159 | 0.279 | 0.179 | 0.199 |
| LB | +shrinkage | 0.183 | 0.159 | 0.272 | 0.179 | 0.198 |
| LB | EPA alone | 0.169 | 0.185 | 0.109 | 0.060 | 0.131 |
| DB | current grade | 0.098 | 0.110 | 0.124 | 0.138 | 0.118 |
| DB | +EPA trait | 0.098 | 0.110 | 0.124 | 0.138 | 0.118 |
| DB | +fitted weights | 0.090 | 0.110 | 0.131 | 0.146 | 0.119 |
| DB | +shrinkage | 0.096 | 0.112 | 0.127 | 0.145 | 0.120 |
| DB | EPA alone | 0.032 | 0.106 | 0.095 | 0.082 | 0.079 |

Weight share of `epa.credit` under reliability x portability (change 1):

| Group | EPA share | Kept |
| --- | --- | --- |
| QB | 0.230 | yes |
| RB | 0.306 | yes |
| WR | 0.138 | yes |
| TE | 0.302 | yes |
| DL | 0.077 | yes |
| LB | 0.000 | no |
| DB | 0.000 | no |
| P | 0.000 | no |

At LB and DB, EPA credit carries no weight under that rule. Its 8-game
reliability is 0.078 and 0.080, under the 0.10 noise bar, because few
defenders are credited in eight games. So change 1 left LB and DB
unchanged and was not kept there.

Ridge penalties of the kept fits (change 2):

| Group | Penalty |
| --- | --- |
| QB | 5.01 |
| RB | 0.01 |
| WR | 15.85 |
| TE | 25.12 |
| DL | 15.85 |
| LB | 501.19 |
| DB | 251.19 |

The RB penalty sits at the bottom of the grid (0.01), so the RB fit is
close to unpenalized.

Shrinkage games (change 3):

| Group | k_g | Kept |
| --- | --- | --- |
| QB | 12.00 | yes |
| RB | 12.00 | no |
| WR | 4.97 | no |
| TE | 7.86 | yes |
| DL | 12.00 | no |
| LB | 7.47 | no |
| DB | 12.00 | yes |
| OL | 12.00 | no |
| P | 11.76 | no |

The shrinkage gains are small: 0.003 at QB, 0.001 at TE and DB.

## Weights

For a fitted position `coefficient` is the ridge coefficient per standard
deviation of the trait, and `weight` is the coefficient over the sum of
absolute coefficients at the position. For a position on the starting rule
`weight` is reliability x portability normalized, and there is no
coefficient. Under fitted weights a trait fitted to zero is `unweighted`;
a weighted trait whose measured portability is at or below zero would be
`team-driven (weighted)` (none is). `not kept` marks `epa.credit` where
change 1 was not kept.

| Group | Trait | Reliability | Portability | Method | Coefficient | Weight | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QB | qb.drive_engine | 0.191 | 1.000 | fitted | 0.1071 | 0.189 | weighted |
| QB | qb.explosive | 0.288 | 0.360 | fitted | 0.0000 | 0 | unweighted |
| QB | qb.ball_security | 0.115 | 1.000 | fitted | 0.0000 | 0 | unweighted |
| QB | qb.floor | 0.269 | 0.672 | fitted | 0.0000 | 0 | unweighted |
| QB | qb.legs | 0.160 | 1.000 | fitted | 0.0732 | 0.129 | weighted |
| QB | qb.pressure_handling | 0.351 | 1.000 | fitted | 0.0077 | 0.014 | weighted |
| QB | qb.receiver_independence | 0.537 | 0.851 (imputed) | fitted | 0.0000 | 0.000 | weighted |
| QB | epa.credit | 0.503 | 0.927 | fitted | 0.3789 | 0.668 | weighted |
| RB | rb.yards_over_expected | 0.150 | 1.000 | fitted | 0.0168 | 0.034 | weighted |
| RB | rb.stuff_avoidance | 0.108 | 1.000 | fitted | 0.0000 | 0 | unweighted |
| RB | rb.breakaway | 0.172 | 0.434 | fitted | 0.0238 | 0.049 | weighted |
| RB | rb.short_yardage | -0.044 | 0.960 | fitted | 0.0000 | 0 | unweighted |
| RB | rb.receiving | 0.190 | 1.000 | fitted | 0.0360 | 0.073 | weighted |
| RB | rb.ball_security | -0.051 | 0.875 (imputed) | fitted | 0.0000 | 0 | unweighted |
| RB | rb.line_independence | 0.107 | 1.000 | fitted | 0.0000 | 0 | unweighted |
| RB | epa.credit | 0.378 | 0.733 | fitted | 0.4148 | 0.844 | weighted |
| WR | rec.target_share | 0.715 | 0.354 | fitted | 0.2133 | 0.392 | weighted |
| WR | rec.value_per_target | 0.339 | 0.755 | fitted | 0.0000 | 0 | unweighted |
| WR | rec.explosive | 0.225 | 0.749 | fitted | 0.0285 | 0.052 | weighted |
| WR | rec.late_down | 0.299 | 0.966 | fitted | 0.0000 | 0 | unweighted |
| WR | rec.qb_independence | 0.145 | 0.878 | fitted | 0.0000 | 0 | unweighted |
| WR | epa.credit | 0.428 | 0.409 | fitted | 0.3021 | 0.555 | weighted |
| TE | rec.target_share | 0.714 | 0.235 | fitted | 0.1895 | 0.316 | weighted |
| TE | rec.value_per_target | 0.228 | 1.000 | fitted | 0.0000 | 0 | unweighted |
| TE | rec.explosive | 0.131 | 1.000 | fitted | 0.0424 | 0.071 | weighted |
| TE | rec.late_down | 0.207 | 0.876 | fitted | 0.0220 | 0.037 | weighted |
| TE | rec.qb_independence | 0.117 | 1.000 | fitted | 0.0000 | 0 | unweighted |
| TE | epa.credit | 0.403 | 0.885 | fitted | 0.3458 | 0.577 | weighted |
| DL | def.havoc | 0.364 | 0.618 | fitted | 0.1908 | 0.539 | weighted |
| DL | def.pass_rush | 0.130 | 1.000 | fitted | 0.1408 | 0.397 | weighted |
| DL | def.finishing | 0.073 | 1.000 | fitted | 0.0000 | 0 | unweighted |
| DL | def.run_stop | 0.179 | 0.346 | fitted | 0.0000 | 0 | unweighted |
| DL | def.every_week | 0.270 | 0.545 | fitted | 0.0000 | 0 | unweighted |
| DL | def.impact |  | 0.654 (imputed) | fitted |  | 0 | unmeasured_reliability |
| DL | epa.credit | 0.113 | 0.416 | fitted | 0.0227 | 0.064 | weighted |
| LB | def.havoc | 0.424 | 0.596 | fitted | 0.0407 | 0.166 | weighted |
| LB | def.pass_rush | 0.138 | 0.808 | fitted | 0.0647 | 0.263 | weighted |
| LB | def.finishing | 0.141 | 1.000 | fitted | 0.0013 | 0.005 | weighted |
| LB | def.run_stop | 0.290 | 0.522 | fitted | 0.0516 | 0.210 | weighted |
| LB | def.every_week | 0.360 | 0.635 | fitted | 0.0362 | 0.147 | weighted |
| LB | def.tackle_share | 0.771 | 0.643 | fitted | 0.0388 | 0.158 | weighted |
| LB | def.coverage | 0.188 | 0.529 | fitted | 0.0127 | 0.052 | weighted |
| LB | def.impact |  | 1.000 | fitted |  | 0 | unmeasured_reliability |
| LB | epa.credit | 0.078 | 0.000 | fitted |  | 0 | not kept |
| DB | def.havoc | 0.290 | 0.495 | fitted | 0.0521 | 0.485 | weighted |
| DB | def.every_week | 0.259 | 0.593 | fitted | 0.0377 | 0.351 | weighted |
| DB | def.ball_production | 0.262 | 0.504 | fitted | 0.0000 | 0 | unweighted |
| DB | def.run_support | 0.485 | 0.353 | fitted | 0.0177 | 0.164 | weighted |
| DB | def.impact |  | 0.000 | fitted |  | 0 | unmeasured_reliability |
| DB | def.pass_impact |  | 0.000 | fitted |  | 0 | unmeasured_reliability |
| DB | epa.credit | 0.080 | 1.000 | fitted |  | 0 | not kept |
| OL | ol.protection | 0.311 | 0.716 (imputed) | reliability |  | 0.337 | unmeasured |
| OL | ol.stuff_rate | 0.253 | 0.716 (imputed) | reliability |  | 0.274 | unmeasured |
| OL | ol.short_yardage | 0.024 | 0.716 (imputed) | reliability |  | 0 | noise |
| OL | ol.line_yards | 0.360 | 0.716 (imputed) | reliability |  | 0.389 | unmeasured |
| K | k.field_goal | 0.036 | 0.800 | reliability |  | 0 | noise |
| K | epa.credit |  | 0.800 (imputed) | reliability |  | 0 | not kept |
| P | p.punt_average | 0.497 | 0.580 | reliability |  | 1.000 | weighted |
| P | epa.credit |  | 0.580 (imputed) | reliability |  | 0 | not kept |

What the table says:

- EPA credit carries most of the weight where it was kept: 0.844 at RB,
  0.668 at QB, 0.577 at TE and 0.555 at WR. Of the traits, the fit keeps
  drive engine (0.189), legs (0.129) and pressure handling (0.014) at QB;
  target share (0.392 at WR, 0.316 at TE); and receiving, breakaway and
  yards over expected at RB, each under 0.08.
- On defense the fit keeps the box score rates. DL is havoc (0.539) and
  pass rush (0.397) with EPA credit at 0.064. LB spreads weight over seven
  traits, led by pass rush (0.263), run stop (0.210) and havoc (0.166). DB
  is havoc (0.485), every week (0.351) and run support (0.164).
- Many traits the reliability rule weighted are fitted to zero:
  `qb.explosive`, `qb.floor` and `qb.ball_security`, value per target at
  WR and TE, and every week and run stop at DL.
  `qb.receiver_independence` keeps a coefficient of 0.0000244, effectively
  zero.
- OL, K and P keep the starting rule. OL units never move, so OL
  portability is the all-position mean (0.716), and P is punt average
  alone.

## The grade

For one week, among qualified FBS players at the position:

- each weighted trait is z-scored on the pool's mean and sd, multiplied by
  its sign (a trait with no spread gives everyone 0) and capped at plus or
  minus 3. The cap (`Z_CAP`) is a guard against single-game extremes and
  thin pools, not a tuned value; the trait values and percentiles on the
  card stay uncapped;
- a trait is available at the position that week when at least half of
  the week's qualified FBS players there have a value for it. An
  unavailable trait leaves the composite for everyone that week, so every
  player at the position is graded on the same traits;
- `composite = sum(w_t z_t) / sum(|w_t|)` over the available traits, with
  z = 0 (the position average) for an available trait the player has no
  value for, then multiplied by `games / (games + k_g)` at QB, TE and DB;
- a player is graded only if he has a value for his position's primary
  exposure trait and the available traits he has values for carry at least
  0.6 of the available absolute weight (`PRIMARY_TRAITS`,
  `MIN_WEIGHT_COVERAGE`, `MIN_TRAIT_AVAILABILITY`, `Z_CAP` and the
  shrinkage constants in `backend/players/traits/grade.py`);
- `grade = 50 + 10 x (composite - mean) / sd` over the graded players,
  clipped to [20, 80];
- `grade_percentile` ranks the composite among them, 0 to 100;
- `uncertainty = 10 x sqrt(1 - r)`, with `r` the composite's split-half
  reliability at the player's game count (linear from r = 0 at zero games,
  flat past 12 games).

| Group | Primary trait |
| --- | --- |
| QB | qb.explosive (dropbacks) |
| RB | rb.yards_over_expected (carries) |
| WR, TE | rec.value_per_target (targets) |
| DL, LB, DB | def.every_week (games) |
| OL | ol.line_yards |
| K | k.field_goal |
| P | p.punt_average |

The primary trait is required whatever its weight, so a trait fitted to
zero (`qb.explosive`, value per target) still decides who has a role at the
position. Unqualified and non-FBS players, and qualified players who fail
the coverage rule, keep their trait rows and get no grade row.

Players who lose the grade to the rule at the final week of 2025, from the
evaluation file's `grade_coverage` table (every season is there):

| Group | Qualified FBS | Graded | No primary trait | Coverage under 0.6 |
| --- | --- | --- | --- | --- |
| QB | 172 | 170 | 2 | 0 |
| RB | 403 | 403 | 0 | 0 |
| WR | 660 | 659 | 1 | 0 |
| TE | 248 | 248 | 0 | 0 |
| DL | 1050 | 1050 | 0 | 0 |
| LB | 716 | 716 | 0 | 0 |
| DB | 1168 | 1168 | 0 | 0 |
| OL | 136 | 136 | 0 | 0 |
| P | 136 | 136 | 0 | 0 |

## Composite reliability

The composite's own split-half correlation with the kept weights, pooled
over the development seasons. Each player's games alternate between two
halves in week order, each half is scored as its own snapshot week, and
the two composites are correlated; EPA credit enters each half as adjusted
EPA per game. Shrinkage is a constant within a half, so it does not change
this curve. The curve sets the uncertainty band and `k_g`.

| Group | 4 games | 6 games | 8 games | 12 games | Players at 4 | Players at 6 | Players at 8 | Players at 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QB | 0.322 | 0.427 | 0.424 | 0.440 | 934 | 751 | 598 | 260 |
| RB | 0.260 | 0.318 | 0.351 | 0.427 | 2116 | 1772 | 1430 | 592 |
| WR | 0.462 | 0.541 | 0.587 | 0.640 | 3270 | 2688 | 2100 | 717 |
| TE | 0.302 | 0.434 | 0.505 | 0.560 | 1134 | 838 | 585 | 115 |
| DL | 0.229 | 0.308 | 0.362 | 0.379 | 5375 | 4454 | 3500 | 1018 |
| LB | 0.429 | 0.483 | 0.506 | 0.527 | 4155 | 3404 | 2711 | 1157 |
| DB | 0.296 | 0.321 | 0.339 | 0.301 | 6247 | 5084 | 4012 | 1468 |
| OL | 0.206 | 0.312 | 0.346 | 0.407 | 653 | 640 | 612 | 496 |
| P | 0.261 | 0.334 | 0.442 | 0.504 | 655 | 607 | 557 | 386 |

With EPA credit and fitted weights the offensive composites are far more
stable than the trait-only grade was: RB 0.351 at 8 games, WR 0.587, TE
0.505 and QB 0.424.

## Portability

Development pairs only (2020-2021 through 2023-2024), so the 2025 season
plays no part in the weights. Final-week values of qualified players.

| Group | Trait | Stayers | Movers | Stay slope | Move slope | Move slope SE | Portable share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QB | qb.ball_security | 235 | 81 | 0.043 | 0.046 | 0.177 | 1.07 |
| QB | qb.drive_engine | 235 | 81 | 0.202 | 0.291 | 0.111 | 1.44 |
| QB | qb.explosive | 235 | 81 | 0.407 | 0.146 | 0.156 | 0.36 |
| QB | qb.floor | 204 | 72 | 0.254 | 0.171 | 0.123 | 0.67 |
| QB | qb.legs | 235 | 81 | 0.117 | 0.277 | 0.098 | 2.37 |
| QB | qb.pressure_handling | 235 | 81 | 0.159 | 0.255 | 0.082 | 1.60 |
| QB | qb.receiver_independence | 28 | 12 | 0.108 | 0.276 | 0.492 |  |
| RB | rb.ball_security | 673 | 115 | -0.010 | -0.061 | 0.132 |  |
| RB | rb.breakaway | 671 | 115 | 0.217 | 0.094 | 0.086 | 0.43 |
| RB | rb.line_independence | 664 | 110 | 0.076 | 0.238 | 0.086 | 3.15 |
| RB | rb.receiving | 643 | 112 | 0.141 | 0.258 | 0.101 | 1.83 |
| RB | rb.short_yardage | 605 | 108 | 0.108 | 0.103 | 0.087 | 0.96 |
| RB | rb.stuff_avoidance | 664 | 110 | 0.084 | 0.122 | 0.092 | 1.45 |
| RB | rb.yards_over_expected | 671 | 115 | 0.192 | 0.238 | 0.084 | 1.24 |
| WR | rec.explosive | 952 | 200 | 0.159 | 0.119 | 0.096 | 0.75 |
| WR | rec.late_down | 935 | 191 | 0.147 | 0.142 | 0.060 | 0.97 |
| WR | rec.qb_independence | 947 | 199 | 0.204 | 0.180 | 0.069 | 0.88 |
| WR | rec.target_share | 956 | 201 | 0.598 | 0.212 | 0.063 | 0.35 |
| WR | rec.value_per_target | 952 | 200 | 0.332 | 0.251 | 0.066 | 0.75 |
| TE | rec.explosive | 286 | 34 | 0.101 | 0.586 | 0.292 | 5.80 |
| TE | rec.late_down | 269 | 32 | 0.121 | 0.106 | 0.154 | 0.88 |
| TE | rec.qb_independence | 286 | 34 | 0.136 | 0.545 | 0.191 | 4.00 |
| TE | rec.target_share | 288 | 34 | 0.699 | 0.165 | 0.165 | 0.24 |
| TE | rec.value_per_target | 286 | 34 | 0.231 | 0.323 | 0.168 | 1.40 |
| DL | def.every_week | 1879 | 224 | 0.373 | 0.204 | 0.061 | 0.55 |
| DL | def.finishing | 1593 | 198 | 0.060 | 0.092 | 0.066 | 1.54 |
| DL | def.havoc | 1879 | 224 | 0.537 | 0.332 | 0.062 | 0.62 |
| DL | def.impact | 1372 | 165 | -0.032 | -0.022 | 0.078 |  |
| DL | def.pass_rush | 1879 | 224 | 0.225 | 0.302 | 0.088 | 1.34 |
| DL | def.run_stop | 1879 | 224 | 0.288 | 0.100 | 0.057 | 0.35 |
| LB | def.coverage | 1394 | 162 | 0.285 | 0.151 | 0.068 | 0.53 |
| LB | def.every_week | 1394 | 162 | 0.332 | 0.211 | 0.075 | 0.63 |
| LB | def.finishing | 1039 | 115 | 0.026 | 0.170 | 0.102 | 6.47 |
| LB | def.havoc | 1394 | 162 | 0.435 | 0.259 | 0.079 | 0.60 |
| LB | def.impact | 694 | 100 | 0.017 | 0.019 | 0.112 | 1.12 |
| LB | def.pass_rush | 1394 | 162 | 0.249 | 0.201 | 0.078 | 0.81 |
| LB | def.run_stop | 1394 | 162 | 0.289 | 0.151 | 0.081 | 0.52 |
| LB | def.tackle_share | 1394 | 162 | 0.641 | 0.413 | 0.067 | 0.64 |
| DB | def.ball_production | 2048 | 281 | 0.369 | 0.186 | 0.061 | 0.50 |
| DB | def.every_week | 2048 | 281 | 0.302 | 0.179 | 0.059 | 0.59 |
| DB | def.havoc | 2048 | 281 | 0.348 | 0.172 | 0.061 | 0.50 |
| DB | def.impact | 1251 | 175 | 0.013 | -0.126 | 0.079 | -9.68 |
| DB | def.pass_impact | 1251 | 175 | 0.005 | -0.038 | 0.076 | -7.07 |
| DB | def.run_support | 2048 | 281 | 0.448 | 0.158 | 0.053 | 0.35 |
| OL | ol.line_yards | 506 | 0 | 0.375 |  |  |  |
| OL | ol.protection | 506 | 0 | 0.288 |  |  |  |
| OL | ol.short_yardage | 506 | 0 | 0.103 |  |  |  |
| OL | ol.stuff_rate | 506 | 0 | 0.302 |  |  |  |
| K | k.field_goal | 281 | 24 | 0.233 | 0.186 | 0.193 | 0.80 |
| P | p.punt_average | 387 | 20 | 0.446 | 0.258 | 0.212 | 0.58 |
| QB | epa.credit | 235 | 81 | 0.433 | 0.401 | 0.123 | 0.93 |
| RB | epa.credit | 673 | 115 | 0.511 | 0.374 | 0.093 | 0.73 |
| WR | epa.credit | 953 | 201 | 0.526 | 0.215 | 0.069 | 0.41 |
| TE | epa.credit | 288 | 34 | 0.543 | 0.480 | 0.200 | 0.88 |
| DL | epa.credit | 1729 | 209 | 0.167 | 0.070 | 0.081 | 0.42 |
| LB | epa.credit | 1251 | 148 | 0.123 | -0.007 | 0.081 | -0.05 |
| DB | epa.credit | 1880 | 262 | 0.066 | 0.081 | 0.057 | 1.24 |
| K | epa.credit | 3 | 0 | 0.180 |  |  |  |
| P | epa.credit | 18 | 1 | -0.061 |  |  |  |

Shares above 1 are clipped to 1 in the weights. Most of them come from
small stayer slopes with wide mover standard errors (TE explosive 5.80,
LB finishing 6.47), so they say the trait is not worse after a move, not
that it improves.

## Next-season validation

The honest development numbers are the dev-pair table above, where the
fitted weights were scored out of sample. The final model's own
development rows are in the evaluation file's `next_season` table; they are
in sample for the fitted weights and are not repeated here.

The 2024 to 2025 holdout, computed once with the final model, over players
graded at the final week of both seasons at the same position:

| Group | Players | Grade to next grade | Grade to next EPA | EPA to next EPA | Better at next EPA |
| --- | --- | --- | --- | --- | --- |
| QB | 82 | 0.560 | 0.571 | 0.546 | grade |
| RB | 178 | 0.275 | 0.274 | 0.256 | grade |
| WR | 270 | 0.424 | 0.398 | 0.397 | grade |
| TE | 81 | 0.368 | 0.307 | 0.294 | grade |
| DL | 453 | 0.354 | 0.124 | 0.106 | grade |
| LB | 341 | 0.367 | 0.185 | 0.103 | grade |
| DB | 517 | 0.257 | 0.135 | 0.085 | grade |
| OL | 134 | 0.329 |  |  |  |
| P | 70 | 0.557 | -0.054 |  |  |

Grade to next grade is the grade's own persistence and is not comparable
with EPA to next EPA, which predicts a different number. The fair test is
the two next-EPA columns. OL has no v4 value; the P grade-to-EPA column
covers only the few punters with one.

## Draft check

Rank correlation of the final-week grade and the v4 value with draft slot
(pick negated, so higher is better) and with NFL scrimmage snaps over the
first three seasons. Draft year y reads season y-1, so draft classes 2021
through 2026 enter the pick columns and only the mature classes 2021
through 2023 enter the NFL columns. Every season except 2025 is a
development season, and the fitted weights were fit on next-season
production in those seasons, so these are in sample; only the 2026 class
is out of sample. A player needs both a grade and a v4 value. They are
reported for the draft board and were never used to set a weight.

| Group | Players | Pick, v4 | Pick, grade | NFL snaps, v4 | NFL snaps, grade |
| --- | --- | --- | --- | --- | --- |
| QB | 61 | 0.418 | 0.460 | 0.032 | 0.144 |
| RB | 109 | 0.417 | 0.397 | 0.386 | 0.360 |
| WR | 179 | 0.433 | 0.402 | 0.268 | 0.308 |
| TE | 82 | 0.388 | 0.375 | 0.148 | 0.179 |
| DL | 253 | 0.212 | 0.283 | 0.037 | 0.122 |
| LB | 146 | 0.077 | 0.085 | 0.064 | 0.098 |
| DB | 285 | 0.250 | 0.206 | 0.210 | 0.203 |

The grade ranks the draft better than v4 at QB (0.460 against 0.418 on
slot, 0.144 against 0.032 on snaps) and DL (0.283 against 0.212, 0.122
against 0.037), and is ahead on NFL snaps at WR, TE and LB. It is behind
on slot at RB, WR, TE and DB. OL units have no v4 value and do not enter.

## Data coverage

Target tagging in `/plays/stats` is uneven. The builder recovers targets
from play text where it names a receiver on the roster. A season whose share
of incompletions with no target from either source is over 0.5 switches its
target-based traits to receptions.

| Season | Incompletions | Feed target | Text target | Uncovered | Uncovered share | Mode |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 12970 | 9018 | 911 | 3041 | 0.234 | targets |
| 2021 | 20052 | 9201 | 2192 | 8659 | 0.432 | targets |
| 2022 | 33022 | 8384 | 9346 | 15292 | 0.463 | targets |
| 2023 | 33276 | 6745 | 8965 | 17566 | 0.528 | receptions |
| 2024 | 35488 | 3610 | 7323 | 24555 | 0.692 | receptions |
| 2025 | 36489 | 16874 | 8960 | 10655 | 0.292 | targets |

2023 and 2024 ran in receptions mode, so two of the five development
seasons measure the receiver traits on receptions. Every season, served or
backtested, decides its own mode from its own coverage, and the scored
artifact records it in a `poor_coverage` column. The 2026 data is in
receptions mode (uncovered share 0.699).

Portal rows are matched to an athlete id by name on both the destination
roster of the portal season and the origin roster of the season before.
The rate is matched rows with an FBS destination over rows naming an FBS
destination.

| Portal season | Portal rows | Rows naming an FBS destination | Matched | Matched with an FBS destination | Rate |
| --- | --- | --- | --- | --- | --- |
| 2021 | 1770 | 816 | 566 | 514 | 0.630 |
| 2022 | 2273 | 1092 | 1021 | 845 | 0.774 |
| 2023 | 2502 | 1418 | 1300 | 1223 | 0.862 |
| 2024 | 3378 | 2066 | 1963 | 1697 | 0.821 |
| 2025 | 4499 | 2905 | 2857 | 2361 | 0.813 |

Portability uses the 2021 through 2024 portal seasons only. 2025 transfers
are matched and counted here but feed no weight.

## Scoring 2026 and serving

`position-traits --seasons 2026` scores with the frozen model (log
`output/position-grade-2026-202610162323.log`). Local raw data for 2026 ends
at week 1, so the artifact holds weeks 0 and 1: 72,447 rows, with 4,737
qualified FBS players graded and 2,914 war rows. The artifact now also
carries `epa.credit` trait rows. It ran in receptions mode
and used the frozen expectation tables, reading no earlier play-by-play
season. There is no calibration replay for 2026, so the WAR leverage came
from the stored weekly serving anchors for weeks 0 and 1.

The coverage rule in week 1:

| Group | Qualified FBS | Graded | No primary trait | Coverage under 0.6 |
| --- | --- | --- | --- | --- |
| QB | 179 | 157 | 17 | 5 |
| RB | 391 | 381 | 10 | 0 |
| WR | 613 | 512 | 101 | 0 |
| TE | 239 | 199 | 40 | 0 |
| DL | 1100 | 1092 | 0 | 8 |
| LB | 840 | 840 | 0 | 0 |
| DB | 1288 | 1288 | 0 | 0 |
| OL | 137 | 137 | 0 | 0 |
| P | 126 | 126 | 0 | 0 |

With EPA credit carrying most of the offensive weight, a player without a
v4 value yet falls under the 0.6 bar: 5 QBs and 8 defensive linemen in
week 1. In receptions mode many receivers have no value per target
yet, which is why 101 WRs and 40 TEs lack the primary trait.

`backtest.score_cached_season(season)` scores an already-built season from
its cached snapshot, coverage and credit with the latest model before that
season, without rebuilding anything or reading the expectation tables, and
writes `processed/players/position_traits/scored_<season>.parquet` with the
runner's columns. For 2025 it wrote 1,208,464 rows, 81,403 of them grade
rows and 66,626 war rows (log
`output/position-grade-scored-2025-202610162323.log`), for a local preview
of the validation season.

Each artifact row carries the trait snapshot columns plus `uncertainty`
(empty on trait rows), `in_grade` (true on a trait row whose trait has
positive weight at the position, and on every grade row),
`points_above_replacement` (only on war rows), `fbs` and `poor_coverage`.
Grade rows have `trait = "grade"`, the grade as `value`, its percentile as
`percentile` and `kind = "graded"`. War rows have `trait = "war"`, wins
above replacement as `value`, its percentile among qualified FBS players
at the position that week, and `kind = "measured"`. The publisher keeps
only the latest week and only FBS rows; the table has no `fbs` or
`poor_coverage` column, and the migration
`sql/20261016200339_position_traits.sql` (not yet applied) declares
`in_grade`, `uncertainty` and `points_above_replacement`.

The frozen model is two files in `backend/data/processed/players/models/`:
`position_grade_2024.json` (weights, signs, statuses and the composite
reliability curve) and `position_grade_expectations_2024.parquet` (the
rushing, conversion and field goal baselines, fit on every play season from
2019 through 2025). The weekly runner loads both and refits neither. The
pair travels in the reviewed player runtime bundle (`export_runtime_bundle`
adds it, `install_runtime_bundle` accepts a bundle with or without it and
refuses one that does not precede the season), and every weekly run
uploads it as the `position-traits-fit-SEASON` artifact. The first
production run needs a re-exported bundle.

The draft research watchlist reads the trait artifact and adds `grade`,
`grade_percentile` and `grade_games` beside the v4 columns, and its audit
lists `graded_groups`. It never overwrites v4's value, rank or model
version.

## Wins above replacement

v4 measures a player's points above positional replacement through week w:
`points_above_replacement = games x (shrunk_per_game - replacement_per_game)`,
its `value_above_replacement`. A point is worth more in a close game than in
a rout, so WAR weights each game the player appeared in by the win
probability a point buys at the team model's pregame projection:

- `leverage_g = phi(home_margin_g / margin_sd_g) / margin_sd_g`, the normal
  density of the projected margin over its sd, divided by the sd: the slope
  of win probability per point at the projection;
- `wins_above_replacement = (shrunk_per_game - replacement_per_game) x
  sum(leverage_g)` over his credited games through week w.

The projection is the calibration replay
(`backend.serving.anchors.load_serving_anchors`), each game projected only
from earlier games; a game without a projection takes the season's mean
leverage. When a season has no replay (2026), the stored weekly serving
anchors stand in. Nothing is fitted.

The 2025 leaders at the final week (week 17), qualified FBS players at any
position, from the fit log:

| Rank | Player | Team | Group | Games | Points | Wins |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Diego Pavia | Vanderbilt | QB | 13 | 101.1 | 2.08 |
| 2 | Trinidad Chambliss | Ole Miss | QB | 15 | 90.1 | 1.88 |
| 3 | Brad Jackson | Texas State | QB | 13 | 76.7 | 1.59 |
| 4 | Darian Mensah | Duke | QB | 14 | 70.5 | 1.57 |
| 5 | Haynes King | Georgia Tech | QB | 12 | 78.2 | 1.53 |
| 6 | Fernando Mendoza | Indiana | QB | 16 | 85.4 | 1.50 |
| 7 | Taylen Green | Arkansas | QB | 12 | 64.9 | 1.46 |
| 8 | Anthony Colandrea | UNLV | QB | 14 | 69.4 | 1.45 |
| 9 | Gunner Stockton | Georgia | QB | 14 | 69.3 | 1.44 |
| 10 | Brendan Sorsby | Cincinnati | QB | 12 | 69.0 | 1.44 |

Every leader is a quarterback. That is v4's credit, which gives passers the
largest share of every pass. Leverage reorders players with similar points:
Mendoza has more points than King (85.4 against 78.2) but fewer wins
(1.50 against 1.53), because Indiana was projected to win its games by
more.

Defensive credit in v4 covers only tagged disruptions (sacks,
interceptions, breakups and forced and recovered fumbles on plays, plus box
score tackles for loss and passes defended), so defensive WAR is undercounted and a defender without
credited plays has no war row. OL units have no v4 value and no WAR.

## Limits

There are no tackle attributions on plays, no snap counts, no pressures and
no coverage assignments, so defensive traits rest on box score counts and
with-without comparisons. Offensive linemen are rated as a unit, one row
per team, and a unit never transfers, so OL portability cannot be
measured. 2019 has no opponent prior, so development
starts in 2020. Targets are missing for most incompletions in 2023, 2024
and 2026, which ran in receptions mode. Win probability is missing for
games with an FCS side.

The grade's volume terms are the coverage rule, the z cap and, at QB, TE
and DB, shrinkage by games. The fitted weights are fit on next-season v4
credit, so where EPA credit carries most of the weight the grade is
largely v4 with trait corrections, and it inherits v4's credit rules. The uncertainty band reads
only games, not how many traits a player has. Two of the development seasons are in receptions mode. The
weights are as precise as split-half correlations on hundreds to thousands
of players allow, which is about two decimals; the QB receiver independence
weight rests on 36 players and should be read as rough. 2026 was never used
and is judged only as it is served.

## Findings

- **The grade now predicts next season's production better than EPA credit
  alone at every position with a v4 baseline, on the holdout.** Grade to
  next EPA against EPA to next EPA, 2024 to 2025: QB 0.571 against 0.546,
  RB 0.274 against 0.256, WR 0.398 against 0.397, TE 0.307 against 0.294,
  DL 0.124 against 0.106, LB 0.185 against 0.103, DB 0.135 against 0.085.
  At WR the margin is 0.001, a tie.
- **On offense most of that comes from EPA credit itself.** The fitted
  weights put 0.555 to 0.844 of the weight on `epa.credit` at QB, RB, WR and
  TE. The traits add a correction: on the dev pairs the grade beats EPA
  alone at QB (0.432 against 0.422) and WR (0.387 against 0.366), and still
  trails it at RB (0.360 against 0.364) and TE (0.343 against 0.353).
  Offensive grades are not trait-only profiles any more.
- **On defense the traits carry it.** EPA credit has no weight at LB and DB
  and 0.064 at DL. The grade beats defensive EPA credit on every dev pair
  at DL and DB (narrowly in 2021 to 2022 at DB, 0.112 against 0.106) and
  on three of four at LB. On the holdout the defensive numbers are slightly
  lower than the previous trait-only grade gave (DL 0.124 against 0.126,
  LB 0.185 against 0.199, DB 0.135 against 0.137, from
  `output/position-grade-fit-202610162203.log`), though fitted weights
  raised the dev-pair means at all three. That small dev gain did not carry
  to 2025.
- **Fitted weights were the largest gain at WR** (0.268 to 0.387 on the
  dev pairs, scored out of sample); EPA credit was the largest at QB, RB
  and TE. Shrinkage helped by 0.003 or less where it was kept.
- **The grade ranks the draft better than v4 at QB and DL**, and on NFL
  snaps also at WR, TE and LB. These are in-sample checks except the 2026
  class.
- **WAR is a QB list.** All ten 2025 leaders are quarterbacks, because v4
  credits passers with the largest share of every pass and defenders only
  for tagged disruptions. WAR compares players within a position; across
  positions it mostly reflects the credit rules.
- **The 80 clip is crowded at DL.** In the 2025 week 17 scored file, 16
  defensive linemen sit at 80 (`scored_2025.parquet`). The fitted DL weights
  rest on havoc and pass rush, whose capped z-scores bunch at the top.

## Punters, four traits (2026-10-16 refit)

Refit with `position-traits-fit --dev-seasons 2020 2021 2022 2023 2024
--validate 2025 --rebuild` after the punter traits moved to
`backend/players/traits/punting.py` (`output/position-grade-fit-punters-20261016.log`).
Every other position's weights, coefficients and reliability curves are
unchanged to the last digit. Punters have no v4 credit, so the EPA-based
steps do not apply and the P grade is the reliability x portability weights:

| Trait | Reliability at 8 | Portability | Weight |
| --- | --- | --- | --- |
| p.punt_average | 0.497 | 0.580 | 0.389 |
| p.fair_catch_share | 0.293 | 0.661 | 0.261 |
| p.no_return_share | 0.418 | 0.379 | 0.213 |
| p.net | 0.333 | 0.303 | 0.137 |

Portability is measured on 20 movers against 380 stayers, exactly the sample
floor. The P composite's split-half reliability is 0.18, 0.29, 0.35 and 0.44
at 4, 6, 8 and 12 games, so it shrinks toward 50 with k = 12. Final-week
grade against the next season's final-week grade (Spearman): 0.390 (2020 to
2021, 62 punters), 0.501 (75), 0.339 (79), 0.333 (83); the held-out 2024 to
2025 pair is 0.266 over 70 punters. All 136 qualified 2025 punters were
graded. Kickers stay out.
