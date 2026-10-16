# Position traits v1: per-position trait ratings

## What the model is

Every FBS player gets a set of named traits at his position and, where the
fit finds signal, one headline number. A trait is a per-game rate built from
tagged plays or box score counts, adjusted for the opponent where the data
allows, and turned into a position percentile among qualified FBS players
that week. Traits are `counted` when the feed tags the player directly and
`inferred` when they come from unit outcomes or teammate comparisons. The
headline is the player's own points per game in the fitted model: his share
of his team's usage at the group times the fitted combination of his traits
and his player value v4 EPA credit, measured above the 30th percentile of
qualified FBS players at his group. Only qualified FBS players carry a
headline (see The headline number).

The headline weights come from one joint fit across positions. Each
completed FBS against FBS game from model week 2 on is one row. The target is
the home margin. Two regressors are unpenalized and never dropped: home field
and `projection`, the team model's pregame projected home margin for the
game. The penalized predictors are, for every group and trait, the home
team's aggregate minus the away team's, where a team's aggregate is the
usage-share-weighted sum of its players' standardized trait values in the
snapshot from the week before the game. The v4 EPA credit enters the same fit
as one more trait per group, so a trait earns weight only for what it adds
beyond EPA credit. Nothing from the week being rated enters its own rating.

The projection is there because the design asks for the margin net of the
opponent's team rating and home field. Without it, a roster aggregate earns
weight for carrying team quality the team model already knows: good teams
have good players. With it, a trait earns weight only for what it predicts
about the margin beyond the team model's own projection. The projection is
the team model's chronological replay
(`backend/data/processed/calibration/joint_scoring_predictions.parquet`,
read through `backend.serving.anchors.load_serving_anchors`, the source live
serving reads): each game's projection comes from a model trained only on
games before that week. Games without a projection are dropped (table below).

Development seasons are 2020 through 2024. 2025 is scored once with the
frozen weights as the independent check. 2026 was never fitted. The fit is
`backend/data/processed/players/models/position_traits_2024.json`, the full
evaluation is
`backend/data/processed/players/position_traits/evaluation_2024.parquet`, and
every table below is copied from those two files and from the run log of
`position-traits-fit --dev-seasons 2020 2021 2022 2023 2024 --validate 2025`
on 2026-10-16 (`output/position-traits-fit-20261016-projection.log`), unless
it says it was recomputed. Player values for 2020 through 2025 were rebuilt
under `cfb_player_value_v4` before the run, so the baseline is one model.

## Data coverage

Target tagging in `/plays/stats` is uneven. The builder recovers targets from
play text where it names a receiver on the roster. A season whose share of
incompletions with no target from either source is over 0.5 switches its
target-based traits to receptions.

| Season | Incompletions | Feed target | Text target | Uncovered | Uncovered share | Mode |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 12970 | 9018 | 911 | 3041 | 0.234 | targets |
| 2021 | 20052 | 9201 | 2192 | 8659 | 0.432 | targets |
| 2022 | 33022 | 8384 | 9346 | 15292 | 0.463 | targets |
| 2023 | 33276 | 6745 | 8965 | 17566 | 0.528 | receptions |
| 2024 | 35488 | 3610 | 7323 | 24555 | 0.692 | receptions |
| 2025 | 36489 | 16874 | 8960 | 10655 | 0.292 | targets |

2023 and 2024 ran in receptions mode. Every season, served or backtested,
decides its own mode from its own coverage, and the scored artifact records
it in a `poor_coverage` column. The 2026 week 1 data is in receptions mode
(uncovered share 0.699). The fit stores 2025's mode (targets) for
information only.

Every game in the matrix needs a pregame projection. The evaluation file's
`anchors` table counts the matrix rows before and after the join:

| Season | Games | With a projection | Dropped |
| --- | --- | --- | --- |
| 2020 | 492 | 492 | 0 |
| 2021 | 720 | 720 | 0 |
| 2022 | 723 | 723 | 0 |
| 2023 | 740 | 740 | 0 |
| 2024 | 759 | 759 | 0 |
| 2025 | 760 | 759 | 1 |

Portal rows are matched to an athlete id by name on both the destination
roster of the portal season and the origin roster of the season before
(`match_transfers`). The table is the evaluation file's `portal` table, built
by `backtest.portal_match_table`: it counts the portal rows, the rows whose
destination is an FBS school in that season's teams file, the rows
`match_transfers` tied to one athlete, and the matched rows whose destination
is FBS. The rate is the last count over the FBS-destination rows.

| Portal season | Portal rows | Rows naming an FBS destination | Matched | Matched with an FBS destination | Rate |
| --- | --- | --- | --- | --- | --- |
| 2021 | 1770 | 816 | 566 | 514 | 0.630 |
| 2022 | 2273 | 1092 | 1021 | 845 | 0.774 |
| 2023 | 2502 | 1418 | 1300 | 1223 | 0.862 |
| 2024 | 3378 | 2066 | 1963 | 1697 | 0.821 |
| 2025 | 4499 | 2905 | 2857 | 2361 | 0.813 |

Unmatched rows (portal rows minus matched) number 1204, 1252, 1202, 1415 and
1642. Most of them name no FBS destination: the rows without one (portal rows
minus FBS-destination rows) number 954, 1181, 1084, 1312 and 1594, and only
302, 247, 195, 369 and 544 rows naming an FBS destination stay unmatched.


## How the weights were chosen

Each feature column is standardized once over the development rows. A column
with no spread is dropped as constant; `OL:epa` is the only one, because v4
gives linemen no credit. The fit is a ridge in closed form with the
intercept, home field and projection unpenalized. The penalty is chosen from
36 log-spaced values between 0.01 and 100,000 by pooled
leave-one-season-out MAE over the five development seasons. A feature is
then dropped when its weight is under twice the larger of its
delete-one-season jackknife sd and its analytic ridge standard error, and
the rest are refit once at the same penalty.

The chosen penalty is 10,000, inside the grid. Recomputed with the same
functions on the same development rows, the pooled leave-one-season-out MAE
is 13.165 at 0.01, 13.072 at 1000, 13.028 at 6310, 13.027 at 10,000, 13.028
at 15,849 and 13.035 at 100,000. The curve is flat near its minimum, and the
whole spread between the largest penalty, where every trait weight is close
to zero, and the best one is 0.008 points. The penalty is large: the weights
are heavily shrunk.

## Weights

Weights are points of home margin per development-season standard deviation
of the feature; `projection` is points of margin per point of projected
margin. `loso_sd` is the sd of the weight across the five
leave-one-season-out refits. Two decimals is as much precision as these
spreads support. The spreads shown are from the final refit on the kept
features; the drop rule was applied to the first pass over every feature.
Every kept weight also clears the rule on the refit spreads.

| Feature | Weight | LOSO sd | Jackknife sd | Analytic SE |
| --- | --- | --- | --- | --- |
| intercept | -1.38 | 0.27 | 0.48 | 0.99 |
| home_field | 1.65 | 0.39 | 0.69 | 1.03 |
| projection | 1.07 | 0.01 | 0.03 | 0.03 |
| RB:rb.yards_over_expected | 0.14 | 0.01 | 0.02 | 0.06 |
| RB:rb.breakaway | 0.18 | 0.03 | 0.06 | 0.06 |
| RB:rb.ball_security | -0.22 | 0.04 | 0.07 | 0.07 |
| DL:def.havoc | 0.18 | 0.04 | 0.07 | 0.07 |
| OL:ol.line_yards | 0.16 | 0.02 | 0.04 | 0.06 |

The projection weight is 1.07: the margin moves slightly more than one for
one with the projection. Home field keeps 1.65 points beyond the projection,
against an intercept of -1.38, so a neutral-site game sits 1.38 below the
projection and a home game 0.27 above it.

Dropped from the headline and kept on the card as descriptive:

| Group | Dropped |
| --- | --- |
| QB | qb.drive_engine, qb.explosive, qb.ball_security, qb.floor, qb.legs, qb.pressure_handling, qb.receiver_independence, epa |
| RB | rb.stuff_avoidance, rb.short_yardage, rb.receiving, rb.line_independence, epa |
| WR | rec.target_share, rec.value_per_target, rec.explosive, rec.late_down, rec.qb_independence, epa |
| TE | rec.target_share, rec.value_per_target, rec.explosive, rec.late_down, rec.qb_independence, epa |
| DL | def.pass_rush, def.finishing, def.run_stop, def.every_week, def.impact, epa |
| LB | def.havoc, def.pass_rush, def.finishing, def.run_stop, def.every_week, def.tackle_share, def.coverage, def.impact, epa |
| DB | def.havoc, def.every_week, def.ball_production, def.run_support, def.impact, def.pass_impact, epa |
| OL | epa (constant), ol.protection, ol.stuff_rate, ol.short_yardage |
| K | k.field_goal, epa |
| P | p.punt_average, epa |

Once the team model's projection is in the fit, five features keep weight:
three running back traits (yards over expected, breakaway, ball security),
DL havoc and OL line yards. The v4 EPA credit keeps no weight at any group.
QB, WR, TE, LB, DB, K and P keep no feature, so they carry no headline. The
headline is not EPA per play with a new name; it is a small residual signal
at three groups, each weight under 0.25 points per standard deviation.

`RB:rb.ball_security` is negative (-0.22) although higher means fewer fumble
costs, and its split-half reliability is near zero. That weight is listed
under Findings.

## The headline number

`raw_score` is the fitted combination of a player's standardized traits and
EPA credit at his group: the margin a team would gain if he had all of its
usage there. The headline scales it by his actual usage. His share is his
exposure (dropbacks for a QB, carries for a back, targets for a receiver,
games for a defender, attempts for a specialist) over the summed exposure of
his team's players at the group that week; an OL unit has share 1. With
`replacement` the 30th percentile of `raw_score` among qualified FBS players
at the group and `sd` their standard deviation:

- `headline = share x (raw_score - replacement)`, points per game the player
  adds over a replacement-level player in his role;
- `headline_percentile` ranks the headline among qualified FBS players at
  the group, the same pool the trait percentiles use;
- `uncertainty = share x sd x sqrt(1 - r)`, with `r` the headline's
  split-half reliability at the player's game count.

Only qualified FBS players get a headline row; unqualified and non-FBS
players keep their trait rows. A group whose every feature was dropped (QB,
WR, TE, LB, DB, K and P in this fit) gets no headline rows, `replaces_value`
False and no trait counted `in_headline`. The projection is a game-level
offset, not a player feature, so it plays no part in scoring.

## 2025 validation

Next-game margin MAE of the frozen development fit on 2025, against a fit of
the v4 EPA credit columns alone on the same development seasons. Both fits
carry home field and the projection. The v4 fit keeps no EPA column (its
weights are intercept -1.40, home field 1.61 and projection 1.09, recomputed
with the same functions), so the v4 column is in effect the team model's
projection alone.

| Week bucket | Games | traits | v4 | v4 minus traits |
| --- | --- | --- | --- | --- |
| 2-5 | 198 | 13.065 | 13.056 | -0.009 |
| 6-9 | 218 | 12.205 | 12.237 | 0.031 |
| 10+ | 343 | 12.913 | 12.930 | 0.016 |
| all | 759 | 12.749 | 12.764 | 0.014 |

The traits beat the projection alone by 0.014 points over the season,
lose by 0.009 in weeks 2 to 5 and win by 0.016 to 0.031 later. That is a
small effect on 759 games and should not be read as more than a sign.

The previous run, superseded by this one, fit the same features to the
margin with home field and no projection. It validated at 14.386 for the
traits against 15.046 for v4 (760 games), a gain of 0.660, kept 18 features
including EPA credit at QB and RB and traits at QB, RB, TE, DL, DB, OL and K,
and replaced v4 at DL and DB. The difference is the projection: most of that
gain was team quality the roster aggregates carried and the team model
already had. With the projection in the fit the traits MAE falls by 1.64
points and v4's by 2.28, and the gap between them nearly closes.

## Which positions replace v4

The swap test refits on the development seasons with one group's full
feature set and only the EPA column of every other group, then scores 2025.
Rene's rule: the headline replaces v4 at a group when the improvement over
the v4-only MAE is above zero and the headline's six-game split-half
reliability is at least v4's, both measured on the same players and games. A
group with no headline cannot replace v4.

| Group | MAE v4 only | MAE with traits | Improvement | Reliability 6, v4 | Reliability 6, headline | Replaces v4 |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 12.764 | 12.768 | -0.004 | 0.463 |  | False |
| RB | 12.764 | 12.757 | 0.006 | 0.334 | 0.080 | False |
| WR | 12.764 | 12.764 | 0.000 | 0.375 |  | False |
| TE | 12.764 | 12.764 | 0.000 | 0.311 |  | False |
| DL | 12.764 | 12.747 | 0.017 | 0.149 | 0.246 | True |
| LB | 12.764 | 12.764 | 0.000 | 0.199 |  | False |
| DB | 12.764 | 12.764 | 0.000 | 0.092 |  | False |
| OL | 12.764 | 12.745 | 0.019 |  |  | False |
| K | 12.764 | 12.764 | 0.000 | -0.004 |  | False |
| P | 12.764 | 12.764 | 0.000 | -0.084 |  | False |

The headline replaces v4 at DL only: an improvement of 0.017 points and a
six-game reliability of 0.246 against v4's 0.149. RB improves the margin by
0.006 but its headline is far less reliable than v4 (0.080 against 0.334).
OL improves the most (0.019), but v4 has no OL rating to compare against, and
OL, K and P never replace v4 in v1. Where the improvement is 0.000 the swap
fit kept nothing beyond the projection. Everything except DL ships beside v4.

## Reliability

Split-half reliability correlates a player's odd-game value with his
even-game value at a fixed game count. The headline and v4 rows are pooled
over the development seasons and paired: same athletes, same games. The
headline side correlates `raw_score`, the share-1 value, and follows the
headline's own rule, so non-FBS athletes drop out of both sides; a group with
no headline keeps its v4 side.

| Group | v4 4 | Headline 4 | v4 6 | Headline 6 | v4 8 | Headline 8 | v4 12 | Headline 12 | Players at 6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QB | 0.359 |  | 0.463 |  | 0.465 |  | 0.477 |  | 763 |
| RB | 0.271 | 0.045 | 0.334 | 0.080 | 0.366 | 0.088 | 0.451 | 0.153 | 1792 |
| WR | 0.305 |  | 0.375 |  | 0.428 |  | 0.502 |  | 2725 |
| TE | 0.209 |  | 0.311 |  | 0.400 |  | 0.475 |  | 850 |
| DL | 0.021 | 0.102 | 0.149 | 0.246 | 0.217 | 0.332 | 0.266 | 0.411 | 4033 |
| LB | 0.106 |  | 0.199 |  | 0.253 |  | 0.295 |  | 3005 |
| DB | 0.018 |  | 0.092 |  | 0.127 |  | 0.114 |  | 4596 |
| OL |  |  |  |  |  |  |  |  |  |
| K |  |  | -0.004 |  | 0.013 |  |  |  | 14 |
| P | -0.180 |  | -0.084 |  | -0.150 |  | 0.090 |  | 61 |

Per-trait reliability is computed separately for each development season.
The table shows the mean over 2020 through 2024 weighted by players; the
per-season rows are in the evaluation file.

| Group | Trait | 4 | 6 | 8 | 12 | Players at 8 | In headline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QB | qb.ball_security | -0.003 | 0.026 | 0.014 | -0.005 | 571 |  |
| QB | qb.drive_engine | 0.143 | 0.148 | 0.187 | 0.188 | 572 |  |
| QB | qb.explosive | 0.149 | 0.239 | 0.288 | 0.417 | 571 |  |
| QB | qb.floor |  |  | 0.245 | 0.193 | 495 |  |
| QB | qb.legs | 0.096 | 0.089 | 0.162 | 0.170 | 524 |  |
| QB | qb.pressure_handling | 0.195 | 0.297 | 0.339 | 0.483 | 566 |  |
| QB | qb.receiver_independence | 0.223 | 0.360 |  |  |  |  |
| RB | rb.ball_security | -0.002 | 0.010 | -0.054 | -0.121 | 1438 | yes |
| RB | rb.breakaway | 0.063 | 0.124 | 0.168 | 0.200 | 1380 | yes |
| RB | rb.line_independence | 0.109 | 0.061 | 0.110 | 0.216 | 1085 |  |
| RB | rb.receiving | 0.063 | 0.108 | 0.162 | 0.251 | 564 |  |
| RB | rb.short_yardage | 0.014 | 0.018 | -0.046 |  | 402 |  |
| RB | rb.stuff_avoidance | 0.064 | 0.079 | 0.106 | 0.167 | 1085 |  |
| RB | rb.yards_over_expected | 0.091 | 0.136 | 0.153 | 0.196 | 1380 | yes |
| WR | rec.explosive | 0.085 | 0.144 | 0.178 | 0.215 | 2027 |  |
| WR | rec.late_down | 0.085 | 0.069 | 0.130 | -0.054 | 660 |  |
| WR | rec.qb_independence | 0.057 | 0.104 | 0.113 | 0.161 | 1864 |  |
| WR | rec.target_share | 0.600 | 0.675 | 0.714 | 0.728 | 2110 |  |
| WR | rec.value_per_target | 0.094 | 0.121 | 0.136 | 0.222 | 2027 |  |
| TE | rec.explosive | 0.029 | 0.098 | 0.126 | 0.065 | 566 |  |
| TE | rec.late_down | 0.062 | 0.102 | 0.089 |  | 84 |  |
| TE | rec.qb_independence | 0.019 | 0.128 | 0.095 | 0.167 | 524 |  |
| TE | rec.target_share | 0.529 | 0.620 | 0.711 | 0.746 | 593 |  |
| TE | rec.value_per_target | 0.026 | 0.106 | 0.152 | 0.117 | 566 |  |
| DL | def.every_week | 0.171 | 0.218 | 0.267 | 0.344 | 3501 |  |
| DL | def.finishing | 0.099 | 0.068 | 0.063 |  | 319 |  |
| DL | def.havoc | 0.236 | 0.310 | 0.362 | 0.421 | 3501 | yes |
| DL | def.pass_rush | 0.031 | 0.067 | 0.119 | 0.156 | 3497 |  |
| DL | def.run_stop | 0.094 | 0.150 | 0.170 | 0.263 | 3501 |  |
| LB | def.coverage | 0.118 | 0.165 | 0.184 | 0.190 | 2718 |  |
| LB | def.every_week | 0.274 | 0.329 | 0.353 | 0.359 | 2718 |  |
| LB | def.finishing | 0.040 | 0.088 | 0.136 |  | 134 |  |
| LB | def.havoc | 0.314 | 0.388 | 0.421 | 0.462 | 2718 |  |
| LB | def.pass_rush | 0.082 | 0.120 | 0.139 | 0.182 | 2713 |  |
| LB | def.run_stop | 0.205 | 0.254 | 0.283 | 0.333 | 2718 |  |
| LB | def.tackle_share | 0.666 | 0.736 | 0.769 | 0.792 | 2710 |  |
| DB | def.ball_production | 0.203 | 0.229 | 0.260 | 0.291 | 4063 |  |
| DB | def.every_week | 0.223 | 0.241 | 0.257 | 0.219 | 4063 |  |
| DB | def.havoc | 0.230 | 0.267 | 0.288 | 0.280 | 4063 |  |
| DB | def.run_support | 0.348 | 0.432 | 0.476 | 0.532 | 4063 |  |
| OL | ol.line_yards | 0.246 | 0.330 | 0.350 | 0.384 | 612 | yes |
| OL | ol.protection | 0.206 | 0.255 | 0.315 | 0.429 | 609 |  |
| OL | ol.short_yardage | 0.039 | -0.009 | 0.028 | 0.157 | 602 |  |
| OL | ol.stuff_rate | 0.143 | 0.193 | 0.221 | 0.297 | 612 |  |
| K | k.field_goal | 0.015 | 0.047 | 0.041 | 0.062 | 566 |  |
| P | p.punt_average | 0.324 | 0.400 | 0.495 | 0.516 | 794 |  |

`def.impact` and `def.pass_impact` have no rows: they are cumulative weekly
estimates with no per-game value, so split-half reliability does not apply
to them. The card's uncertainty band reads the headline rows of the first
table.

## Portability

Portability regresses each player's final-week trait value in season t on
his value in season t-1, separately for players who stayed and for matched
transfers, pooled over consecutive pairs from 2020-2021 through 2024-2025.
`portable_share` is the mover slope over the stayer slope, shown only when
the stayer slope is positive and at least 20 players moved.

| Group | Trait | Stayers | Movers | Stay slope | Move slope | Move slope SE | Portable share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QB | qb.ball_security | 295 | 103 | 0.048 | 0.049 | 0.102 | 1.02 |
| QB | qb.drive_engine | 295 | 103 | 0.226 | 0.239 | 0.100 | 1.06 |
| QB | qb.explosive | 295 | 103 | 0.393 | 0.056 | 0.131 | 0.14 |
| QB | qb.floor | 257 | 90 | 0.288 | 0.201 | 0.110 | 0.70 |
| QB | qb.legs | 295 | 103 | 0.159 | 0.179 | 0.094 | 1.12 |
| QB | qb.pressure_handling | 295 | 103 | 0.133 | 0.526 | 0.118 | 3.96 |
| QB | qb.receiver_independence | 33 | 15 | 0.016 | 0.027 | 0.373 |  |
| RB | rb.ball_security | 804 | 162 | -0.011 | -0.076 | 0.128 |  |
| RB | rb.breakaway | 802 | 162 | 0.210 | 0.250 | 0.085 | 1.19 |
| RB | rb.line_independence | 793 | 157 | 0.075 | 0.195 | 0.072 | 2.59 |
| RB | rb.receiving | 767 | 159 | 0.104 | 0.172 | 0.086 | 1.65 |
| RB | rb.short_yardage | 724 | 152 | 0.081 | 0.056 | 0.078 | 0.68 |
| RB | rb.stuff_avoidance | 793 | 157 | 0.087 | 0.050 | 0.074 | 0.57 |
| RB | rb.yards_over_expected | 802 | 162 | 0.199 | 0.229 | 0.074 | 1.15 |
| WR | rec.explosive | 1129 | 293 | 0.111 | 0.035 | 0.068 | 0.32 |
| WR | rec.late_down | 1110 | 284 | 0.034 | 0.007 | 0.052 | 0.21 |
| WR | rec.qb_independence | 1122 | 292 | 0.175 | 0.111 | 0.052 | 0.63 |
| WR | rec.target_share | 1134 | 294 | 0.592 | 0.199 | 0.054 | 0.34 |
| WR | rec.value_per_target | 1129 | 293 | 0.156 | -0.026 | 0.058 | -0.17 |
| TE | rec.explosive | 350 | 51 | 0.097 | 0.589 | 0.203 | 6.08 |
| TE | rec.late_down | 327 | 48 | 0.054 | 0.080 | 0.155 | 1.48 |
| TE | rec.qb_independence | 350 | 51 | 0.128 | 0.315 | 0.137 | 2.45 |
| TE | rec.target_share | 354 | 51 | 0.639 | 0.220 | 0.114 | 0.34 |
| TE | rec.value_per_target | 350 | 51 | 0.129 | 0.165 | 0.167 | 1.28 |
| DL | def.every_week | 2237 | 349 | 0.376 | 0.241 | 0.048 | 0.64 |
| DL | def.finishing | 1904 | 308 | 0.054 | 0.117 | 0.055 | 2.15 |
| DL | def.havoc | 2237 | 349 | 0.540 | 0.334 | 0.047 | 0.62 |
| DL | def.impact | 1632 | 248 | -0.013 | 0.002 | 0.063 |  |
| DL | def.pass_rush | 2237 | 349 | 0.225 | 0.147 | 0.050 | 0.66 |
| DL | def.run_stop | 2237 | 349 | 0.297 | 0.153 | 0.047 | 0.52 |
| LB | def.coverage | 1696 | 231 | 0.272 | 0.110 | 0.057 | 0.41 |
| LB | def.every_week | 1696 | 231 | 0.320 | 0.217 | 0.061 | 0.68 |
| LB | def.finishing | 1269 | 169 | 0.033 | 0.223 | 0.083 | 6.84 |
| LB | def.havoc | 1696 | 231 | 0.431 | 0.256 | 0.064 | 0.59 |
| LB | def.impact | 847 | 144 | 0.026 | 0.065 | 0.097 | 2.50 |
| LB | def.pass_rush | 1696 | 231 | 0.272 | 0.189 | 0.061 | 0.69 |
| LB | def.run_stop | 1696 | 231 | 0.297 | 0.152 | 0.064 | 0.51 |
| LB | def.tackle_share | 1696 | 231 | 0.635 | 0.364 | 0.060 | 0.57 |
| DB | def.ball_production | 2483 | 425 | 0.379 | 0.192 | 0.047 | 0.51 |
| DB | def.every_week | 2483 | 425 | 0.304 | 0.154 | 0.048 | 0.51 |
| DB | def.havoc | 2483 | 425 | 0.351 | 0.149 | 0.048 | 0.43 |
| DB | def.impact | 1511 | 252 | 0.035 | -0.193 | 0.066 | -5.57 |
| DB | def.pass_impact | 1511 | 252 | 0.011 | -0.082 | 0.062 | -7.36 |
| DB | def.run_support | 2483 | 425 | 0.439 | 0.154 | 0.043 | 0.35 |
| OL | ol.line_yards | 640 | 0 | 0.367 |  |  |  |
| OL | ol.protection | 640 | 0 | 0.270 |  |  |  |
| OL | ol.short_yardage | 640 | 0 | 0.078 |  |  |  |
| OL | ol.stuff_rate | 640 | 0 | 0.291 |  |  |  |
| K | k.field_goal | 353 | 37 | 0.179 | 0.171 | 0.164 | 0.95 |
| P | p.punt_average | 493 | 35 | 0.479 | 0.388 | 0.173 | 0.81 |

Usage traits travel least. Target share keeps about a third of its
persistence after a move at WR and TE, and the counted defensive rates keep
between a third and two thirds. That is the expected signature of role and
scheme. `qb.explosive` travels worst of the QB traits: its portable share is
0.14 (mover slope 0.056 against a stayer slope of 0.393, 103 movers), so a
transferring QB's explosive percentile says little about his next team. It
carries no headline weight in this fit.

The inferred traits say less than hoped. The with-without impacts have no
year-over-year persistence even among stayers (stay slopes between -0.013
and 0.035), so a share computed from them is noise. The pressure handling
mover slope (0.526, SE 0.118) is well above the stayer slope (0.133), which
suggests the trait follows the passer rather than his line, but it carries
no headline weight. Line independence, TE QB independence and the finishing
rates also show mover slopes above small stayer slopes, mostly within two
mover standard errors, so they cannot be read as travelling more than they
persist. WR QB independence (0.63) behaves like an ordinary player trait. QB
receiver independence has 15 movers and no usable estimate. OL units have no
movers by construction, so the OL traits are uncalibrated. Of the inferred
traits only OL line yards carries headline weight.

## Draft check

Rank correlation of the final-week headline and v4 value with draft slot
(pick negated, so higher is better) and with NFL scrimmage snaps over the
first three seasons. Draft year y reads season y-1, so draft classes 2021
through 2026 enter the pick columns and only the mature classes 2021 through
2023 enter the NFL columns. Every season except 2025 is a development season,
so these correlations are in sample for the fit. Only the 2026 class is out
of sample. They are reported for the draft board and were never used in
fitting. A player needs both a headline and a v4 value, so only qualified FBS
players at groups with a headline enter, which leaves RB and DL (OL units
have no v4 value).

| Group | Players | Pick, v4 | Pick, headline | NFL snaps, v4 | NFL snaps, headline |
| --- | --- | --- | --- | --- | --- |
| RB | 109 | 0.417 | 0.407 | 0.386 | 0.338 |
| DL | 253 | 0.212 | 0.264 | 0.037 | 0.119 |

At RB the two measures are close on draft slot and v4 is higher on NFL snaps
(0.386 against 0.338). At DL the headline is higher on both (0.264 against
0.212 on slot, 0.119 against 0.037 on snaps, where v4 is near zero). These
are in-sample rank correlations, and the NFL columns use only the mature
classes.

## The with-without shrink

`def.impact` and `def.pass_impact` compare a unit's opponent-adjusted EPA
allowed in games a defender appears in the box score with games he does
not, shrunk toward zero with `WITH_WITHOUT_SHRINK_GAMES = 6.0` prior games.
6.0 is the untuned starting value. It was not chosen from data. Split-half
reliability cannot be measured for these traits because they have no
per-game values, and the measurable stand-in, year-over-year persistence
among stayers, is zero at every group (DL -0.013, LB 0.026, DB 0.035 for
`def.impact`; DB 0.011 for `def.pass_impact`). Both traits were dropped from
the headline at every group, so the shrink value changes no headline in v1.
Tuning it is not worth doing until the estimator shows persistence.

## Scoring 2026

`position-traits --seasons 2026` scores with the frozen 2024 fit (log
`output/position-traits-2026-projection.log`). Local raw data for 2026 ends
at week 1, so the artifact holds weeks 0 and 1. Week 1 is in receptions mode
(uncovered share 0.699), and every row records `poor_coverage` True. The
scoring used the frozen expectation tables and read no earlier play-by-play
season.

Only RB, DL and OL carry a headline: 1,628 qualified FBS players in week 1
(391 backs, 1,100 defensive linemen and 137 line units). There is no QB
headline. The top week 1 backs are Jalen Buckley (Western Michigan) 3.66,
Justin Cook (Pittsburgh) 1.16 and Kamari Moulton (Iowa) 0.53, and the top
defensive linemen Derek Wilkins (California) 0.50 and Lamont Lester Jr.
(Colorado) 0.35, each on one game.

Every artifact row carries an `fbs` flag from the team's classification (an
OL unit by its team), and the publisher keeps only FBS rows; the table has
no `fbs` or `poor_coverage` column. The latest 2026 week the publisher reads
has 28,725 FBS rows (27,097 trait rows and 1,628 headline rows) of 46,445.
For a late-season size, the 2025 week 17 snapshot scored the same way has
47,428 FBS rows (45,839 trait rows and 1,589 headline rows). The trait table
keeps only the latest week per season.

## Serving and the weekly step

The frozen model is two artifacts in `backend/data/processed/players/models/`:
the fit, `position_traits_2024.json`, and the expectation tables,
`position_traits_expectations_2024.parquet`. The tables are the down,
distance and field position baselines for rushing yards and conversions and
the distance baseline for field goals, fit on every play season on disk from
2019 through 2025, with the source seasons stored in the parquet metadata.
The weekly runner loads both and refits neither, so it needs no historical
play-by-play and no projection: the projection enters only the fit.

The pair reaches the runner two ways. It travels in the reviewed player
runtime bundle: `export_runtime_bundle` adds the latest fit before the
season and its tables to the Heisman model, WPA baseline and opponent prior,
and `install_runtime_bundle` accepts a bundle with or without the pair,
refusing a fit or tables that do not precede the season. And every weekly
run uploads the pair as the `position-traits-fit-SEASON` artifact, which the
next run pinned to that run id restores into the models directory beside the
`heisman-model-SEASON` artifact. A fit saved before it recorded the
validation season's coverage mode still loads.

The first production run needs a re-exported bundle, because the bundle and
run the workflow points at today predate the pair. Export it with
`poetry run python -m backend player-prior --season 2026 --runtime-bundle <zip>`,
which rebuilds the 2026 opponent prior from local data before exporting (on
2026-10-16 the rebuild matched the frozen prior's numbers and changed only its
model version label from v3 to v4), then review it and point
`PLAYER_RUNTIME_BUNDLE_URL` and its checksum at it. The weekly player step runs
`player-values`, then `heisman`, then `position-traits` with its failure
captured, then `publish-players`, and exits with the trait status. A trait
failure (a missing fit, say) therefore still publishes the Heisman board and
then fails the job. `publish-players` also skips `cfb.position_traits`, with
one log line, while the table is not migrated. The runbook change for this is
a follow-up.

The draft research watchlist reads the trait artifact and adds `headline`,
`headline_percentile`, `headline_games` and `headline_replaces_value` beside
the v4 columns; it never overwrites v4's value, rank or model version. The
production draft publisher still reads `cfb.player_values` only; moving it to
the trait table is a follow-up.

## Findings

These are results from the sanity read that did not match expectations. None
was tuned away.

With the team model's projection in the fit, the traits add 0.014 points of
MAE on 2025 over the projection alone, and v4's EPA credit adds nothing at
any group. Most of what the earlier target credited to players was team
strength. The headline survives only at RB, DL and OL, and it replaces v4
only at DL.

`rb.yards_over_expected` reliability at 8 games is below 0.3 in every season
(0.128, 0.109, 0.107, 0.270, 0.130), yet it keeps headline weight, and the RB
headline is a quarter as reliable as v4 at six games (0.080 against 0.334).
`RB:rb.ball_security` is kept at -0.22, where fewer fumble costs predict a
worse margin on a trait with near-zero reliability. The RB headline should
not be shown as a ranking without that caveat.

Every kept weight is more than three times its leave-one-season-out sd and
clears the drop rule on the refit spreads as well as the first pass.

The penalty (10,000) is interior, and the leave-one-season-out curve is flat
around it; the largest penalty, which shrinks every trait close to zero, is
only 0.008 points worse.

LB has no headline in either the joint fit or the swap test.

K no longer meets the replacement rule: with the projection in the fit, the
K swap fit scores exactly as the projection alone (improvement 0.000) and
the joint fit drops `k.field_goal`.

The first scoring run gave every player a headline row, scored as if he had
all of his team's usage, and ranked it against non-FBS defenders and
specialists too. The headline is now share-scaled and kept for qualified FBS
players only.

The first version of the weekly step refit its expectation tables from the
2019-2025 play-by-play, which the weekly runner does not have; with only the
current season on disk it raised "no season before 2026 to fit expectations
on". The tables are now frozen with the fit, and the 2026 scoring was checked
with every earlier play-by-play read made to fail.

All five portal seasons clear 60 percent of rows naming an FBS destination;
2021 is the lowest at 0.630. The traits variant's validation MAE is finite
in every bucket.

## Limits

There are no tackle attributions on plays, no snap counts, no pressures and
no coverage assignments, so defensive traits rest on box score counts and
with-without comparisons. Offensive linemen are rated as a unit; an
individual lineman carries the unit score flagged inferred. 2019 has no
opponent prior, so it cannot be opponent adjusted and development starts in
2020; 2019 still feeds the 2020 expectation tables. Targets are missing for
most incompletions in 2023, 2024 and 2026 week 1, which ran in receptions
mode. Win probability is missing for games with an FCS side, so the traits
weighted by win probability get no leverage signal from those plays.

The projection is chronological in its training data but not in its
configuration: the team model's hyperparameters were chosen on 2019 to 2022
with 2023 to 2025 as its holdout, so the 2020 to 2022 projections are in
sample for that configuration choice. The effect on the trait fit is
probably mild, since the configuration is a handful of smoothing and prior
settings rather than game-level fits; it is stated here, not corrected. 2026
was never fitted and is judged only as it is served.
