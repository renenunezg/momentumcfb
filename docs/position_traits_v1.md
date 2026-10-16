# Position traits v1: per-position trait ratings

## What the model is

Every FBS player gets a set of named traits at his position and one headline
number. A trait is a per-game rate built from tagged plays or box score
counts, adjusted for the opponent where the data allows, and turned into a
position percentile among qualified FBS players that week. Traits are
`counted` when the feed tags the player directly and `inferred` when they
come from unit outcomes or teammate comparisons. The headline is the
player's own points per game in the fitted model: his share of his team's
usage at the group times the fitted combination of his traits and his player
value v4 EPA credit, measured above the 30th percentile of qualified FBS
players at his group. Only qualified FBS players carry a headline (see The
headline number).

The headline weights come from one joint fit across positions. Each
completed FBS against FBS game from model week 2 on is one row. The target is
the home margin. The predictors are home field and, for every group and
trait, the home team's aggregate minus the away team's, where a team's
aggregate is the usage-share-weighted sum of its players' standardized trait
values in the snapshot from the week before the game. The v4 EPA credit
enters the same fit as one more trait per group, so a trait earns weight
only for what it adds beyond EPA credit. Nothing from the week being rated
enters its own rating.

Development seasons are 2020 through 2024. 2025 is scored once with the
frozen weights as the independent check. 2026 was never fitted. The fit is
`backend/data/processed/players/models/position_traits_2024.json`, the full
evaluation is
`backend/data/processed/players/position_traits/evaluation_2024.parquet`, and
every table below is copied from those two files and from the run log of
`position-traits-fit --dev-seasons 2020 2021 2022 2023 2024 --validate 2025`
on 2026-10-16, re-run after the penalty grid was extended, the replacement
and serving fixes went in, and the headline became share-scaled over
qualified FBS players (log `output/position-traits-fit-20261016-final.log`).
Player values for 2020 through 2025 were rebuilt under
`cfb_player_value_v4` before the run, so the baseline is one model.

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

2023 and 2024 ran in receptions mode. A served season does not choose its
own mode: the fit stores the validation season's mode (`poor_coverage`,
False for 2025) and the weekly runner builds in that mode. The 2026 week 1
data would be in receptions mode on its own (uncovered share 0.699) and is
built in targets mode.

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
intercept and home field unpenalized. The penalty is chosen from 36
log-spaced values between 0.01 and 100,000 by pooled leave-one-season-out MAE
over the five development seasons. A feature is then dropped when its weight
is under twice the larger of its delete-one-season jackknife sd and its
analytic ridge standard error, and the rest are refit once at the same
penalty.

The chosen penalty is 1000. A first run searched only up to 1000 and landed
on that edge, so the grid was extended two decades. The minimum stayed at
1000, now inside the grid. Recomputed with the same functions on the same
development rows, the pooled leave-one-season-out MAE is 14.380 at 631,
14.371 at 1000, 14.371 at 1585 and 14.384 at 2512, so the curve is flat
between 1000 and 1585 and rises on either side. The penalty is large: the
weights are heavily shrunk.

## Weights

Weights are points of home margin per development-season standard deviation
of the feature. `loso_sd` is the sd of the weight across the five
leave-one-season-out refits. Two decimals is as much precision as these
spreads support. The spreads shown are from the final refit on the kept
features; the drop rule was applied to the first pass over every feature, so
a kept weight can sit under twice its refit spread.

| Feature | Weight | LOSO sd | Jackknife sd | Analytic SE |
| --- | --- | --- | --- | --- |
| intercept | -1.21 | 0.27 | 0.49 | 1.09 |
| home_field | 4.83 | 0.45 | 0.81 | 1.14 |
| QB:qb.explosive | 1.01 | 0.03 | 0.05 | 0.25 |
| QB:qb.floor | 1.00 | 0.24 | 0.43 | 0.25 |
| QB:epa | 2.56 | 0.17 | 0.31 | 0.25 |
| RB:rb.yards_over_expected | 1.11 | 0.12 | 0.21 | 0.25 |
| RB:rb.breakaway | 0.48 | 0.15 | 0.27 | 0.25 |
| RB:rb.ball_security | -0.96 | 0.22 | 0.39 | 0.24 |
| RB:epa | 1.50 | 0.16 | 0.29 | 0.26 |
| TE:rec.target_share | 0.76 | 0.19 | 0.34 | 0.24 |
| TE:rec.qb_independence | 0.62 | 0.16 | 0.28 | 0.24 |
| DL:def.havoc | 1.44 | 0.15 | 0.27 | 0.24 |
| DL:def.run_stop | -1.01 | 0.11 | 0.19 | 0.25 |
| DL:def.every_week | 1.48 | 0.31 | 0.55 | 0.25 |
| DB:def.havoc | 1.24 | 0.17 | 0.31 | 0.24 |
| DB:def.every_week | 0.60 | 0.10 | 0.18 | 0.24 |
| DB:def.run_support | -1.81 | 0.14 | 0.25 | 0.24 |
| OL:ol.protection | 1.22 | 0.06 | 0.10 | 0.24 |
| OL:ol.line_yards | 1.64 | 0.14 | 0.25 | 0.26 |
| K:k.field_goal | 0.68 | 0.15 | 0.27 | 0.24 |

Dropped from the headline and kept on the card as descriptive:

| Group | Dropped |
| --- | --- |
| QB | qb.drive_engine, qb.ball_security, qb.legs, qb.pressure_handling, qb.receiver_independence |
| RB | rb.stuff_avoidance, rb.short_yardage, rb.receiving, rb.line_independence |
| WR | rec.target_share, rec.value_per_target, rec.explosive, rec.late_down, rec.qb_independence, epa |
| TE | rec.value_per_target, rec.explosive, rec.late_down, epa |
| DL | def.pass_rush, def.finishing, def.impact, epa |
| LB | def.havoc, def.pass_rush, def.finishing, def.run_stop, def.every_week, def.tackle_share, def.coverage, def.impact, epa |
| DB | def.ball_production, def.impact, def.pass_impact, epa |
| OL | epa (constant), ol.stuff_rate, ol.short_yardage |
| K | epa |
| P | p.punt_average, epa |

Unconventional traits earned weight at QB (explosive, floor), RB (yards over
expected, breakaway), TE (target share, QB independence), DL and DB. The
headline is not EPA per play with a new name: v4 credit keeps weight only at
QB and RB. WR, LB and P keep no feature at all, so they carry no headline,
none of their traits is counted in the headline, and v4 remains their rating.

Three signs are negative. `DB:def.run_support` has no preset sign, and the
fit says a safety or corner who makes many run-support tackles is on a
worse defense, which is what the design expected. `DL:def.run_stop` is
negative alongside a positive `DL:def.havoc`, which already counts the same
tackles for loss; the pair reads as havoc credited once and run-game TFL
discounted. `RB:rb.ball_security` is negative although higher means fewer
fumble costs, and its split-half reliability is near zero. That weight is
listed under Findings.

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
players keep their trait rows. A group whose every feature was dropped (WR,
LB and P in this fit) gets no headline rows, `replaces_value` False and no
trait counted `in_headline`. The share scaling is what keeps a player with
one dropback from topping the QB list on a single deep completion.

## 2025 validation

Next-game margin MAE of the frozen development fit on 2025, against a fit of
the v4 EPA credit columns alone on the same development seasons.

| Week bucket | Games | traits | v4 | v4 minus traits |
| --- | --- | --- | --- | --- |
| 2-5 | 198 | 15.472 | 15.879 | 0.407 |
| 6-9 | 218 | 13.807 | 14.458 | 0.652 |
| 10+ | 344 | 14.129 | 14.939 | 0.811 |
| all | 760 | 14.386 | 15.046 | 0.660 |

The trait headline beats v4 in every bucket, by 0.66 points over the season.
Part of the gain is the offensive line, which v4 cannot represent at all.

## Which positions replace v4

The swap test refits on the development seasons with one group's full
feature set and only the EPA column of every other group, then scores 2025.
Rene's rule: the headline replaces v4 at a group when the improvement over
the v4-only MAE is above zero and the headline's six-game split-half
reliability is at least v4's, both measured on the same players and games.

| Group | MAE v4 only | MAE with traits | Improvement | Reliability 6, v4 | Reliability 6, headline | Replaces v4 |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 15.046 | 14.929 | 0.117 | 0.463 | 0.424 | False |
| RB | 15.046 | 14.986 | 0.060 | 0.334 | 0.167 | False |
| WR | 15.046 | 15.078 | -0.031 | 0.375 |  | False |
| TE | 15.046 | 15.081 | -0.035 | 0.311 | 0.417 | False |
| DL | 15.046 | 14.788 | 0.258 | 0.149 | 0.177 | True |
| LB | 15.046 | 14.875 | 0.171 | 0.199 |  | False |
| DB | 15.046 | 14.869 | 0.177 | 0.092 | 0.144 | True |
| OL | 15.046 | 14.774 | 0.273 |  |  | False |
| K | 15.046 | 14.975 | 0.072 | -0.004 | 0.244 | False |
| P | 15.046 | 15.055 | -0.009 | -0.084 |  | False |

The headline replaces v4 at DL and DB. It ships beside v4 everywhere
else. QB and RB improve the margin but are less reliable than v4 at six
games. WR and TE do not improve the margin. LB improves it in the swap test,
but the joint fit keeps no LB feature, so LB has no headline and no headline
reliability. OL improves the most, but v4 has no OL rating to compare
against, so it cannot replace by construction. P neither improves nor has a
headline.

OL, K and P never replace v4 in v1; the code holds them back whatever their
numbers, and the table still shows those numbers. OL and P would not have
replaced anyway. K met the rule: v4's six-game reliability for kickers is
-0.004 on 14 players, the headline's is 0.244, and the swap improvement is
0.072 points. On a sample of 14 kickers that is not evidence, so K is held
back pending Rene's decision.

## Reliability

Split-half reliability correlates a player's odd-game value with his
even-game value at a fixed game count. The headline and v4 rows are pooled
over the development seasons and paired: same athletes, same games. The
headline side correlates `raw_score`, the share-1 value, and follows the
headline's own rule, so non-FBS athletes drop out of both sides; a group with
no headline keeps its v4 side.

| Group | v4 4 | Headline 4 | v4 6 | Headline 6 | v4 8 | Headline 8 | v4 12 | Headline 12 | Players at 6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QB | 0.359 | 0.333 | 0.463 | 0.424 | 0.465 | 0.400 | 0.477 | 0.448 | 763 |
| RB | 0.271 | 0.127 | 0.334 | 0.167 | 0.366 | 0.192 | 0.451 | 0.272 | 1792 |
| WR | 0.305 |  | 0.375 |  | 0.428 |  | 0.502 |  | 2725 |
| TE | 0.209 | 0.336 | 0.311 | 0.417 | 0.400 | 0.449 | 0.475 | 0.470 | 850 |
| DL | 0.021 | 0.023 | 0.149 | 0.177 | 0.217 | 0.277 | 0.266 | 0.355 | 4033 |
| LB | 0.106 |  | 0.199 |  | 0.253 |  | 0.295 |  | 3005 |
| DB | 0.018 | 0.048 | 0.092 | 0.144 | 0.127 | 0.219 | 0.114 | 0.348 | 4596 |
| OL |  |  |  |  |  |  |  |  |  |
| K |  |  | -0.004 | 0.244 | 0.013 | -0.044 |  |  | 14 |
| P | -0.180 |  | -0.084 |  | -0.150 |  | 0.090 |  | 61 |

Per-trait reliability is computed separately for each development season.
The table shows the mean over 2020 through 2024 weighted by players; the
per-season rows are in the evaluation file.

| Group | Trait | 4 | 6 | 8 | 12 | Players at 8 | In headline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QB | qb.ball_security | -0.003 | 0.026 | 0.014 | -0.005 | 571 |  |
| QB | qb.drive_engine | 0.143 | 0.148 | 0.187 | 0.188 | 572 |  |
| QB | qb.explosive | 0.149 | 0.239 | 0.288 | 0.417 | 571 | yes |
| QB | qb.floor |  |  | 0.245 | 0.193 | 495 | yes |
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
| TE | rec.qb_independence | 0.019 | 0.128 | 0.095 | 0.167 | 524 | yes |
| TE | rec.target_share | 0.529 | 0.620 | 0.711 | 0.746 | 593 | yes |
| TE | rec.value_per_target | 0.026 | 0.106 | 0.152 | 0.117 | 566 |  |
| DL | def.every_week | 0.171 | 0.218 | 0.267 | 0.344 | 3501 | yes |
| DL | def.finishing | 0.099 | 0.068 | 0.063 |  | 319 |  |
| DL | def.havoc | 0.236 | 0.310 | 0.362 | 0.421 | 3501 | yes |
| DL | def.pass_rush | 0.031 | 0.067 | 0.119 | 0.156 | 3497 |  |
| DL | def.run_stop | 0.094 | 0.150 | 0.170 | 0.263 | 3501 | yes |
| LB | def.coverage | 0.118 | 0.165 | 0.184 | 0.190 | 2718 |  |
| LB | def.every_week | 0.274 | 0.329 | 0.353 | 0.359 | 2718 |  |
| LB | def.finishing | 0.040 | 0.088 | 0.136 |  | 134 |  |
| LB | def.havoc | 0.314 | 0.388 | 0.421 | 0.462 | 2718 |  |
| LB | def.pass_rush | 0.082 | 0.120 | 0.139 | 0.182 | 2713 |  |
| LB | def.run_stop | 0.205 | 0.254 | 0.283 | 0.333 | 2718 |  |
| LB | def.tackle_share | 0.666 | 0.736 | 0.769 | 0.792 | 2710 |  |
| DB | def.ball_production | 0.203 | 0.229 | 0.260 | 0.291 | 4063 |  |
| DB | def.every_week | 0.223 | 0.241 | 0.257 | 0.219 | 4063 | yes |
| DB | def.havoc | 0.230 | 0.267 | 0.288 | 0.280 | 4063 | yes |
| DB | def.run_support | 0.348 | 0.432 | 0.476 | 0.532 | 4063 | yes |
| OL | ol.line_yards | 0.246 | 0.330 | 0.350 | 0.384 | 612 | yes |
| OL | ol.protection | 0.206 | 0.255 | 0.315 | 0.429 | 609 | yes |
| OL | ol.short_yardage | 0.039 | -0.009 | 0.028 | 0.157 | 602 |  |
| OL | ol.stuff_rate | 0.143 | 0.193 | 0.221 | 0.297 | 612 |  |
| K | k.field_goal | 0.015 | 0.047 | 0.041 | 0.062 | 566 | yes |
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
between a third and two thirds. That is the expected signature of role and scheme.

The inferred traits say less than hoped. The with-without impacts have no
year-over-year persistence even among stayers (stay slopes between -0.013
and 0.035), so a share computed from them is noise. The pressure handling
mover slope (0.526, SE 0.118) is well above the stayer slope (0.133), which
suggests the trait follows the passer rather than his line, but it carries
no headline weight. Line independence, TE QB independence and the finishing
rates also show mover slopes above small stayer slopes, mostly within two
mover standard errors, so they cannot be read as travelling more than they
persist. WR QB independence (0.63) behaves like an ordinary player trait. QB receiver independence
has 15 movers and no usable estimate. OL units have no movers by
construction, so the OL traits are uncalibrated. Of the inferred traits only
TE QB independence and the two OL unit traits carry headline weight.

## Draft check

Rank correlation of the final-week headline and v4 value with draft slot
(pick negated, so higher is better) and with NFL scrimmage snaps over the
first three seasons. Draft year y reads season y-1, so draft classes 2021
through 2026 enter the pick columns and only the mature classes 2021 through
2023 enter the NFL columns. Every season except 2025 is a development season,
so these correlations are in sample for the fit. Only the 2026 class is out
of sample. They are reported for the draft board and were never used in
fitting. A player needs both a headline and a v4 value, so only qualified FBS
players at groups with a headline enter; WR, LB and P have no rows.

| Group | Players | Pick, v4 | Pick, headline | NFL snaps, v4 | NFL snaps, headline |
| --- | --- | --- | --- | --- | --- |
| QB | 61 | 0.418 | 0.471 | 0.032 | 0.141 |
| RB | 109 | 0.417 | 0.416 | 0.386 | 0.417 |
| TE | 82 | 0.388 | 0.293 | 0.148 | 0.211 |
| DL | 253 | 0.212 | 0.232 | 0.037 | 0.103 |
| DB | 285 | 0.250 | 0.253 | 0.210 | 0.224 |
| K | 1 |  |  |  |  |

On draft slot the two measures are close except at QB, where the headline
is higher (0.471 against 0.418), and TE, where it is lower (0.293 against
0.388). On NFL snaps they are not close at QB (0.141 against 0.032) or DL
(0.103 against 0.037), where v4 is near zero; the headline is a little
higher at RB, TE and DB. These are in-sample rank correlations on 61 QBs and
253 linemen, and the NFL columns use only the mature classes.

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
`output/position-traits-2026-final.log`). Local raw data for 2026 ends at
week 1, so the artifact holds weeks 0 and 1, with 3,440 qualified FBS
players scored in week 1. The top five QB headlines in week 1 are Josh Hoover
(Indiana) 8.10, Lincoln Kienholz (Louisville) 7.34, Rocco Becht (Penn State)
6.97, Darian Mensah (Miami) 6.70 and Beau Pribula (Virginia) 6.29, each with
share 1.0 on one game. Chaston Ditta (North Texas), first under the old
share-1 headline on one dropback, now scores 0.29 at share 1/28. The scoring
used the frozen expectation tables, read no earlier play-by-play season, and
built in the fit's targets mode. WR, LB and P have no headline rows.

Every artifact row carries an `fbs` flag from the team's classification (an
OL unit by its team), and the publisher keeps only FBS rows; the table has
no `fbs` column. The latest 2026 week the publisher reads has 31,091 FBS
rows (27,651 trait rows and 3,440 headline rows) of 48,811. For a
late-season size, the 2025 week 17 snapshot scored the same way has 49,140
FBS rows (45,839 trait rows and 3,301 headline rows). The trait table keeps
only the latest week per season.

## Serving and the weekly step

The frozen model is two artifacts in `backend/data/processed/players/models/`:
the fit, `position_traits_2024.json`, and the expectation tables,
`position_traits_expectations_2024.parquet`. The tables are the down,
distance and field position baselines for rushing yards and conversions and
the distance baseline for field goals, fit on every play season on disk from
2019 through 2025, with the source seasons stored in the parquet metadata.
The weekly runner loads both and refits neither, so it needs no historical
play-by-play.

The pair reaches the runner two ways. It travels in the reviewed player
runtime bundle: `export_runtime_bundle` adds the latest fit before the
season and its tables to the Heisman model, WPA baseline and opponent prior,
and `install_runtime_bundle` accepts a bundle with or without the pair,
refusing a fit or tables that do not precede the season. And every weekly
run uploads the pair as the `position-traits-fit-SEASON` artifact, which the
next run pinned to that run id restores into the models directory beside the
`heisman-model-SEASON` artifact.

The fit also freezes the receptions-or-targets mode (see Data coverage), so
a served season is built the way the validation season was.

The first production run needs a re-exported bundle, because the bundle and
run the workflow points at today predate the pair, and a fit saved before the
mode was stored is refused. Export it with
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

The first run's penalty sat at 1000, the top of the original grid, with the
error still falling. The grid now runs to 100,000 and the minimum is still
1000, so that was a search range defect and not a hidden larger penalty.

`qb.explosive` carries the largest QB trait weight (1.01) but travels worst
of the QB traits: its portable share is 0.14 (mover slope 0.056 against a
stayer slope of 0.393, 103 movers). Most of what it measures stays with the
team, the receivers or the scheme, so a transferring QB's explosive
percentile says little about his next team.

`qb.explosive` reliability at 8 games is above 0.3 in 2020 (0.317), 2022
(0.334) and 2023 (0.320) but not in 2021 (0.187) or 2024 (0.286).
`rb.yards_over_expected` is below 0.3 in every season (0.128, 0.109, 0.107,
0.270, 0.130). Both still earn headline weight, but the RB headline is only
half as reliable as v4 at six games (0.167 against 0.334).

Every kept weight is more than three times its leave-one-season-out sd. The
weights are stable. One kept weight would fail the drop rule on the refit
spreads: `RB:rb.breakaway` is 0.48 against a refit jackknife sd of 0.27
(twice that is 0.54); it passed on the first-pass spreads, where the rule is
applied. The ones that are hard to explain are
`RB:rb.ball_security` at -0.96, where fewer fumble costs predict a worse
margin on a trait with near-zero reliability, and `DL:def.run_stop` at
-1.01, which reads as an offset to havoc.

K met the replacement rule, against the design's expectation. The decision
rests on 14 kickers with six paired games; at eight games the headline
reliability is -0.044. K is held back with OL and P until Rene decides.

LB improves the margin in the swap test (0.171) but has no headline,
because the swap test and the joint fit select features separately and the
joint fit drops every LB feature.

The first scoring run gave every player a headline row, scored as if he had
all of his team's usage, and ranked it against non-FBS defenders and
specialists too. Its week 1 QB list was led by a passer with one dropback.
The headline is now share-scaled and kept for qualified FBS players only.
The 2026 week 1 publish is 31,091 FBS rows against 32,113 before; the trait
rows changed too, because 2026 is now built in the fit's targets mode.

The first version of the weekly step refit its expectation tables from the
2019-2025 play-by-play, which the weekly runner does not have; with only the
current season on disk it raised "no season before 2026 to fit expectations
on". The tables are now frozen with the fit, and the 2026 scoring above was
checked with every earlier play-by-play read made to fail.

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
most incompletions in 2023 and 2024, which ran in receptions mode, and 2026
is built in the 2025 targets mode although 70 percent of its week 1
incompletions have no target, so its targets are undercounted. Win
probability is missing for games with an FCS side, so the traits weighted by
win probability get no leverage signal from those plays. The fit target is the home margin with home field as a
regressor and no pregame team rating, so roster aggregates carry some team
quality. 2026 was never fitted and is judged only as it is served.
