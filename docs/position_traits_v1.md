# Position traits v1: per-position trait ratings and the grade

## What the model is

Every FBS player gets a set of named traits at his position and two numbers.

- The **grade** is a 20 to 80 scouting number built from his traits. It has
  mean 50 and sd 10 among qualified FBS players at his position that week.
  It says how good his profile is, not how many points he added.
- The **impact** is player value v4 (`cfb_player_value_v4`, published in
  `cfb.player_values`): points per game above positional replacement from
  his credited plays, opponent adjusted, with win probability added beside
  it. It is not refit here.

A trait is a per-game rate built from tagged plays or box score counts,
adjusted for the opponent where the data allows, and turned into a position
percentile among qualified FBS players that week. Traits are `counted` when
the feed tags the player directly and `inferred` when they come from unit
outcomes or teammate comparisons. Nothing from the week being rated enters
its own rating.

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
on 2026-10-16 (`output/position-grade-fit-202610162142.log`). Player values
for 2020 through 2025 are all `cfb_player_value_v4`, so the EPA baseline is
one model.

## How the weights were set

A trait's weight at a position comes from two pieces of player-level
evidence on the development seasons, and nothing else.

- **Reliability** is the split-half correlation at 8 games: a player's value
  over his first four odd games against his value over his first four even
  games, with the five development seasons pooled (each season's athletes
  kept apart). It asks whether the trait is stable within a player.
- **Portability** is the transfer test. For each consecutive pair of
  development seasons, last season's final-week value is regressed on this
  season's, once for players who stayed and once for matched transfers. The
  portable share is the mover slope over the stayer slope, clipped to
  [0, 1]. It asks whether the trait follows the player or stays with the
  team. It is unmeasured when fewer than 20 players moved or the stayer
  slope is not positive.

`weight = max(reliability, 0) x portability`, with an unmeasured
portability counted as 1, then normalized to sum to 1 within the position.
Each trait gets a status:

| Status | Rule | Weight |
| --- | --- | --- |
| weighted | reliability at least 0.10, portability measured and positive | reliability x portability, normalized |
| unmeasured | reliability at least 0.10, portability not measurable | reliability, normalized |
| noise | reliability under 0.10, or no per-game values to measure it | 0 |
| team-driven | portability measured and at or below zero | 0 |

Noise and team-driven traits stay on the card, dimmed, as descriptive.

Orientation: a trait registered higher-is-better enters as is, one
registered lower-is-better is negated, and one with no registered direction
takes the sign of its Spearman correlation with v4 `shrunk_per_game` among
qualified FBS players at the position, at the final week of each development
season. Only `def.run_support` (DB tackles in the run game) has no
registered direction. Its correlation with v4 is 0.200 over 4,795 player
seasons, so it enters with a positive sign. The sign is stored in the model.

## Weights

Reliability is the pooled 8-game split-half correlation; portability the
clipped portable share. Weights sum to 1 within each position.

| Group | Trait | Reliability | Portability | Weight | Status |
| --- | --- | --- | --- | --- | --- |
| QB | qb.drive_engine | 0.191 | 1.000 | 0.117 | weighted |
| QB | qb.explosive | 0.288 | 0.360 | 0.063 | weighted |
| QB | qb.ball_security | 0.115 | 1.000 | 0.070 | weighted |
| QB | qb.floor | 0.269 | 0.672 | 0.110 | weighted |
| QB | qb.legs | 0.160 | 1.000 | 0.097 | weighted |
| QB | qb.pressure_handling | 0.351 | 1.000 | 0.214 | weighted |
| QB | qb.receiver_independence | 0.537 |  | 0.328 | unmeasured |
| RB | rb.yards_over_expected | 0.150 | 1.000 | 0.238 | weighted |
| RB | rb.stuff_avoidance | 0.108 | 1.000 | 0.171 | weighted |
| RB | rb.breakaway | 0.172 | 0.434 | 0.119 | weighted |
| RB | rb.short_yardage | -0.044 | 0.960 | 0 | noise |
| RB | rb.receiving | 0.190 | 1.000 | 0.302 | weighted |
| RB | rb.ball_security | -0.051 |  | 0 | noise |
| RB | rb.line_independence | 0.107 | 1.000 | 0.170 | weighted |
| WR | rec.target_share | 0.715 | 0.354 | 0.231 | weighted |
| WR | rec.value_per_target | 0.339 | 0.755 | 0.234 | weighted |
| WR | rec.explosive | 0.225 | 0.749 | 0.154 | weighted |
| WR | rec.late_down | 0.299 | 0.966 | 0.264 | weighted |
| WR | rec.qb_independence | 0.145 | 0.878 | 0.116 | weighted |
| TE | rec.target_share | 0.714 | 0.235 | 0.204 | weighted |
| TE | rec.value_per_target | 0.228 | 1.000 | 0.276 | weighted |
| TE | rec.explosive | 0.131 | 1.000 | 0.158 | weighted |
| TE | rec.late_down | 0.207 | 0.876 | 0.220 | weighted |
| TE | rec.qb_independence | 0.117 | 1.000 | 0.142 | weighted |
| DL | def.havoc | 0.364 | 0.618 | 0.399 | weighted |
| DL | def.pass_rush | 0.130 | 1.000 | 0.230 | weighted |
| DL | def.finishing | 0.073 | 1.000 | 0 | noise |
| DL | def.run_stop | 0.179 | 0.346 | 0.110 | weighted |
| DL | def.every_week | 0.270 | 0.545 | 0.261 | weighted |
| DL | def.impact |  |  | 0 | noise |
| LB | def.havoc | 0.424 | 0.596 | 0.171 | weighted |
| LB | def.pass_rush | 0.138 | 0.808 | 0.075 | weighted |
| LB | def.finishing | 0.141 | 1.000 | 0.095 | weighted |
| LB | def.run_stop | 0.290 | 0.522 | 0.102 | weighted |
| LB | def.every_week | 0.360 | 0.635 | 0.154 | weighted |
| LB | def.tackle_share | 0.771 | 0.643 | 0.335 | weighted |
| LB | def.coverage | 0.188 | 0.529 | 0.067 | weighted |
| LB | def.impact |  | 1.000 | 0 | noise |
| DB | def.havoc | 0.290 | 0.495 | 0.239 | weighted |
| DB | def.every_week | 0.259 | 0.593 | 0.256 | weighted |
| DB | def.ball_production | 0.262 | 0.504 | 0.220 | weighted |
| DB | def.run_support | 0.485 | 0.353 | 0.285 | weighted |
| DB | def.impact |  | 0.000 | 0 | noise |
| DB | def.pass_impact |  | 0.000 | 0 | noise |
| OL | ol.protection | 0.311 |  | 0.337 | unmeasured |
| OL | ol.stuff_rate | 0.253 |  | 0.274 | unmeasured |
| OL | ol.short_yardage | 0.024 |  | 0 | noise |
| OL | ol.line_yards | 0.360 |  | 0.389 | unmeasured |
| K | k.field_goal | 0.036 | 0.800 | 0 | noise |
| P | p.punt_average | 0.497 | 0.580 | 1.000 | weighted |

Reliability sample sizes at 8 games are in the evaluation file's
`reliability` table (season `pooled`). Most are in the hundreds or
thousands. Three are small: `qb.receiver_independence` 36 players,
`rec.late_down` at TE 84, and `def.finishing` 319 at DL and 134 at LB.

What the table says:

- Usage traits are the most reliable and among the least portable. Target
  share is 0.715 reliable at WR and keeps 0.354 of its persistence after a
  move; at TE it keeps 0.235. Tackle share at LB is 0.771 and 0.643. The
  product rule keeps them in the grade but cuts them back.
- `qb.explosive` is reliable (0.288) but travels poorly (0.360), so it has
  the smallest QB weight (0.063).
- `qb.receiver_independence` carries the largest QB weight (0.328) on a
  reliability measured over 36 players and no portability (12 movers). It
  exists for few QBs, and most QBs are graded without it.
- The with-without impacts (`def.impact`, `def.pass_impact`) are noise:
  they have no per-game values, so their reliability cannot be measured,
  and at DB their portability is at or below zero as well.
- No trait was marked team-driven by portability alone: every trait with a
  non-positive portable share was already noise by reliability.
- K has no grade: `k.field_goal` is not reliable (0.036). P is graded on
  punt average alone.
- OL units never move, so the three OL traits that pass the reliability
  cut are unmeasured for portability and carry their reliability weight.

## The grade

For one week, among qualified FBS players at the position:

- each weighted trait is z-scored on the pool's mean and sd and multiplied
  by its sign (a trait with no spread gives everyone 0);
- `composite = sum(w_t z_t) / sum(w_t)` over the weighted traits the player
  has a value for;
- `grade = 50 + 10 x (composite - mean) / sd` over the pool, clipped to
  [20, 80];
- `grade_percentile` ranks the composite within the pool, 0 to 100;
- `uncertainty = 10 x sqrt(1 - r)`, with `r` the composite's split-half
  reliability at the player's game count, read off the curve below (linear
  from r = 0 at zero games, flat past 12 games).

Only qualified FBS players get a grade row; unqualified and non-FBS players
keep their trait rows.

## Composite reliability

The composite's own split-half correlation, pooled over the development
seasons. Each player's games alternate between two halves in week order,
each half is scored as its own snapshot week with the frozen weights, and
the two composites are correlated. This curve sets the uncertainty band.

| Group | 4 games | 6 games | 8 games | 12 games | Players at 6 | Players at 12 |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 0.101 | 0.237 | 0.333 | 0.408 | 763 | 260 |
| RB | 0.063 | 0.084 | 0.119 | 0.089 | 1793 | 592 |
| WR | 0.207 | 0.262 | 0.258 | 0.272 | 2729 | 717 |
| TE | 0.167 | 0.214 | 0.271 | 0.152 | 850 | 115 |
| DL | 0.218 | 0.290 | 0.351 | 0.417 | 4454 | 1018 |
| LB | 0.525 | 0.582 | 0.606 | 0.614 | 3404 | 1157 |
| DB | 0.338 | 0.362 | 0.381 | 0.341 | 5084 | 1468 |
| OL | 0.200 | 0.315 | 0.349 | 0.413 | 640 | 496 |
| P | 0.258 | 0.336 | 0.440 | 0.502 | 607 | 386 |

The RB composite is barely reliable: 0.119 at 8 games, lower than three of
its five weighted traits alone. The band for a back stays near the full
10 points all season.

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

Shares above 1 are clipped to 1 in the weights. Most of them come from
small stayer slopes with wide mover standard errors (TE explosive 5.80,
LB finishing 6.47), so they say the trait is not worse after a move, not
that it improves.

## Next-season validation

Does a player's final-week grade predict next season, compared with his EPA
credit alone? For each consecutive pair of seasons, over players graded at
the final week of both seasons at the same position, three Spearman
correlations:

- grade to next grade: does the grade persist;
- grade to next EPA: does the grade predict next season's v4
  `shrunk_per_game`;
- EPA to next EPA: the EPA-only baseline, v4 `shrunk_per_game` to next
  season's.

The fair test of "predicts production" is the second column against the
third: both predict the same outcome. The first column is the grade's own
persistence and is not comparable with the third, which predicts a
different number.

The 2024 to 2025 row is the holdout: 2025 entered no weight and no
reliability curve. The other rows are in sample, because the weights read
those seasons. Where at least 10 players have all four numbers every
correlation uses only them; OL has no v4 value and few punters have one,
so those rows keep every graded player and their EPA columns are mostly
empty.

| Group | Seasons | Players | Grade to next grade | Grade to next EPA | EPA to next EPA | Holdout |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 2020-2021 | 62 | 0.255 | 0.292 | 0.374 |  |
| QB | 2021-2022 | 89 | 0.165 | 0.305 | 0.518 |  |
| QB | 2022-2023 | 83 | 0.327 | 0.239 | 0.408 |  |
| QB | 2023-2024 | 82 | 0.406 | 0.387 | 0.390 |  |
| QB | 2024-2025 | 82 | 0.219 | 0.319 | 0.546 | yes |
| RB | 2020-2021 | 167 | 0.291 | 0.343 | 0.328 |  |
| RB | 2021-2022 | 197 | 0.211 | 0.224 | 0.340 |  |
| RB | 2022-2023 | 225 | 0.283 | 0.280 | 0.373 |  |
| RB | 2023-2024 | 199 | 0.255 | 0.257 | 0.417 |  |
| RB | 2024-2025 | 178 | 0.181 | 0.242 | 0.256 | yes |
| WR | 2020-2021 | 250 | 0.118 | 0.186 | 0.382 |  |
| WR | 2021-2022 | 307 | 0.202 | 0.281 | 0.377 |  |
| WR | 2022-2023 | 305 | 0.156 | 0.163 | 0.322 |  |
| WR | 2023-2024 | 292 | 0.309 | 0.262 | 0.372 |  |
| WR | 2024-2025 | 271 | 0.211 | 0.224 | 0.399 | yes |
| TE | 2020-2021 | 67 | 0.153 | 0.262 | 0.329 |  |
| TE | 2021-2022 | 85 | 0.107 | 0.135 | 0.337 |  |
| TE | 2022-2023 | 80 | 0.307 | 0.341 | 0.469 |  |
| TE | 2023-2024 | 90 | 0.245 | 0.318 | 0.287 |  |
| TE | 2024-2025 | 83 | 0.273 | 0.336 | 0.274 | yes |
| DL | 2020-2021 | 344 | 0.433 | 0.299 | 0.142 |  |
| DL | 2021-2022 | 492 | 0.449 | 0.250 | 0.122 |  |
| DL | 2022-2023 | 562 | 0.384 | 0.196 | 0.129 |  |
| DL | 2023-2024 | 540 | 0.430 | 0.251 | 0.153 |  |
| DL | 2024-2025 | 453 | 0.384 | 0.125 | 0.106 | yes |
| LB | 2020-2021 | 306 | 0.491 | 0.174 | 0.169 |  |
| LB | 2021-2022 | 360 | 0.477 | 0.147 | 0.185 |  |
| LB | 2022-2023 | 358 | 0.427 | 0.262 | 0.109 |  |
| LB | 2023-2024 | 374 | 0.361 | 0.167 | 0.060 |  |
| LB | 2024-2025 | 341 | 0.409 | 0.197 | 0.103 | yes |
| DB | 2020-2021 | 450 | 0.286 | 0.098 | 0.032 |  |
| DB | 2021-2022 | 554 | 0.316 | 0.109 | 0.106 |  |
| DB | 2022-2023 | 566 | 0.349 | 0.124 | 0.095 |  |
| DB | 2023-2024 | 564 | 0.312 | 0.139 | 0.082 |  |
| DB | 2024-2025 | 517 | 0.272 | 0.137 | 0.085 | yes |
| OL | 2020-2021 | 112 | 0.370 |  |  |  |
| OL | 2021-2022 | 130 | 0.377 |  |  |  |
| OL | 2022-2023 | 131 | 0.373 |  |  |  |
| OL | 2023-2024 | 133 | 0.514 |  |  |  |
| OL | 2024-2025 | 134 | 0.328 |  |  | yes |
| P | 2020-2021 | 63 | 0.284 | -0.191 |  |  |
| P | 2021-2022 | 76 | 0.425 | -0.043 |  |  |
| P | 2022-2023 | 80 | 0.358 | 0.067 |  |  |
| P | 2023-2024 | 83 | 0.361 | -0.191 |  |  |
| P | 2024-2025 | 70 | 0.557 | -0.054 |  | yes |

The P grade-to-EPA column covers only the few punters with a v4 value, and
v4 is not built to rate punters; it is shown, not read.

Holdout, grade to next EPA against EPA to next EPA:

| Group | Grade to next EPA | EPA to next EPA | Better |
| --- | --- | --- | --- |
| QB | 0.319 | 0.546 | EPA |
| RB | 0.242 | 0.256 | EPA |
| WR | 0.224 | 0.399 | EPA |
| TE | 0.336 | 0.274 | grade |
| DL | 0.125 | 0.106 | grade |
| LB | 0.197 | 0.103 | grade |
| DB | 0.137 | 0.085 | grade |

## Draft check

Rank correlation of the final-week grade and the v4 value with draft slot
(pick negated, so higher is better) and with NFL scrimmage snaps over the
first three seasons. Draft year y reads season y-1, so draft classes 2021
through 2026 enter the pick columns and only the mature classes 2021
through 2023 enter the NFL columns. Every season except 2025 is a
development season, so these are in sample for the weights; only the 2026
class is out of sample. A player needs both a grade and a v4 value. They are
reported for the draft board and were never used to set a weight.

| Group | Players | Pick, v4 | Pick, grade | NFL snaps, v4 | NFL snaps, grade |
| --- | --- | --- | --- | --- | --- |
| QB | 61 | 0.418 | 0.338 | 0.032 | -0.098 |
| RB | 109 | 0.417 | 0.344 | 0.386 | 0.303 |
| WR | 179 | 0.433 | 0.223 | 0.268 | 0.070 |
| TE | 82 | 0.388 | 0.352 | 0.148 | 0.176 |
| DL | 253 | 0.212 | 0.287 | 0.037 | 0.149 |
| LB | 146 | 0.078 | 0.054 | 0.064 | 0.120 |
| DB | 285 | 0.251 | 0.176 | 0.211 | 0.186 |
| P | 1 |  |  |  |  |

The grade ranks the draft better than v4 only at DL (0.287 against 0.212 on
slot, 0.149 against 0.037 on snaps). It is ahead on NFL snaps at LB and TE
and behind everywhere else. At WR it is well behind on both (0.223 against
0.433 on slot, 0.070 against 0.268 on snaps). OL units have no v4 value and
do not enter.

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
`output/position-grade-2026-20261016.log`). Local raw data for 2026 ends at
week 1, so the artifact holds weeks 0 and 1: 65,295 rows, with 4,913
qualified FBS players graded at QB, RB, WR, TE, DL, LB, DB, OL and P. It ran
in receptions mode and used the frozen expectation tables, reading no
earlier play-by-play season.

Each artifact row carries the trait snapshot columns plus `uncertainty`
(empty on trait rows), `in_grade` (true on a trait row whose trait has
positive weight at the position, and on every grade row), `fbs` and
`poor_coverage`. Grade rows have `trait = "grade"`, the grade as `value`,
its percentile as `percentile` and `kind = "graded"`. The publisher keeps
only the latest week and only FBS rows; the table has no `fbs` or
`poor_coverage` column, and the migration
`sql/20261016200339_position_traits.sql` (not yet applied) declares
`in_grade` and `uncertainty`.

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

## Limits

There are no tackle attributions on plays, no snap counts, no pressures and
no coverage assignments, so defensive traits rest on box score counts and
with-without comparisons. Offensive linemen are rated as a unit, one row
per team, and a unit never transfers, so OL portability cannot be
measured. 2019 has no opponent prior, so development
starts in 2020. Targets are missing for most incompletions in 2023, 2024
and 2026, which ran in receptions mode. Win probability is missing for
games with an FCS side.

The grade has no volume term. A player with one weighted trait is graded on
that trait alone, at full scale, so a thinly measured player can sit at the
top of his position; the uncertainty band reads only games, not how many
traits he has. Two of the development seasons are in receptions mode. The
weights are as precise as split-half correlations on hundreds to thousands
of players allow, which is about two decimals; the QB receiver independence
weight rests on 36 players and should be read as rough. 2026 was never used
and is judged only as it is served.

## Findings

Plainly, against EPA credit alone on the 2024 to 2025 holdout:

- **Defense: the grade wins.** At DL, LB and DB a player's grade predicts
  next season's EPA credit better than his EPA credit does (DL 0.125
  against 0.106, LB 0.197 against 0.103, DB 0.137 against 0.085). The same
  holds in every development pair at DL and DB and in three of four at LB.
  Defensive EPA credit barely persists (0.085 to 0.106 on the holdout); the
  box score traits do. The grade also persists far better than defensive
  EPA (DL 0.384, LB 0.409, DB 0.272 grade to grade), and at DL it ranks the
  draft better than v4.
- **QB and WR: EPA credit wins clearly.** QB 0.319 against 0.546, WR 0.224
  against 0.399, and EPA wins in every development pair at WR and in all
  four at QB. At WR the grade is also far behind v4 on the draft. For
  passers and receivers the traits do not yet add up to what the credited
  plays already say.
- **RB: EPA credit wins narrowly** (0.242 against 0.256), and the RB grade
  is the least reliable composite (0.084 at 6 games). It should not be
  shown as a ranking without its band.
- **TE: the grade wins on the holdout** (0.336 against 0.274) but lost in
  three of four development pairs, so the holdout win is not a pattern yet.
- **OL and P** have a grade that persists (OL 0.328, P 0.557 grade to grade
  on the holdout) and no EPA baseline to compare with. K has no grade.

The grade is not EPA per play with a new name: it is built from traits
only, and on defense it carries information EPA credit does not. On
offense it is a descriptive profile that predicts less than EPA credit, and
the card should lead with impact there.
