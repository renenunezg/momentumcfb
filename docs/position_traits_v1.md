# Position traits v1: per-position trait ratings and the grade

## What the model is

Every FBS player gets a set of named traits at his position and two numbers.

- The **grade** is a 20 to 80 scouting number built from his traits. It has
  mean 50 and sd 10 among qualified FBS players at his position that week.
  It says how good his profile is, not how many points he added.
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
on 2026-10-16 (`output/position-grade-fit-202610162203.log`). Player values
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

A reliability measured on fewer than 30 players counts as unmeasured. 30
is a sample floor, not a tuned value. An unmeasured portability is imputed
as the mean measured portability of the position's traits (the mean over
every position's traits when the position has none measured), so an
unmeasured trait is weighted like a typical trait at its position rather
than like a perfectly portable one; the model flags it
`portability_imputed`. Then `weight = max(reliability, 0) x portability`,
normalized to sum to 1 within the position. Each trait gets a status:

| Status | Rule | Weight |
| --- | --- | --- |
| weighted | reliability at least 0.10, portability measured and positive | reliability x portability, normalized |
| unmeasured | reliability at least 0.10, portability not measurable | reliability x imputed portability, normalized |
| noise | reliability under 0.10 | 0 |
| unmeasured_reliability | reliability cannot be measured: under 30 players, or no per-game values | 0 |
| team-driven | portability measured and at or below zero | 0 |

Traits with no weight stay on the card, dimmed, as descriptive.

Orientation: a trait registered higher-is-better enters as is, one
registered lower-is-better is negated, and one with no registered direction
takes the sign of its Spearman correlation with v4 `shrunk_per_game` among
qualified FBS players at the position, at the final week of each development
season. Only `def.run_support` (DB tackles in the run game) has no
registered direction. Its correlation with v4 is 0.200 over 4,795 player
seasons, so it enters with a positive sign. The sign is stored in the model.

## Weights

Reliability is the pooled 8-game split-half correlation; portability the
clipped portable share, or the imputed position mean where marked. Weights
sum to 1 within each position.

| Group | Trait | Reliability | Portability | Weight | Status |
| --- | --- | --- | --- | --- | --- |
| QB | qb.drive_engine | 0.191 | 1.000 | 0.123 | weighted |
| QB | qb.explosive | 0.288 | 0.360 | 0.067 | weighted |
| QB | qb.ball_security | 0.115 | 1.000 | 0.074 | weighted |
| QB | qb.floor | 0.269 | 0.672 | 0.116 | weighted |
| QB | qb.legs | 0.160 | 1.000 | 0.103 | weighted |
| QB | qb.pressure_handling | 0.351 | 1.000 | 0.226 | weighted |
| QB | qb.receiver_independence | 0.537 | 0.839 (imputed) | 0.290 | unmeasured |
| RB | rb.yards_over_expected | 0.150 | 1.000 | 0.238 | weighted |
| RB | rb.stuff_avoidance | 0.108 | 1.000 | 0.171 | weighted |
| RB | rb.breakaway | 0.172 | 0.434 | 0.119 | weighted |
| RB | rb.short_yardage | -0.044 | 0.960 | 0 | noise |
| RB | rb.receiving | 0.190 | 1.000 | 0.302 | weighted |
| RB | rb.ball_security | -0.051 | 0.899 (imputed) | 0 | noise |
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
| DL | def.impact |  | 0.702 (imputed) | 0 | unmeasured_reliability |
| LB | def.havoc | 0.424 | 0.596 | 0.171 | weighted |
| LB | def.pass_rush | 0.138 | 0.808 | 0.075 | weighted |
| LB | def.finishing | 0.141 | 1.000 | 0.095 | weighted |
| LB | def.run_stop | 0.290 | 0.522 | 0.102 | weighted |
| LB | def.every_week | 0.360 | 0.635 | 0.154 | weighted |
| LB | def.tackle_share | 0.771 | 0.643 | 0.335 | weighted |
| LB | def.coverage | 0.188 | 0.529 | 0.067 | weighted |
| LB | def.impact |  | 1.000 | 0 | unmeasured_reliability |
| DB | def.havoc | 0.290 | 0.495 | 0.239 | weighted |
| DB | def.every_week | 0.259 | 0.593 | 0.256 | weighted |
| DB | def.ball_production | 0.262 | 0.504 | 0.220 | weighted |
| DB | def.run_support | 0.485 | 0.353 | 0.285 | weighted |
| DB | def.impact |  | 0.000 | 0 | unmeasured_reliability |
| DB | def.pass_impact |  | 0.000 | 0 | unmeasured_reliability |
| OL | ol.protection | 0.311 | 0.716 (imputed) | 0.337 | unmeasured |
| OL | ol.stuff_rate | 0.253 | 0.716 (imputed) | 0.274 | unmeasured |
| OL | ol.short_yardage | 0.024 | 0.716 (imputed) | 0 | noise |
| OL | ol.line_yards | 0.360 | 0.716 (imputed) | 0.389 | unmeasured |
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
  the smallest QB weight (0.067).
- `qb.receiver_independence` still carries the largest QB weight (0.290).
  Its reliability rests on 36 players, above the 30 floor, and its
  portability (12 movers) is imputed at the QB mean, 0.839. At the final
  week of 2025 only 69 of the 170 qualified FBS QBs have a value for it
  (`scored_2025.parquet`), under the availability bar, so that week it
  leaves every QB's composite (see The grade).
- The with-without impacts (`def.impact`, `def.pass_impact`) carry no
  weight: they have no per-game values, so their reliability cannot be
  measured, and at DB their portability is at or below zero as well.
- No trait was marked team-driven: every trait with a non-positive portable
  share already had no measurable reliability.
- K has no grade: `k.field_goal` is not reliable (0.036). P is graded on
  punt average alone.
- OL units never move, so OL has no measured portability. Its traits take
  the mean over every position's measured traits, 0.716, the same for all
  three OL traits that pass the reliability cut, so their relative weights
  follow reliability.

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
  player at the position is graded on the same traits. Structural gaps
  (`qb.floor` before a fourth game, prior-week baselines in week one,
  receiver independence before most QBs have it) drop out this way;
- `composite = sum(w_t z_t) / sum(w_t)` over the available traits, the
  weights renormalized for the week, with z = 0 (the position average) for
  an available trait the player has no value for, so a thin profile is
  pulled toward average instead of being judged on the few traits it has;
- a player is graded only if he has a value for his position's primary
  exposure trait and the available traits he has values for carry at least
  0.6 of the available weight, so one unusually thin profile still fails
  (`PRIMARY_TRAITS`, `MIN_WEIGHT_COVERAGE`, `MIN_TRAIT_AVAILABILITY` and
  `Z_CAP` in `backend/players/traits/grade.py`);
- `grade = 50 + 10 x (composite - mean) / sd` over the graded players,
  clipped to [20, 80];
- `grade_percentile` ranks the composite among them, 0 to 100;
- `uncertainty = 10 x sqrt(1 - r)`, with `r` the composite's split-half
  reliability at the player's game count, read off the curve below (linear
  from r = 0 at zero games, flat past 12 games).

| Group | Primary trait |
| --- | --- |
| QB | qb.explosive (dropbacks) |
| RB | rb.yards_over_expected (carries) |
| WR, TE | rec.value_per_target (targets) |
| DL, LB, DB | def.every_week (games) |
| OL | ol.line_yards |
| K | k.field_goal |
| P | p.punt_average |

Unqualified and non-FBS players, and qualified players who fail the
coverage rule, keep their trait rows and get no grade row.

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

At the end of a season the rule removes almost no one: in the development
seasons 0 or 1 QB, 0 to 4 backs, 1 or 2 receivers and 0 to 3 tight ends a
season, all for lacking the primary trait. Early in a season it removes
more: see Scoring 2026.

## Composite reliability

The composite's own split-half correlation, pooled over the development
seasons. Each player's games alternate between two halves in week order,
each half is scored as its own snapshot week with the frozen weights, and
the two composites are correlated. This curve sets the uncertainty band.

| Group | 4 games | 6 games | 8 games | 12 games | Players at 4 | Players at 6 | Players at 8 | Players at 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QB | 0.193 | 0.245 | 0.333 | 0.371 | 934 | 751 | 598 | 260 |
| RB | 0.084 | 0.088 | 0.121 | 0.103 | 2101 | 1771 | 1430 | 592 |
| WR | 0.169 | 0.223 | 0.233 | 0.270 | 3310 | 2698 | 2101 | 717 |
| TE | 0.124 | 0.163 | 0.204 | 0.153 | 1153 | 838 | 585 | 115 |
| DL | 0.224 | 0.302 | 0.362 | 0.422 | 5375 | 4454 | 3500 | 1018 |
| LB | 0.547 | 0.582 | 0.605 | 0.611 | 4155 | 3404 | 2711 | 1157 |
| DB | 0.341 | 0.365 | 0.383 | 0.340 | 6247 | 5084 | 4012 | 1468 |
| OL | 0.206 | 0.312 | 0.346 | 0.407 | 653 | 640 | 612 | 496 |
| P | 0.261 | 0.334 | 0.442 | 0.504 | 655 | 607 | 557 | 386 |

Each half is graded under the same rules, with availability judged within
the half, so a half too short for `qb.floor` drops it for every QB.

The RB composite is barely reliable: 0.121 at 8 games, lower than three of
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
| QB | 2020-2021 | 62 | 0.214 | 0.209 | 0.374 |  |
| QB | 2021-2022 | 89 | 0.335 | 0.381 | 0.518 |  |
| QB | 2022-2023 | 83 | 0.280 | 0.188 | 0.408 |  |
| QB | 2023-2024 | 82 | 0.413 | 0.450 | 0.390 |  |
| QB | 2024-2025 | 82 | 0.371 | 0.399 | 0.546 | yes |
| RB | 2020-2021 | 166 | 0.306 | 0.348 | 0.328 |  |
| RB | 2021-2022 | 197 | 0.188 | 0.215 | 0.340 |  |
| RB | 2022-2023 | 224 | 0.285 | 0.281 | 0.373 |  |
| RB | 2023-2024 | 199 | 0.261 | 0.263 | 0.417 |  |
| RB | 2024-2025 | 178 | 0.177 | 0.241 | 0.256 | yes |
| WR | 2020-2021 | 248 | 0.110 | 0.204 | 0.395 |  |
| WR | 2021-2022 | 305 | 0.203 | 0.289 | 0.383 |  |
| WR | 2022-2023 | 304 | 0.150 | 0.157 | 0.316 |  |
| WR | 2023-2024 | 292 | 0.307 | 0.261 | 0.372 |  |
| WR | 2024-2025 | 270 | 0.205 | 0.215 | 0.397 | yes |
| TE | 2020-2021 | 67 | 0.147 | 0.259 | 0.329 |  |
| TE | 2021-2022 | 85 | 0.108 | 0.135 | 0.337 |  |
| TE | 2022-2023 | 79 | 0.316 | 0.327 | 0.473 |  |
| TE | 2023-2024 | 89 | 0.203 | 0.291 | 0.275 |  |
| TE | 2024-2025 | 81 | 0.237 | 0.284 | 0.294 | yes |
| DL | 2020-2021 | 344 | 0.437 | 0.301 | 0.142 |  |
| DL | 2021-2022 | 492 | 0.451 | 0.251 | 0.122 |  |
| DL | 2022-2023 | 562 | 0.387 | 0.197 | 0.129 |  |
| DL | 2023-2024 | 540 | 0.429 | 0.251 | 0.153 |  |
| DL | 2024-2025 | 453 | 0.388 | 0.126 | 0.106 | yes |
| LB | 2020-2021 | 306 | 0.493 | 0.174 | 0.169 |  |
| LB | 2021-2022 | 360 | 0.484 | 0.149 | 0.185 |  |
| LB | 2022-2023 | 358 | 0.421 | 0.261 | 0.109 |  |
| LB | 2023-2024 | 374 | 0.362 | 0.167 | 0.060 |  |
| LB | 2024-2025 | 341 | 0.408 | 0.199 | 0.103 | yes |
| DB | 2020-2021 | 450 | 0.286 | 0.098 | 0.032 |  |
| DB | 2021-2022 | 554 | 0.317 | 0.110 | 0.106 |  |
| DB | 2022-2023 | 566 | 0.348 | 0.124 | 0.095 |  |
| DB | 2023-2024 | 564 | 0.311 | 0.138 | 0.082 |  |
| DB | 2024-2025 | 517 | 0.272 | 0.137 | 0.085 | yes |
| OL | 2020-2021 | 112 | 0.371 |  |  |  |
| OL | 2021-2022 | 130 | 0.374 |  |  |  |
| OL | 2022-2023 | 131 | 0.373 |  |  |  |
| OL | 2023-2024 | 133 | 0.515 |  |  |  |
| OL | 2024-2025 | 134 | 0.329 |  |  | yes |
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
| QB | 0.399 | 0.546 | EPA |
| RB | 0.241 | 0.256 | EPA |
| WR | 0.215 | 0.397 | EPA |
| TE | 0.284 | 0.294 | EPA |
| DL | 0.126 | 0.106 | grade |
| LB | 0.199 | 0.103 | grade |
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
| QB | 61 | 0.418 | 0.423 | 0.032 | 0.064 |
| RB | 109 | 0.417 | 0.355 | 0.386 | 0.310 |
| WR | 179 | 0.433 | 0.223 | 0.268 | 0.046 |
| TE | 82 | 0.388 | 0.339 | 0.148 | 0.134 |
| DL | 253 | 0.212 | 0.287 | 0.037 | 0.137 |
| LB | 146 | 0.077 | 0.045 | 0.064 | 0.115 |
| DB | 285 | 0.250 | 0.178 | 0.210 | 0.188 |
| P | 1 |  |  |  |  |

The grade ranks the draft better than v4 at DL (0.287 against 0.212 on
slot, 0.137 against 0.037 on snaps) and narrowly at QB (0.423 against 0.418
on slot, 0.064 against 0.032 on snaps, on 61 players). It is ahead on NFL
snaps at LB (0.115 against 0.064) and behind everywhere else. At WR it is
well behind on both (0.223 against 0.433 on slot, 0.046 against 0.268 on
snaps). OL units have no v4 value and
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
`output/position-grade-2026-202610162203.log`). Local raw data for 2026 ends
at week 1, so the artifact holds weeks 0 and 1: 67,812 rows, with 4,552
qualified FBS players graded and 2,914 war rows. It ran in receptions mode
and used the frozen expectation tables, reading no earlier play-by-play
season. There is no calibration replay for 2026, so the WAR leverage came
from the stored weekly serving anchors for weeks 0 and 1.

The coverage rule in week 1:

| Group | Qualified FBS | Graded | No primary trait | Coverage under 0.6 |
| --- | --- | --- | --- | --- |
| QB | 179 | 160 | 17 | 2 |
| RB | 391 | 316 | 10 | 65 |
| WR | 613 | 512 | 101 | 0 |
| TE | 239 | 199 | 40 | 0 |
| DL | 1100 | 1100 | 0 | 0 |
| LB | 840 | 840 | 0 | 0 |
| DB | 1288 | 1288 | 0 | 0 |
| OL | 137 | 137 | 0 | 0 |
| P | 126 | 126 | 0 | 0 |

After one game `qb.floor` and receiver independence are unavailable for
everyone, so they leave the QB denominator and 160 of 179 QBs are graded.
The 65 backs under 0.6 have values for too few of the traits most backs
already have. In receptions mode many receivers have no value per target
yet, which is why 101 WRs and 40 TEs lack the primary trait.

`backtest.score_cached_season(season)` scores an already-built season from
its cached snapshot, coverage and credit with the latest model before that
season, without rebuilding anything or reading the expectation tables, and
writes `processed/players/position_traits/scored_<season>.parquet` with the
runner's columns. For 2025 it wrote 1,112,719 rows, 81,386 of them grade
rows and 66,626 war rows (log
`output/position-grade-scored-2025-202610162203.log`), for a local preview
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

The grade has no volume term beyond the coverage rule: a missing trait
counts as average, a player needs his primary trait and 0.6 of the
available weight, and no trait z-score counts beyond plus or minus 3. The uncertainty band reads
only games, not how many traits a player has. Two of the development seasons are in receptions mode. The
weights are as precise as split-half correlations on hundreds to thousands
of players allow, which is about two decimals; the QB receiver independence
weight rests on 36 players and should be read as rough. 2026 was never used
and is judged only as it is served.

## Findings

Plainly, against EPA credit alone on the 2024 to 2025 holdout:

- **Defense: the grade wins.** At DL, LB and DB a player's grade predicts
  next season's EPA credit better than his EPA credit does (DL 0.126
  against 0.106, LB 0.199 against 0.103, DB 0.137 against 0.085). The same
  holds in every development pair at DL and DB and in three of four at LB.
  Defensive EPA credit barely persists (0.085 to 0.106 on the holdout); the
  box score traits do. The grade also persists far better than defensive
  EPA (DL 0.388, LB 0.408, DB 0.272 grade to grade), and at DL it ranks the
  draft better than v4.
- **QB: EPA credit still wins, by less.** 0.399 against 0.546 on the
  holdout. EPA wins three of four development pairs; the grade wins 2023
  to 2024 (0.450 against 0.390). On the draft the QB grade is level with v4
  (0.423 against 0.418 on slot, 61 players).
- **WR: EPA credit wins clearly** (0.215 against 0.397), in every
  development pair, and the grade is far behind v4 on the draft. For
  receivers the traits do not yet add up to what the credited plays say.
- **RB and TE: EPA credit wins narrowly** (RB 0.241 against 0.256, TE 0.284
  against 0.294). The grade won one of four development pairs at each. The
  RB grade is the least reliable composite (0.088 at 6 games) and should
  not be shown as a ranking without its band.
- **OL and P** have a grade that persists (OL 0.329, P 0.557 grade to grade
  on the holdout) and no EPA baseline to compare with. K has no grade.
- **WAR is a QB list.** All ten 2025 leaders are quarterbacks, because v4
  credits passers with the largest share of every pass and defenders only
  for tagged disruptions. WAR compares players within a position; across
  positions it mostly reflects the credit rules.

The grade is not EPA per play with a new name: it is built from traits
only, and on defense it carries information EPA credit does not. On
offense it is a descriptive profile that predicts less than EPA credit, and
the card should lead with impact (WAR) there.
