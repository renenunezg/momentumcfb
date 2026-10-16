# Position trait ratings: design

Date: 2026-10-16. Status: approved by Rene on 2026-10-16, including the seven CFBD portal calls. Implementation complete on 2026-10-16; the backtest report is at `docs/position_traits_v1.md`.

## Intent

Rate every FBS player at their position with a few named traits and one headline number, so that a weekly list of the best players at each position, a conversation about who the best defensive lineman is, and the draft board all read from the same evidence. The rating must say something beyond EPA per play and the usual rate stats, it must be opponent adjusted, and every trait must earn its place by predicting what happens in future games.

This is not an extension of the Heisman model, which predicts vote share. It extends the player value model (`backend/players/value.py`, `cfb_player_value_v4`), which already credits each play to the players the feed tags, adjusts for the opponent's unit strength fitted strictly before that week, shrinks toward the position mean, and subtracts a replacement baseline. v4 becomes the baseline the new traits are measured against and, if the new headline beats it, the thing it replaces.

## Goals

- A per-position card: headline rating in points per game above replacement, position percentile, an honest uncertainty band, and four to seven named traits as position percentiles.
- Traits that are explainable in one sentence to a fan and that separate the player from the team around them wherever the data allows.
- A backtest that decides, per position, which traits carry weight in the headline, reported as a document a reader can check.
- Weekly snapshots for the live season with the same chronological discipline as player value v4: nothing from the week being rated enters its own rating.
- Replace player value v4 on the site and in the draft board if the headline is better, by the rule in Evaluation.

## Non-goals

- No refit of team models, unit ratings, or the in-game win probability baseline.
- No new per-game CFBD pulls beyond what the player ingest already makes; no play-by-play from other providers.
- No tuning of any weight on 2026 outcomes. 2026 is served and judged, never fitted.
- No snap counts, pressures, air yards, coverage assignments, or tackler attribution. The data does not have them, and the design does not pretend otherwise.
- No betting framing. Player ratings are not pick inputs.
- Heisman board unchanged.

## Anti-goals (what would make this a failure even if it runs)

- A headline that is EPA per play with a new name. If no unconventional trait earns weight at a position, the report must say so plainly rather than keeping it for show.
- Traits with weights picked by hand or on a coarse grid.
- A rating that moves because of who a player's teammates are rather than what the player did, where the data could have separated them.
- A card that shows a percentile without the games behind it.

## Constraints

- Data: per-play attribution (`/plays/stats`) covers essentially every FBS game from 2019 on, roughly 850 to 890 games a season. Box scores add per-game defensive counts (solo and total tackles, TFL, sacks, hurries, passes defended) from 2014 on. Opponent adjustment the v4 way needs a previous-season prior built from play-by-play, which starts in 2019, so 2019 cannot be adjusted consistently. Development seasons are therefore 2020 through 2024, validation is 2025, live is 2026, matching the v4 document. (Claude's call after the reviewer pass.)
- The stored historical player values mix model versions (2019 is v1, 2020 is v3). The builder rebuilds 2020 through 2025 under v4 before any backtest so the baseline is one model.
- Target tagging in `/plays/stats` is inconsistent: a Target row exists on about half of 2021 incompletions and about a fifth of 2024 incompletions. Play text names the intended receiver on some incompletions. The builder recovers targets by parsing `play_text` where it names a receiver, matched to the roster, and reports per-season target coverage; where coverage for a season is poor the target-based traits for that season fall back to receptions and the report says so. (Claude's call.)
- Offense is tagged per play with passer, receiver or target, rusher, fumbler. Defense is tagged per play only on sacks, interceptions, pass breakups (sparse), forced and recovered fumbles. Tacklers are never tagged and play text does not name them.
- Every play carries down, distance, yards to goal, clock, score, and PPA, and the frozen in-game baseline (`backend/model/ingame.py`) turns any play into win probability added.
- The competitive flag and garbage-time rule from `backend/features/possessions.py` apply to every trait.
- Opponent adjustment reuses the per-play unit effects from v4 (`season_unit_effects`), fitted from games before the week being rated.
- Portal transfers are on disk for 2026 only (`backend/data/raw/preseason/2026/portal.parquet`), with names and no athlete id. Calibrating the teammate-relative traits on 2020 through 2025 needs one CFBD `/player/portal` call per season, about seven calls, plus name matching to rosters. This pull needs Rene's approval under the API budgeting rule. Returning production (`backend/data/raw/returning`) is team level only, so the "returning line" component of the OL split and of the QB pressure baseline is dropped; those use the team's sack rate with other passers and transfers alone.
- The weekly workflow already runs `player-values` and publishes `cfb.player_values`; the draft research reads `processed/players/player_values/<season>.parquet` by `athlete_id`.
- Supabase free plan: egress and database size must stay inside it. One snapshot row per player per week at the current cardinality is fine; the trait table must not multiply that by much.
- Project rule: weights are chosen on chronological development data with a bounded optimizer or fine search, validated independently, and never claimed to more precision than the sample supports.

## Decided: the card

Per position group (QB, RB, WR/TE, DL, LB, DB, OL, K/P, in that priority order), each player-week carries:

- `headline`: points per game above positional replacement, the fitted combination of traits described below.
- `position_percentile`: headline rank among qualified players at the position that week.
- `uncertainty`: a band from the split-half reliability of the headline at the player's game count.
- Traits: each a position percentile with its raw value, its games, and a flag saying whether it is counted (from tagged plays or box counts) or inferred (from unit outcomes or teammate comparisons), and whether it carries headline weight at that position.

Candidate traits, as agreed from the mockup. The backtest may drop any of them from the headline; the card keeps dropped traits as descriptive and says so.

QB
- Drive engine: late-down (third and fourth) and two-minute conversion rate above what the situation expects, leverage weighted.
- Explosive: share of passing value from throws of 20 or more yards.
- Ball security: win probability cost of interceptions and fumbles per dropback, regressed toward the position mean.
- Floor: the 20th percentile of per-game value, not the mean.
- Legs: scramble and designed-run success rate, opponent adjusted.
- Pressure handling (inferred): sacks taken per dropback relative to the line's protection baseline, where the baseline is the team's sack rate allowed with other passers and the previous season's returning line.
- Receiver independence (experimental): the passer's value beyond what his receivers produce with other passers, from plays that tag both.

RB
- Yards over expected per carry given down, distance, field position, and the opponent's run defense.
- Stuff avoidance: share of carries at or below zero, relative to the team's other backs.
- Breakaway: rate of carries of 10 and 20 or more yards.
- Short yardage: conversion rate on third and fourth down with two or fewer to go, and goal-to-go.
- Receiving value per target.
- Ball security as for QB.
- Line independence (inferred): efficiency relative to teammate backs behind the same line in the same games.

WR/TE
- Target share of the team's competitive targets.
- Value per target above expected.
- Explosive catch rate.
- Late-down conversion share.
- QB independence (inferred): value relative to the same passer's other targets.

DL (including EDGE)
- Havoc: TFL, sacks, forced fumbles, breakups per game against the opponent's line.
- Pass rush: sacks and hurries per game, weighted by the win probability each sack swung.
- Finishing: sacks as a share of sacks plus hurries.
- Run stop: TFL and stop share in the run game.
- Every week: share of games with at least one disruption.
- On-field impact (inferred): unit EPA allowed with versus without the player, where absence is a game with no box row, opponent adjusted and shrunk hard.

LB
- Tackle share, run stops (TFL), coverage production (PBU, INT), havoc, on-field impact as for DL.

DB
- Ball production (INT, PBU), pass EPA allowed with versus without, run-support tackles. Tackle counts are not a plus by default because a high count can mean catches allowed; the backtest decides the sign.

OL
- Unit only: sack rate allowed adjusted for the passer, stuff rate, short-yardage conversion, adjusted line yards. Individual linemen carry the unit score flagged inferred, with the share split by returning production and transfers where the data allows. Builder's call on whether to publish individual OL rows at all in the first version.

K/P
- Field goal value above expected by distance; punt net above expected. Counted, low priority.

## Decided: identification levers

These are the parts that make the rating about the player rather than the team, and they are the reason for the project.

- Teammate-relative splits: the feed tags both players on every pass and the rusher on every carry, so a receiver can be compared against the same passer's other targets, a back against other backs behind the same line, a passer against his receivers' output with other passers.
- Transfers as natural experiments: a player who changed teams (portal data) identifies how much of their output travelled with them. Used to calibrate how much to trust the teammate-relative traits and the OL split, not as a trait of its own.
- With and without: for defenders, team unit EPA allowed in games the player appears in the box score versus games they do not, opponent adjusted. Only estimable for players with absences; shrunk toward zero by games present and absent.
- Leverage and shape: win probability added instead of EPA where the trait is about moments; floor and consistency across games where it is about reliability.

## Decided: the headline fit

- One joint fit across positions, not one per position, so that no position's traits can absorb the whole team's quality. The target for a game is the actual margin net of the opponent's team rating and home field, both from the existing pregame team model as of that week. The predictors are, per position, the team's roster aggregated from prior weeks' usage (the exact aggregation is the builder's call) over the standardized traits, and the v4 EPA credit enters the same fit as one trait per position. A trait therefore earns weight only for what it adds beyond EPA credit, which is how the anti-goal is enforced inside the fitting criterion rather than by hand. A player's headline is the fitted combination of their own traits, including their EPA credit, scaled to points per game above positional replacement. (Claude's call after the reviewer pass.)
- The fit uses 2020 through 2024 with each season walked forward week by week; the trait values for week w use only games before w.
- Inferred traits enter the fit whether or not transfer calibration is available for a season; calibration is reported, and a trait whose calibrated share of player signal is near zero is dropped like any other. (Claude's call.)
- Weights come from a bounded optimizer or a fine search, not a coarse grid, with a ridge penalty chosen the same way. Weights are reported to the precision the development seasons support and no more.
- A trait whose weight is not distinguishable from zero on development data is dropped from the headline and kept on the card as descriptive.
- 2025 is scored once with the frozen weights as the independent check. 2026 is never scored during development.

## Decided: evaluation and the replacement rule

The backtest produces a document, `docs/position_traits_v1.md`, patterned on `docs/player_value_v4.md`, with per position:

- Which traits earned headline weight, and the weights.
- Out-of-sample next-game margin error of the headline versus player value v4 on 2025, by week bucket.
- Split-half reliability of each trait and of the headline at 4, 8 and 12 games, which also feeds the card's uncertainty band.
- Secondary checks for the draft board only: rank correlation with NFL draft slot and with the three-season NFL participation cohort already built by the draft research. These are never used in fitting.

Replacement rule, decided by Rene: the new headline replaces player value v4 on the site and in the draft board at a position if it beats v4 on 2025 out-of-sample margin prediction and is at least as reliable at 6 games. Where it does not, it ships beside v4 and the report says why. The position-level margin test swaps one position's representation (headline versus v4 EPA credit, each on its own fitted scale) while holding the others fixed. (Claude's call.)

`cfb.player_values` stays v4 and untouched because the Heisman board reads it. Replacement happens at the consumers, and the trait table carries a per-position flag saying whether the headline replaced v4 there.

## Decided: serving

- New weekly artifact `processed/players/position_traits/<season>.parquet` and a new table `cfb.position_traits`, one row per player-week-trait plus the headline row, published by the existing `publish-players` path after `player-values`. The exact row shape (long versus one wide row per player-week) is the builder's call, bounded by Supabase size.
- The weekly workflow runs the new step after `player-values`. Development fits are an offline command producing a frozen weights artifact, installed like the Heisman model bundle, so the weekly job never refits.
- The site (`momentumweb`) and the draft board consume the new table. Changes to the site are a separate piece of work in that repository and are out of scope for this document beyond the table contract.

## Left to the builder

- Expected-yards and expected-conversion baselines: model form and features, as long as they are fitted on earlier seasons only.
- Shrinkage strengths for each trait and the with-without estimator.
- Exact leverage weighting for the moment traits (WPA directly or capped).
- Whether individual OL rows are published in v1.
- How the trait table is shaped and indexed.
- Which teammate-relative comparisons to keep when the sample behind them is tiny.

## Open questions for Rene

- Approve about seven CFBD `/player/portal` calls (one per season, 2020 through 2025) so transfers can calibrate the teammate-relative traits on the development seasons.
