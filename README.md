# Momentum CFB

Momentum CFB is a college football rating, projection, and model-evaluation pipeline.
It builds preseason and in-season team ratings, prices game margins and totals, captures sportsbook closing lines, and publishes reviewed outputs to the shared Momentum website.

The public interface is maintained in the separate `momentumweb` repository.
Current ratings, methodology, and performance are available at [renenunez.dev/cfb](https://renenunez.dev/cfb/ratings).

## Current status

The 2026 season is live.
The pre-kickoff refresh produced 266 Division I team ratings, 172 week-one game projections, unit ratings, market comparisons, and outcome-free serving anchors.
Completed games are graded against the projection that was published before kickoff and the CFBD closing line, and the graded record is published to the site.
Every settled pick also records the CFBD median closing line for its side and its closing line value, the points by which the pick beat the close.

| Live 2026 measure (as of Sep 1) | Value |
|---|---:|
| Graded games | 53 |
| Graded games with a closing spread | 53 |
| Pure model margin MAE | 12.86 points |
| Closing line margin MAE | 11.80 points |
| 80 percent interval coverage | 0.85 |

Only 8 of those games are FBS versus FBS, so the live sample is too small to draw conclusions from.
Historical walk-forward results below are kept separate from the live record.

## Model design

The in-season model estimates offense, defense, pace, and home-field effect from chronological points-per-possession and EPA-per-possession data.
It produces a joint home and away score distribution with calibrated margin and total uncertainty.
Each team's offense and defense priors are correlated at 0.5, so the scoring-environment direction is shrunk harder than net strength, and the pace prior is one possession; on the 2020 through 2025 walk-forward this keeps projected totals calibrated (actual on projected total slope 0.88 against 0.61 with independent priors) with margins unchanged.

The preseason model starts from prior-season power, scoring environment, and pace.
It then incorporates current CFBD talent, returning production, transfers, quarterback continuity, recruiting, and coaching continuity.
The prior-season, talent, and returning-production weights are fitted to 2022 through 2025 opening-week closing margins and validated leave-one-season-out.
Service academies do not sign rated recruiting classes, so their talent composite is treated as missing instead of as a near-zero roster.
Missing inputs receive neutral contributions and increase uncertainty instead of being guessed.
Each conference pool (FCS as one pool) carries a shift parameter identified by crossover games, so a whole conference can move from the level its teams' priors imply instead of every team being shrunk toward a stale level, and FCS teams without priors are anchored to the fitted FCS level.
Ratings are fitted to a blend of points and competitive-possession EPA, which discounts garbage-time scoring; games between distant pools carry far more of it, so a crossover gain estimated from points residuals adds a share of the pool-level gap to the stronger side (FBS over FCS bias +6.1 to +1.3 and Power Five over Group of Five +3.1 to +0.2 on the 2020 through 2025 walk-forward).
Team strength scales with the game's expected possessions, so a slow or fast matchup changes the projected margin and total the way it changes the scoring.
The published margin is half the joint fit's margin and half a points-only ridge rating fitted on the same games with a nine-point prior around the previous season's final points-only rating (joint_scoring_v12); the total is untouched.
The points-only rating keeps information the blended fit shrinks away, mainly the crossover gain that restarts each season and the FCS pool, and on the 2020 through 2025 walk-forward the blend cut margin MAE by 0.19 points on Division I games with a closing line and 0.07 on FBS versus FBS, negative in every season.
The points-only rating and its carried chain use every completed game with a final score, including the FCS games whose play-by-play features are missing (joint_scoring_v13), which cut 2023 through 2025 margin MAE by a further 0.06 on Division I games and 0.12 on FCS versus FCS.
The market-informed margin is a separate output that never feeds the pure margin: its model side mixes the pure margin with a rating fitted to the closing spreads of games that kicked off before the forecast week (weight 0.55 through week 3, 0.35 after; prior carried from the previous season's closing lines), and that side is then averaged with the game's own consensus line.
Selected on 2020 through 2022 and scored once on 2023 through 2025, the market-history rating cut the market-informed margin MAE from 12.07 to 12.00 (t -4.3), negative in every season.
The projected margin standard deviation is scaled by 0.915, which returns 80 percent interval coverage from 0.85 to 0.81.

The in-game baseline combines the current score and possession state with a frozen pregame anchor.
Serving code reads only outcome-free anchors and play state.
Final scores are available only to evaluation code.

## Validation

Model selection uses 2019 through 2022 for development and reserves 2023 through 2025 as an untouched holdout.
The holdout contains 2,258 games.

| Measure | Holdout result |
|---|---:|
| Pregame margin MAE | 13.284 points |
| Pregame total MAE | 13.081 points |
| In-game baseline log loss | 0.40730 |
| Closing-market anchored log loss | 0.38954 |
| In-game baseline Brier score | 0.13280 |
| Closing-market anchored Brier score | 0.12576 |

The in-game comparison uses 400,878 identical play states across the holdout.
The closing-market result shows that the pregame anchor is the main constraint on in-game accuracy.
It is not evidence that an independent model beats the market.

Two momentum variants improved development results but worsened holdout log loss.
Both were rejected, and the baseline was retained.

## Data and system boundaries

College football schedules, play-by-play, teams, talent, returning production, recruiting, transfers, coaches, and historical lines come from [CollegeFootballData](https://collegefootballdata.com/).
The Odds API is used for timestamped sportsbook snapshots near kickoff.
CFBD lines remain an audit source when timestamped market data is unavailable.

Raw and processed Parquet artifacts are stored under `backend/data/` and are not committed.
This keeps provider data, generated predictions, and model artifacts out of Git while leaving the pipeline and fixture-based tests reproducible.
A compact aggregate evaluation export and a synthetic serving fixture are available under [`examples/`](examples/).

```text
CFBD data -> raw Parquet -> possession and scoring features -> model artifacts
The Odds API -> append-only snapshots -> verified pregame closing anchors
model artifacts -> reviewed publish step -> Supabase cfb schema -> momentumweb
```

## Repository layout

```text
backend/cli.py      Argument parsing, grouped by pipeline, research, serving, market, publish
backend/commands/   One handler module per command group
backend/cfbd/       CFBD API client
backend/etl/        Source ingestion and local Parquet storage
backend/features/   Possession, scoring, in-game, and unit features
backend/model/      Pregame, in-game, calibration, and evaluation models
backend/odds/       Market capture, freshness, quota, and kickoff validation
backend/serving/    Outcome-free anchors, replay, serving, and verification
backend/grading.py  Frozen live-season grading record and metrics
backend/publish.py  Supabase publication boundary
tests/              Production-critical regression and acceptance tests
sql/                Reproducible CFB database schema
```

## Local setup

Python 3.11 or 3.12 and Poetry are required.

```bash
poetry install --no-root
```

Copy the environment template, then fill only the services used by the command you plan to run.

```bash
cp .env.example .env
```

`DATABASE_URL` points at the production Supabase project.
Database mutations are denied unless the process explicitly sets `MOMENTUMCFB_DB_WRITES=1`.

Common commands:

```bash
poetry run python -m backend ingest --seasons 2025
poetry run python -m backend features --seasons 2025
poetry run python -m backend calibrate
poetry run python -m backend preseason --season 2026 --week 1 --refresh
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy
poetry run pytest -q
```

Run `poetry run python -m backend --help` for the complete command list.
The commands fall into six groups.

| Group | Commands | Used by |
|---|---|---|
| Pipeline | `ingest`, `features`, `fit`, `weekly-update`, `calibrate`, `preseason` | Weekly and opening workflows; `fit` is a local dry run of `weekly-update` without publishing |
| Research | `ingame-baseline`, `ingame-momentum`, `ingame-momentum-recency`, `ingame-market-anchor` | Reproduce the frozen in-game evaluations reported under Validation; not part of any workflow |
| Serving | `ingame-stream`, `serving-anchors`, `serve-game`, `serve-verify` | Anchor builds run in workflows; the others are operator checks that prove streamed output equals batch output |
| Market | `live-odds`, `kickoff-check`, `kickoff-run`, `live-replay` | Kickoff capture workflow; `kickoff-check` and `live-replay` are read-only diagnostics |
| Publish | `publish`, `publish-anchors`, `fetch-anchors`, `grade`, `publish-grading` | Workflows; every database write requires `MOMENTUMCFB_DB_WRITES=1` |
| Players | `player-prior`, `ingest-players`, `player-values`, `heisman`, `publish-players` | Weekly workflow; opponent-adjusted player value above replacement and the Heisman ballot forecast behind `/cfb/heisman` |

## Production workflows

`.github/workflows/weekly-update.yml` refreshes completed-season data, rebuilds features, fits the next unstarted model week, and publishes only after readiness checks pass.
`.github/workflows/opening-forecast.yml` intentionally builds and publishes the opening forecast early, then preserves its frozen capture inputs.
`.github/workflows/kickoff-capture.yml` restores that exact forecast artifact and captures the final verified pregame market snapshot without rebuilding or republishing the model.
`.github/workflows/ci.yml` installs the locked environment, validates metadata, checks formatting and linting, type-checks production boundaries, compiles the source, and runs the test suite for every pull request and push to `main`.

Publishing is a separate, explicit boundary.
Local model and evaluation commands do not require database write access.

Player-value v4 uses exposure-weighted rush/pass efficiency and a frozen previous-season opponent prior instead of resetting opening opponents to average; the prior is regressed 0.7 of the way from zero toward last season's rating and the ridge alone, with noise estimated from each channel, decides how far observed games move each unit.
v3 blended the ridge fit a second time toward last season by games played, which left midseason opponent effects about 90 percent previous-season rating; the v4 settings were chosen on 2020 through 2024 walk-forward next-game error and checked on 2025.
Before activating a new player-value version, run `python -m backend player-prior --season 2026 --runtime-bundle /path/to/player-runtime-2026.zip` where the previous season's raw plays and existing Heisman runtime artifacts are cached.
The weekly runner requires that reviewed bundle, validates the prior before paid ingestion, and never downloads historical inputs or silently substitutes zero opponent effects.

## Known limits

- The free historical CFBD feed is not the low-latency source required for live in-game production.
- Injuries and player availability are never inferred from play-by-play or market movement. A starting quarterback reported out or doubtful in `cfb.qb_availability` (a hand-entered row with its source and report time) lowers that team's expected points by a fixed documented amount when the report precedes the forecast or decision cutoff; `qb-starters` lists teams whose last box-score starter changed so reports can be checked.
- FCS teams have less complete preseason data and carry wider uncertainty.
- The 2026 live performance sample is small early in the season and is reported with its game count rather than as a conclusion.
- The project does not automate wagers, size positions, or claim profitability.


## Database recovery and release contract

The secret-free schema snapshot in `sql/baseline/2026_10_05.sql` restores the deployed schema as inspected on October 5, 2026, without production rows.
On an empty Supabase project, install `momentumweb/sql/001_site_revalidate.sql` first, restore this baseline with `psql -X -v ON_ERROR_STOP=1`, then apply `020_decision_forecast.sql`, `021_site_revalidate.sql`, `20261012190441_moneyline_pricing_context.sql`, and `20261012224346_player_opponent_prior_meta.sql` from `sql`.
The baseline includes the earlier migrations; do not replay those on top of it.
The shared function requires pg_net and Vault, with `site_revalidate_secret` provisioned separately and matched to the site's `REVALIDATE_SECRET`.
A plain PostgreSQL test database needs the `anon`, `authenticated`, and `service_role` roles and a local callback substitute.
Production schema changes require separate approval.

New forecast publication reads only a completed, hash-verified run directory.
Late quarterback reports produce a separately recorded `decision_forecast` containing published inputs, adjusted inputs, and the reports available at decision time.
Existing recommended picks remain frozen.
Older forecasts without `market_total_weight` cannot be safely reconstructed from totals alone and must be republished from source artifacts before adjusting late totals.

## Research recommendation policy

CFB picks v11 keeps recommendations enabled and grades each recorded pick at one flat research unit.
The moneyline market reference is the median of fresh same-book home/away implied probabilities after proportional vig removal.
Each book contributes once, both sides must share the event, kickoff, capture, and update timestamp, and only the latest capture can supply the reference and selected price.
Conflicting duplicates, missing pairs, and pairs with implied probability sums outside 1.00 to 1.20 are excluded.
The moneyline pricing margin is 20 percent pure model and 80 percent market-equivalent margin, preserving the previous policy's model weight and margin-space blend.
The market-equivalent margin is the inverse CDF of the paired market probability under the existing pricing dispersion and degrees of freedom; the blended margin is then converted back to win probability.
The 20 percent weight is an explicit research policy, not a claim that historical results validated it.
A moneyline needs positive estimated EV and at least two probability-equivalent points of edge, using the existing margin dispersion and distribution to convert probability gaps into the previous policy's point scale.
These points are a selection yardstick, not a forecast margin.
Moneylines cannot be promoted to meet the volume floor.
Spreads price a separate margin with 20 percent model/history and 80 percent frozen market reference; the published point forecast is unchanged.
Frozen symmetric integer-margin weights in `backend/model/spread_pricing.json` increase football key-number and push mass, fitted on 2020-2022 development data and checked on 2023-2025 historical proxy data.
This improves historical probability scores but does not establish profitable executable edges.
Spread pricing fixes dispersion at 15.35 and Student-t degrees of freedom at 500, matching the calibration contract.
Totals retain the published 50/50 market blend and 15.93-point pricing dispersion.
The 15-pick research target remains enabled for positive-EV spreads and totals, with one unit per pick.
Policy v11 first shortlists up to the remaining target, then caps each market's exploratory additions at half that shortfall, rounded up.
Capped picks are not replaced, so a batch can finish below 15 recommendations; qualified picks are never capped.
Every recommendation records `edge_qualified` or `exploratory` in its pricing context; performance summaries separate these groups using the frozen reason, with unknown older reasons labeled `legacy`.
Spreads additionally record the pricing margin, model/history component, frozen market margin and timestamp, published margin, and calibration version.
The selected-side spread edge breaks into the 20 percent model contribution, the offered-line difference from the frozen market reference, and the price/distribution adjustment; these sum to the recorded edge.
Quote differences do not prove line movement because the reference and offer can come from different books or times.
Decision context records regular season, postseason, or unknown, along with postseason and differing-spread-quote research flags.
These flags describe the forecast context and do not disable recommendations or imply known player absences.

Every new decision stores its probability source, research status, and shared-game exposure in `pricing_context`.
Moneylines additionally store the exact reference pairs, timestamps, normalized probabilities, component margins, blend space, and model weight, separately from the selected execution price and any quarterback adjustment in `decision_forecast`.
Several picks on one game remain several one-unit picks, with correlated outcomes; `decision_batch_game_stake_units` records that batch's exposure, not the eventual full ledger exposure.
Recommendations and estimated edges are research hypotheses, not evidence of profitability.
Existing published picks and settlements remain frozen under their recorded policy.

Run `poetry run python -m examples.audit_moneyline_probabilities` to reproduce the frozen v8 historical audit from local CFBD caches.
The `current_020` scenario name in that audit means the policy at the October 12 audit cutoff, not the latest recommendation policy.
Historical closing quotes lack executable decision timestamps, and the later seasons have been inspected in earlier research, so this replay is a proxy evaluation rather than a pristine validation sample.
No 2026 outcomes are used to fit the replacement moneyline policy.
