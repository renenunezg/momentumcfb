# Momentum CFB

Momentum CFB is a college football rating, projection, and model-evaluation pipeline.
It builds preseason and in-season team ratings, prices game margins and totals, captures sportsbook closing lines, and publishes reviewed outputs to the shared Momentum website.

The public interface is maintained in the separate `momentumweb` repository.
Current ratings, methodology, and performance are available at [renenunez.dev/cfb](https://renenunez.dev/cfb/ratings).

## Current status

The 2026 preseason pipeline is ready for opening week.
The latest pre-kickoff run contains 266 Division I team ratings, 172 game projections, unit ratings, market comparisons, and outcome-free serving anchors.
The 2026 season has not started, so there is not yet a live-season performance sample.
Historical walk-forward results are kept separate from the live 2026 record.

## Model design

The in-season model estimates offense, defense, pace, and home-field effect from chronological points-per-possession and EPA-per-possession data.
It produces a joint home and away score distribution with calibrated margin and total uncertainty.

The preseason model starts from prior-season power, scoring environment, and pace.
It then incorporates current CFBD talent, returning production, transfers, quarterback continuity, recruiting, and coaching continuity.
Missing inputs receive neutral contributions and increase uncertainty instead of being guessed.

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
backend/cfbd/       CFBD API client
backend/etl/        Source ingestion and local Parquet storage
backend/features/   Possession, scoring, in-game, and unit features
backend/model/      Pregame, in-game, calibration, and evaluation models
backend/odds/       Market capture, freshness, quota, and kickoff validation
backend/serving/    Outcome-free anchors, replay, serving, and verification
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

The repository is released under the [MIT License](LICENSE).

## Production workflows

`.github/workflows/weekly-update.yml` refreshes completed-season data, rebuilds features, fits the next unstarted model week, and publishes only after readiness checks pass.
`.github/workflows/opening-forecast.yml` intentionally builds and publishes the opening forecast early, then preserves its frozen capture inputs.
`.github/workflows/kickoff-capture.yml` restores that exact forecast artifact and captures the final verified pregame market snapshot without rebuilding or republishing the model.
`.github/workflows/ci.yml` installs the locked environment, validates metadata, checks formatting and linting, type-checks production boundaries, compiles the source, and runs the test suite for every pull request and push to `main`.

Publishing is a separate, explicit boundary.
Local model and evaluation commands do not require database write access.

## Known limits

- The free historical CFBD feed is not the low-latency source required for live in-game production.
- Injuries and player availability are not inferred from play-by-play or market movement.
- FCS teams have less complete preseason data and carry wider uncertainty.
- The 2026 live performance section will remain empty until completed games can be graded against frozen pregame predictions.
- The project does not automate wagers, size positions, or claim profitability.
