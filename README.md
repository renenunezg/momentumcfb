# Momentum CFB

Momentum CFB is the college football rating, projection, and evaluation pipeline behind [renenunez.dev/cfb](https://renenunez.dev/cfb/ratings).
It builds preseason and in-season team ratings, prices game margins and totals, records research picks against sportsbook prices, grades them, and publishes everything to the `cfb` schema of the shared Momentum Supabase project.
The site lives in the separate `momentumweb` repository and reads only that schema.
The two repositories share no code.
Live performance and methodology are on the site, not in this file.

## Model approach

The in-season model estimates each team's offense, defense, and pace from chronological points-per-possession and EPA-per-possession data, plus a fitted home-field effect, and produces a joint home and away score distribution.
Ratings are fitted to a blend of points and competitive-possession EPA, which discounts garbage time.
Conference pools carry their own level, identified by crossover games, so a whole conference can move instead of every team being shrunk toward a stale level.
The published margin averages that joint fit with a points-only rating that uses every completed game, including FCS games without play-by-play.
A frozen calibration layer then applies corrections fitted on earlier seasons, such as process-stat evidence from earlier games and visitor travel distance.

The preseason prior starts from last season's power, scoring environment, and pace, and adds talent, returning production, transfers, quarterback and roster continuity, recruiting, and coaching continuity.
Missing inputs receive neutral contributions and wider uncertainty instead of guesses.
A reported starting-quarterback absence (`cfb.qb_availability`) adjusts expected points only when the report precedes the forecast or decision cutoff.

A separate market-informed margin mixes the model with a rating fitted to earlier closing spreads and the game's consensus line.
It never feeds the pure model margin.
The live win-probability model combines the score, clock, and possession with the frozen pregame forecast.

Every weight is chosen on chronological development seasons and checked on later seasons.
Live-season outcomes are never used for tuning.

## Picks

Picks are a research ledger, not betting advice.
Each market (moneyline, spread, total) is priced from the decision-time market reference moved toward the model by a fitted share, which for spreads is close to zero.
A pick needs positive estimated EV and a minimum edge (a few fixed rules, such as the rivalry road-favorite rule in `backend/rivalries.py`, relax the edge gate), stakes one flat unit, and records its full pricing context, including whether it qualified on edge or was added to reach the weekly research target.
The current policy version and its constants live in `backend/recommendations.py` and `backend/model/spread_pricing.py`.
Published picks, forecasts, and settlements stay frozen under the policy they were recorded with, and a new policy never rewrites them.

## Data

- [CollegeFootballData](https://collegefootballdata.com/): schedules, play-by-play, teams, talent, returning production, recruiting, transfers, coaches, rosters, player stats, and historical lines.
- Sportsbook prices: The Odds API or ESPN's DraftKings line, selected by `ODDS_SOURCE`.
  Every captured close is labeled with its source.
- The kickoff capture writes one frozen pregame line per game to `cfb.closing_lines`.
  Where no capture exists, grading falls back to the CFBD median closing line and records which it used.

Raw and processed Parquet lives under `backend/data/` and is not committed.

```text
CFBD -> raw Parquet -> features -> model artifacts -> reviewed publish -> Supabase cfb schema -> momentumweb
sportsbook snapshots -> frozen pregame closing lines -> grading and closing-line value
```

## Repository layout

```text
backend/cli.py          Command-line entry point (python -m backend)
backend/commands/       One handler module per command group
backend/cfbd/           CFBD API client
backend/etl/            Ingestion and local Parquet storage
backend/features/       Possession, scoring, in-game, and unit features
backend/model/          Preseason, in-season, calibration, in-game, and evaluation models
backend/odds/           Market capture, freshness, quota, and kickoff validation
backend/serving/        Outcome-free anchors, live scoring, and verification
backend/players/        Player value, position grades, NIL values, and the Heisman board
backend/playoffs/       College Football Playoff forecast
backend/draft/          NFL draft board publication (see its README)
backend/recommendations.py, backend/grading.py, backend/publish.py
                        Pick policy, grading, and the Supabase publication boundary
sql/                    Ordered cfb schema migrations and the recovery baseline
tests/                  Production-critical regression and acceptance tests
examples/, research/    Reproducible audits and research notes
```

## Local setup

Python 3.11 or 3.12 and Poetry are required.

```bash
poetry install --no-root
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `CFBD_API_KEY` | Any command that calls CFBD |
| `ODDS_API_KEY`, `ODDS_API_BOOKMAKERS`, `ODDS_API_REGIONS` | Sportsbook snapshots from The Odds API |
| `DATABASE_URL` | Reading or publishing serving data (this is production) |
| `MOMENTUMCFB_DB_WRITES` | Set to `1` only to mutate production on purpose |

Core commands:

```bash
poetry run python -m backend ingest --seasons 2025
poetry run python -m backend features --seasons 2025
poetry run python -m backend fit --season 2026
poetry run python -m backend --help
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy
poetry run pytest -q
```

`fit` is a local dry run of the weekly update that publishes nothing.
`--help` lists the rest, including the research, serving, market, and player commands.

## Production

- `weekly-update.yml`: dispatched by Supabase pg_cron Monday morning Pacific.
  It ingests the week, fits and publishes the next forecast and picks, grades completed games, updates players and the Heisman board, and refreshes the playoff forecast.
- `kickoff-capture.yml`: dispatched by pg_cron for each kickoff window.
  It captures the final pregame line from the existing forecast artifact, publishes closing lines, and re-arms itself for the next window without rebuilding the model.
- `live-win-probability.yml`: dispatched by pg_cron while games are in progress.
  It polls the CFBD scoreboard and publishes live win probability.
- `draft-refresh.yml`: runs after a successful weekly update and republishes the draft board.
- `opening-forecast.yml` and `cfbd-live-pilot.yml`: manual dispatch only.
- `ci.yml`: lint, format, type checks, and tests on every pull request and push to `main`.

Rules that must hold:

- `DATABASE_URL` is the live database, and `backend/db.py` refuses every write unless `MOMENTUMCFB_DB_WRITES=1`, including on GitHub Actions.
- Publication reads only a completed, hash-verified forecast run, and readiness checks must pass before anything is written.
- Serving code reads only outcome-free anchors and play state; final scores are available only to evaluation code.
- Production schema changes require separate approval.

## Database ownership and recovery

This repository owns the `cfb` schema, and `momentumweb` owns the shared cache callback `public.site_revalidate()`.
To rebuild on an empty Supabase project, install `momentumweb/sql/001_site_revalidate.sql` (pg_net and Vault required, with `site_revalidate_secret` provisioned separately), restore `sql/baseline/2026_10_05.sql` with `psql -X -v ON_ERROR_STOP=1`, then apply every migration in `sql/` from `020` onward in filename order.
The baseline already contains migrations `001` through `019`; do not replay them.
A plain PostgreSQL test database needs the `anon`, `authenticated`, and `service_role` roles and a local substitute for the callback.
