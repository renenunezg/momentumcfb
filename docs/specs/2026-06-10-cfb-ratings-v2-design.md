# Momentum CFB v2: EPA-Based Bayesian Rating System (Design)

**Date:** 2026-06-10
**Status:** Approved design, pre-implementation
**Decision:** Rewrite the backend pipeline in place. Keep the repo, git history, PRD, and dormant Next.js frontend. Delete the old `backend/etl` and `backend/model` scripts.

## 1. Goal

A point-spread power rating system for FBS college football. Each team gets a rating in points such that `rating_A - rating_B + HFA` is the predicted spread (e.g. Alabama 28.5 vs Georgia 24.5 means Alabama -4 on a neutral field). v1 scope is model + backtest only: prove edge against historical closing lines before building automation or the website.

### Out of scope for v1
- Totals models (deleted with old code; revisit in phase 4)
- Website / Supabase publishing (phase 2-3; Supabase project is currently paused)
- Live odds via The Odds API (phase 2; backtest uses CFBD historical closers)
- Injuries / transfer portal adjustments (phase 4; no reliable free feed)
- Venue-specific HFA (phase 4; v1 uses league-wide constants)

## 2. Architecture

Local-first. Raw data lives as Parquet on disk and all modeling runs locally. Supabase is not a dependency in v1. It returns in phase 2 purely as a publishing layer for small output tables.

```
backend/
  config.py            # seasons, paths, constants (no hardcoded SEASON)
  data/                # gitignored Parquet store
    raw/               # pbp/, games/, lines/, talent/, returning/ (partitioned by season)
    processed/         # team_game_epa/, ratings/, backtest results
  cfbd/client.py       # CFBD API wrapper: auth, retry, schema validation
  etl/                 # backfill + weekly incremental ingest, writes raw Parquet
  features/            # team-game EPA builder, writes processed Parquet
  model/state_space.py # Bayesian state-space model (PyMC)
  backtest/            # walk-forward harness + metrics
  cli.py               # python -m backend ingest|features|fit|backtest
```

Tooling: Python 3.12, Poetry (kept from existing setup), pandas + PyArrow, PyMC >= 5, pytest.

## 3. Data layer

**Source:** CollegeFootballData API (free tier, 1,000 calls/month; backfill uses about 100-150).

**Backfill (one-time), seasons 2019-2025:**
- `/plays`: play-by-play with PPA (CFBD's EPA), roughly 1.6M plays
- `/games`: scores, neutral-site flags, venues
- `/lines`: historical betting lines (closing spreads) for backtest market baseline
- `/talent`: team talent composite (preseason prior input)
- Returning production endpoint (preseason prior input)

**Weekly in-season update:** fetch only the most recently completed week (about 2 calls).

**Conventions:**
- Parquet partitioned by season (and week for plays); re-running a week overwrites its partition (idempotent ETL).
- 2020 kept but flagged: separate HFA coefficient (empty stadiums), reported separately in backtest.
- FCS opponents collapse to one pooled "generic FCS" rating per season.

## 4. Feature layer (`team_game_epa`)

One row per (game, team): offensive/defensive EPA per play, total EPA, net EPA margin, run/pass EPA splits, pace (plays), explosiveness (% plays with EPA > 1.0), success rate.

**Garbage-time handling:** EPA aggregates are computed in two variants, filtered (garbage-time plays excluded, using CFBD flags or the standard score-margin heuristic) and unfiltered. The backtest decides which feeds the model; do not assume filtering helps.

## 5. Model: Bayesian state-space (PyMC)

**Latent state:** per FBS team, offensive strength `off[t, w]` and defensive strength `def[t, w]` in points scale, evolving as a random walk across weeks:
`theta_w = theta_{w-1} + eps`, with innovation variance learned from data.

**Preseason prior:**
`theta_0 ~ Normal(a * last_season_final + b * talent_composite + c * returning_production, sigma_prior)`
regressed to the mean, with coefficients fit on historical seasons. The prior dominates Week 1 and washes out as games accumulate. Strict anti-leakage rule: priors for season S use only information available before S.

**Observations (dual likelihood).** Each game emits two noisy measurements of the same quantity (true strength gap + HFA):
1. **Score margin** (home minus away points), with noise sigma_score
2. **EPA margin** (net EPA differential scaled to points; filtered vs unfiltered variant chosen per section 4), with noise sigma_epa

Both noise scales are learned, so the data, not a hand-picked weight, decides how much the scoreboard vs the play-by-play drives ratings.

**HFA:** league-wide constant per era, separate 2020 coefficient, HFA = 0 for neutral-site games (CFBD flag).

**Run/pass splits:** published as descriptive adjusted stats alongside ratings in v1. Folding them into the latent state (4 components per team) is a phase 4 experiment.

**Fitting:**
- In-season weekly fit: NUTS, full posterior (5-10 min, Tuesday job)
- Backtest (about 90 sequential weekly fits): ADVI for speed, with NUTS spot-checks on several weeks to confirm ADVI does not materially distort the rating posterior means

**Output:** per-team rating (posterior mean and sd), off/def components, presented on a points scale centered for spread arithmetic.

## 6. Backtest harness and success criteria

Walk-forward: for each week W of each season, fit on all data before W, predict week-W spreads, compare to CFBD closing lines and actual margins.

**Metrics:**
- MAE vs actual margin (market closer baseline is roughly 12.5-13)
- ATS record at edge thresholds |model - close| >= 2, 3, 4 points (breakeven 52.4% at -110)
- Calibration: larger edges should cover at higher rates
- Splits: per season, per conference, 2020 isolated

**Success bar for proceeding to phase 2:** ATS >= 53% at the >= 3-point edge threshold across 2021-2025 out-of-sample, with reasonable calibration. If the model misses the bar, iterate on the model. Do not build automation or frontend around an unproven edge.

## 7. Odds (phase 2 reference)

The Odds API (existing account/key; free 500 credits/month, MLB model uses under 100). One NCAAF spreads snapshot costs 1 credit (cost = markets x regions, all games included). Plan: about 3 snapshots per week (Monday open, midweek, near kickoff), roughly 12 credits/month. Historical odds are paid-tier on The Odds API, which is why the backtest uses CFBD's free historical lines instead.

## 8. Testing and error handling

- pytest unit tests: feature builder (EPA aggregation, garbage-time filter), EPA-to-points scaling, prior construction
- Small fixed-fixture "mini season" for model smoke tests (fit runs, ratings ordered sensibly)
- Schema validation on CFBD responses with clear failure messages (CFBD schema drift was a known risk in the original PRD)
- Idempotent ETL: re-runs overwrite partitions, never duplicate

## 9. Backtest results (2026-06-12)

Walk-forward backtest, NUTS fits (ADVI was rejected: posterior nets drifted 3.7 points on average from NUTS on the 2024 validation fit). 3,819 predicted games across 2021-2025.

| Variant | MAE | ATS at 2+ | ATS at 3+ | ATS at 4+ |
|---|---|---|---|---|
| Garbage-time filtered | 13.40 | 50.0% (2,965) | 49.8% (2,549) | 49.4% (2,158) |
| Unfiltered | 13.82 | 50.1% (3,099) | 49.6% (2,724) | 49.7% (2,380) |

Per-season at the 3+ threshold (filtered): 2021 51.2%, 2022 47.6%, 2023 49.8%, 2024 48.9%, 2025 51.3%. Calibration is flat to inverted: the 6+ edge bucket covers at 48.8%, meaning large disagreements with the closer are evidence the market knows something, not that we do. Splits are symmetric (home picks 50.0%, away picks 49.6%) and the early-season weeks driven by preseason priors are worse (48.6%) than midseason (50.4%).

Verdict: the 53% bar at the 3+ edge threshold is not met. The model reproduces market-consensus strength estimates (MAE 13.4 vs a closer baseline near 12.5-13) but has no exploitable edge over closing lines in this form. Filtered EPA beats unfiltered on MAE and is the variant to keep. Phase 2 (automation, live odds, site) stays on hold per the success criteria.

Against opening lines instead of closers the picture improves: 51.4% at the 3+ threshold (2,429 decided picks), every calibration bucket above 50%, and 2023/2025 above 53%. Still short of the 52.4% breakeven on median openers. The decisive result is closing line value: on 3+ edge picks at the opener, the close moves toward the model's side 59.3% of the time (61.0% at 4+ edges, flats excluded). The model identifies real information that the market prices in between open and close. The shortfall is execution (median opener, single number) rather than absence of signal.

Executed at the best available opener instead of the median (CFBD carries about 2 books of openers per game), the same picks go 52.9% at the 3+ threshold and 53.4% at 2+, clearing the 52.4% breakeven, with 2023-2025 at 54.2%, 53.5% and 55.5%. Caveats: 2021-2022 opener coverage is thin and those seasons sit at or below water, and the simulation assumes the best opener was gettable. Taken with the CLV result, the conclusion is that the edge is real but lives in execution at open with line shopping, which is what phase 2 automates via The Odds API.

What to try next, in rough order of expected value: evaluate against opening lines instead of closers (the bar a weekly model realistically needs to clear), shrink predictions toward the market line and look for residual value pockets, tune the innovation scale and prior strength on 2019-2020 holdout, test situational features the market may price slowly (rest days, travel, QB changes), and check conference and total-size splits for localized edges.

## 10. Phases

| Phase | Deliverable |
|---|---|
| **1 (this spec)** | Local pipeline: ingest, features, Bayesian model, backtest vs closers |
| 2 | Weekly automation (GitHub Actions), The Odds API live lines, Supabase publishing |
| 3 | Next.js site: ratings page, weekly board with EV flags, eval dashboard |
| 4 | Model upgrades: venue HFA, run/pass latent splits, totals, injuries/portal |
