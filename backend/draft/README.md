# Draft data and publication

The production draft workspace lives at `/cfb/draft` in MomentumWeb.
This repository owns source ingestion, position normalization, historical research, and the atomic `cfb.draft_publications` snapshot.
The frontend reads only Supabase and owns mock-selection behavior and presentation.

## Weekly publication

`Refresh draft board` runs after a successful `Weekly CFB update` on main, with a separate manual dispatch for recovery.
It reads the latest already-published college player values without making additional CFBD calls.
Each run retrieves three Pro Football Mania pages, the full Scouting Grade board, and two nflverse files: current depth charts and contracts.
It validates the draft year, seven complete rounds with unique native picks, sequential overall slots, agreement with the first-round order, every team's three priorities, unique prospect identities and ranks, supported position codes, and source freshness before one atomic upsert.
Failures preserve the previous publication and fail the workflow.
The database's statement trigger invalidates the site's CFB cache.

```sh
# Validate current sources without publishing.
poetry run python -m backend.draft.publish --season 2026 --dry-run --output /tmp/draft-publication.json

# Publish deliberately; DATABASE_URL is production.
MOMENTUMCFB_DB_WRITES=1 poetry run python -m backend.draft.publish --season 2026
```

The first publication requires `--bootstrap backend/data/processed/draft/2027` from the audited research workflow below.
Subsequent runs retain the historical cohort and unmeasured college roster catalog, while replacing current scores and NFL context.
Measured college records own their current team and position so a transfer never inherits a stale college score.
The historical cohort remains explicitly dated, rather than pretending its three-season outcomes refresh weekly.
A new draft class requires a reviewed new seed and historical bootstrap.

## Position and selection contract

RCB is right cornerback; NB is nickelback or slot corner.
In a 3-4, WLB and SLB map to EDGE, while LDE and RDE map to interior defensive line.
In a 4-3, defensive ends map to EDGE and outside linebackers to LB.
These are depth-chart role mappings, not observed snap-by-snap alignments.
Nickelback, fullback, and special-teams roles are separate package options, not extra simultaneous starters.
Unknown codes or defensive packages fail publication for review.

PFM supplies provisional order, all-round pick ownership, projected compensatory selections, and editorial needs.
Scouting Grade supplies the complete prospect ranking; the October 5 board contains 323 unique prospects, including Jayden Maiava.
Notre Dame confirms Brauntae and Tae Johnson are the same player, displayed as Tae Johnson.
The October 2 ownership assessment contains 257 projected picks across seven rounds; compensatory picks remain explicitly provisional.
The publisher emits schema version 2 with each pick's round and compensatory status.
Deploy the frontend that accepts both versions 1 and 2, then apply `sql/20261007224645_draft_publication_seven_rounds.sql` before publishing version 2.
The migration preserves version 1 rows and changes only the supported payload-version check.
The dry-run output supports local review without changing the public edition.
College source positions can be broader than projected NFL roles, such as OL versus OT or IOL.
Priority numbers mean importance, not how many players are needed.
The mock rules are transparent heuristics, not a validated selection forecast.
Momentum NFL-potential scores, draft-entry probabilities, and future trade predictions are not yet modeled.

## Research bootstrap

```sh
poetry run python -m backend.draft snapshot --season 2026 --draft-years 2020 2021 2022 2023 2024 2025 2026
poetry run python -m backend.draft build --season 2026 --nfl-data ../momentumnfl/backend/data --output backend/data/processed/draft/2027
```

Snapshot reuses cached files and retrieves only missing sources.
A fresh cache directory creates a new source edition.
CFBD calls are bounded to missing draft years and preserve a 3,000-call reserve.
Build reads local evidence and never writes to production.
The audit records input paths and checksums; local paths are not included in the public payload.

Research output includes a college watchlist, a historical drafted-player cohort, current NFL context, and source audit.
The initial cohort has 1,806 selections across 2020-2026, 1,786 verified identities, and 1,016 mature three-season NFL participation outcomes.
Playing time counts regular-season offense and defense, with special teams kept separately in research.
Participation depends on opportunity and is not independent player quality.
Recent classes have incomplete windows; unresolved player matches do not receive outcomes.
The cohort includes only drafted players, so it cannot estimate probability of being drafted.
Historical college values mix model versions and must be rebuilt consistently before chronological model development.
College class-year data does not establish eligibility or declaration.
Contract year plus duration does not establish expiry, options, release costs, or future departures.

## Verification

`tests/test_draft_publication.py` protects defensive role mappings, alternative package placement, unknown-role rejection, and source-date/year failures.
The real publisher supports `--dry-run` against current sources and the actual database read path.
UI acceptance checks belong to MomentumWeb and cover unique picks, edits, persistence, roster links, all views, and responsive layout.
The earlier standalone HTML preview remains a local research artifact and is not a production serving path.
