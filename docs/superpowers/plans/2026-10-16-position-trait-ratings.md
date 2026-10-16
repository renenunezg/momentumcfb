# Position Trait Ratings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rate every FBS player at their position with named, opponent-adjusted traits and one fitted headline, backtested against future game margins, served weekly beside or in place of player value v4.

**Architecture:** A new package `backend/players/traits/` reuses the v4 loaders (classified plays, unit effects, play credit, win probability added, box scores) and adds trait builders per position, a weekly snapshot with position percentiles, and a joint ridge fit of team margin on share-weighted roster traits. Development fits are an offline command that freezes a weights artifact; the weekly runner only scores. A long `cfb.position_traits` table holds the latest week per season.

**Tech Stack:** Python 3.11, pandas, numpy, scipy (`scipy.optimize`, `scipy.linalg`), pytest, SQLAlchemy to Supabase Postgres, GitHub Actions. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-16-position-trait-ratings-design.md`

## Global Constraints

- Development seasons 2020 through 2024; validation 2025; 2026 is scored and published, never fitted.
- Every trait for week w uses only games with `model_week < w` for its opponent adjustment and only games with `model_week <= w` for the player's own counts, exactly as v4 snapshots do.
- Opponent effects come from `value.season_unit_effects` (v4), never refitted here.
- The competitive flag (`is_competitive`) from `backend/features/possessions.py` gates every counted event. Garbage time counts for WPA only, as in v4.
- FCS opponents weigh `value.FCS_OPPONENT_WEIGHT` (0.5) in every count.
- Position groups are `value.POSITION_GROUPS`; WR and TE share trait definitions but keep separate percentiles.
- Weights come from a ridge with a penalty chosen by leave-one-season-out, reported with their leave-one-season-out spread. No hand-set or grid-rounded weights.
- `cfb.player_values` is untouched. Replacement is a per-position flag consumers read.
- CFBD calls: the portal pull is exactly one `/player/portal` call per season, 2020 through 2025, approved by Rene on 2026-10-16. Nothing else new hits CFBD.
- Run everything with `poetry run` from the repository root. Lint with `poetry run ruff check` before each commit.
- Model version string: `cfb_position_traits_v1`.

## Review Focus

1. A player with games but no roster row, or listed at an unmapped position, must get a position group from their dominant role (as `value._position_groups` does) and never be dropped from the snapshot. Test in Task 11.
2. A team whose only passer all season is one player has no "other passers" baseline; pressure handling must fall back to the opponent-adjusted league sack rate and flag it, not divide by zero or emit NaN. Test in Task 7.
3. A defender whose box rows are all zero in a game (dressed, no stats) counts as present for with-without; a player with no box row is absent. A season with no absences yields a NaN impact trait, not zero. Test in Task 9.
4. A week with no completed games (cancelled slate, bye) must carry the previous cumulative snapshot forward unchanged, and the headline fit must skip it. Test in Task 11 and Task 13.
5. A 2026 run at week 8 with a frozen 2024-cutoff artifact must score without ever calling the fit, and must refuse an artifact whose cutoff season is not before the scored season. Test in Task 15.

---

## File structure

Create:
- `backend/players/traits/__init__.py` — exports `MODEL_VERSION = "cfb_position_traits_v1"` and the trait registry.
- `backend/players/traits/context.py` — `SeasonContext` and `load_season_context(season)`: everything a trait builder needs for one season.
- `backend/players/traits/targets.py` — recover intended receivers from play text.
- `backend/players/traits/expectations.py` — expected rush yards, expected conversion, expected field goal tables fitted on earlier seasons.
- `backend/players/traits/adjust.py` — opponent adjustment for rate traits from prior-week opponent rates.
- `backend/players/traits/teammate.py` — teammate-relative splits.
- `backend/players/traits/absence.py` — with-without estimator.
- `backend/players/traits/offense.py` — QB, RB, WR/TE trait builders.
- `backend/players/traits/defense.py` — DL, LB, DB trait builders.
- `backend/players/traits/line.py` — OL unit and K/P traits.
- `backend/players/traits/snapshots.py` — weekly cumulative snapshots, percentiles, split-half reliability.
- `backend/players/traits/transfers.py` — portal matching and portability report.
- `backend/players/traits/headline.py` — design matrix, ridge fit, drop rule, scoring, artifact I/O.
- `backend/players/traits/backtest.py` — walk-forward evaluation, v4 comparison, swap test, report tables.
- `tests/test_position_traits.py` — all unit tests for the package.
- `sql/<timestamp>_position_traits.sql` — the serving table.
- `docs/position_traits_v1.md` — the evaluation report (Task 18).

Modify:
- `backend/players/ingest.py` — portal pull.
- `backend/players/pipeline.py` — `run_position_traits`, `read_player_artifacts`.
- `backend/publish.py` — publish the trait table.
- `backend/commands/players.py`, `backend/cli.py` — three commands.
- `.github/workflows/weekly-update.yml` — one step.
- `backend/draft/research.py` — read the headline where it replaces v4.

Shared frame contract, used by every trait builder and consumed by Tasks 11 onward. A **trait frame** is long:

| column | type | meaning |
| --- | --- | --- |
| `athlete_id` | str | CFBD athlete id; OL unit rows use `f"unit:{team}"` |
| `model_week` | int | the game's model week (per-game rows) |
| `game_id` | int | the game |
| `trait` | str | trait name, e.g. `qb.drive_engine` |
| `numerator` | float | event value for this game, already FCS-weighted and opponent-adjusted where the trait is a rate |
| `denominator` | float | exposure for this game (dropbacks, carries, targets, games = 1.0), FCS-weighted |
| `kind` | str | `counted` or `inferred` |

A trait's cumulative value through week w is `sum(numerator) / sum(denominator)` over `model_week <= w` (aggregate `rate`), except traits that declare an `aggregate` of `quantile20` (floor), `share_of_games` (every week) or `latest` (with-without impact: the row with the greatest `model_week <= w`), which Task 11 handles by name. The registry in `__init__.py` is `TRAITS: dict[str, TraitSpec]` with `TraitSpec(name, position_group, kind, aggregate, higher_is_better)`.

---

### Task 1: Portal pull for the development seasons

**Files:**
- Modify: `backend/players/ingest.py`
- Modify: `backend/commands/players.py`, `backend/cli.py:390-402`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `ingest_portal(client, season: int) -> pd.DataFrame` writing `players_dir(season) / "portal.parquet"` with columns `season, first_name, last_name, position, origin, destination, transfer_date, rating, stars, eligibility` (snake case via `backend.etl.ingest.to_snake`), and appending a manifest row (`source="portal"`, `endpoint="/player/portal"`) the same way `ingest_player_sources` records sources. Readable later with `read_season_source(season, "portal")`.
- CLI: `ingest-portal --seasons 2020 2021 2022 2023 2024 2025` with `_cfbd_budget_arguments`.

- [ ] **Step 1: Write the failing test**

```python
def test_ingest_portal_writes_one_call_per_season(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    class Client:
        calls = []
        def get(self, path, params=None, **_):
            self.calls.append((path, params))
            return [{"season": 2021, "firstName": "A", "lastName": "B", "position": "QB",
                     "origin": "Troy", "destination": "Auburn", "transferDate": None,
                     "rating": None, "stars": 3, "eligibility": "Immediate"}]
    frame = ingest.ingest_portal(Client(), 2021)
    assert Client.calls == [("/player/portal", {"year": 2021})]
    assert list(frame.columns)[:4] == ["season", "first_name", "last_name", "position"]
    assert ingest.read_season_source(2021, "portal").shape[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_ingest_portal_writes_one_call_per_season -v`
Expected: FAIL with `AttributeError: module 'backend.players.ingest' has no attribute 'ingest_portal'`

- [ ] **Step 3: Implement `ingest_portal` and the command**

Call `client.ensure_budget(1)` first. Add `handle_ingest_portal` in `backend/commands/players.py` mirroring `handle_ingest_players`, and register `ingest-portal` in `_add_pipeline_commands` beside `ingest-players`.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_ingest_portal_writes_one_call_per_season -v`
Expected: PASS

- [ ] **Step 5: Pull the six seasons**

Run: `poetry run python -m backend ingest-portal --seasons 2020 2021 2022 2023 2024 2025 --max-calls 10`
Expected: six log lines, one per season, each with a row count above 500; `backend/data/raw/players/<season>/portal.parquet` exists for each.

- [ ] **Step 6: Commit**

```bash
git add backend/players/ingest.py backend/commands/players.py backend/cli.py tests/test_position_traits.py
git commit -m "Pull portal transfers for the trait development seasons"
```

---

### Task 2: Season context

**Files:**
- Create: `backend/players/traits/__init__.py`, `backend/players/traits/context.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `@dataclass SeasonContext(season, plays, games, effects, credit, box, roster, teams, positions, presence)` where `plays` is `classify_plays(store.read_season_pbp(season))` joined with per-play `wpa` (from `value.play_wpa`) as column `wpa`; `games` is `load_weekly_games(season)`; `effects` is `value.season_unit_effects(...)`; `credit` is the finalized v4 credit frame (`value.finalize_credit` output, columns as `value.CREDIT_COLUMNS` plus `model_week, weight, plays, raw_epa, adjusted_epa, wpa`); `box` is `read_weekly(season, "box")` with `stat` numeric where it parses; `positions` is `value._position_groups(roster, credited)`; `presence` has columns `athlete_id, team, game_id, model_week, present: bool` built by `player_presence(box, credit, games)`.
- Produces: `load_season_context(season: int) -> SeasonContext` and `player_presence(box, credit, games) -> pd.DataFrame`.
- Produces in `__init__.py`: `MODEL_VERSION`, `TraitSpec`, `TRAITS` (filled by later tasks).

- [ ] **Step 1: Write the failing test**

```python
def test_player_presence_counts_zero_stat_box_rows_as_present():
    box = pd.DataFrame([
        {"game_id": 1, "team": "Away", "athlete_id": "de1", "athlete_name": "de1", "category": "defensive", "stat_name": "TOT", "stat": "0"},
        {"game_id": 2, "team": "Away", "athlete_id": "de1", "athlete_name": "de1", "category": "defensive", "stat_name": "TOT", "stat": "3"},
    ])
    games = pd.DataFrame({"game_id": [1, 2, 3], "model_week": [1, 2, 3], "home_team": "Home", "away_team": "Away", "completed": True})
    credit = pd.DataFrame(columns=["athlete_id", "team", "game_id"])
    out = context.player_presence(box, credit, games)
    de1 = out[out["athlete_id"].eq("de1")].set_index("game_id")["present"]
    assert de1.to_dict() == {1: True, 2: True, 3: False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_player_presence_counts_zero_stat_box_rows_as_present -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.players.traits'`

- [ ] **Step 3: Implement `context.py`**

`player_presence` lists every completed game of the player's team (from the season's games, team taken from the player's box or credit rows) and marks present where a box row or a credit row exists for that game. `load_season_context` reuses the exact sequence in `value.build_player_values` up to `credited`, then attaches `play_wpa` to `plays` by `id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_player_presence_counts_zero_stat_box_rows_as_present -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/ tests/test_position_traits.py
git commit -m "Add the season context the position trait builders read"
```

---

### Task 3: Recover targets from play text

**Files:**
- Create: `backend/players/traits/targets.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `recover_targets(plays, play_stats, roster) -> pd.DataFrame` with columns `play_id (str), athlete_id (str), source` where `source` is `feed` for a play-stats `Target` row and `text` for a receiver named in `play_text` matched to the offense team's roster by normalized first and last name (`heisman._normalize_name` style: casefold, strip accents, drop punctuation and suffixes Jr., Sr., II, III). Only incompletions (`play_type == "Pass Incompletion"`) without a feed target are parsed. One row per play; ambiguous roster matches (two players, same normalized name, same team) are skipped.
- Produces: `target_coverage(plays, targets) -> pd.DataFrame` with columns `season, incompletions, feed, text, uncovered`.

- [ ] **Step 1: Write the failing test**

```python
def test_recover_targets_parses_incomplete_pass_text_and_prefers_feed_rows():
    plays = pd.DataFrame([
        {"id": "p1", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "A Quarterback pass incomplete to Jon Smith Jr."},
        {"id": "p2", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "A Quarterback pass incomplete to Jon Smith Jr."},
        {"id": "p3", "season": 2024, "game_id": 1, "offense": "Home", "play_type": "Pass Incompletion", "play_text": "A Quarterback pass incomplete"},
    ])
    stats = pd.DataFrame([{"play_id": "p2", "athlete_id": "wr9", "stat_type": "Target", "team": "Home"}])
    roster = pd.DataFrame([{"id": "wr1", "first_name": "Jon", "last_name": "Smith Jr.", "team": "Home"}])
    out = targets.recover_targets(plays, stats, roster).set_index("play_id")
    assert out.loc["p1", "athlete_id"] == "wr1" and out.loc["p1", "source"] == "text"
    assert out.loc["p2", "athlete_id"] == "wr9" and out.loc["p2", "source"] == "feed"
    assert "p3" not in out.index
    coverage = targets.target_coverage(plays, out.reset_index())
    assert coverage.iloc[0][["incompletions", "feed", "text", "uncovered"]].tolist() == [3, 1, 1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_recover_targets_parses_incomplete_pass_text_and_prefers_feed_rows -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `targets.py`**

Pattern: `pass incomplete to (?P<name>[A-Z][^,;(]+?)(?: for| broken up| intercepted|,|\(|$)` applied to the text after the passer's name. Trim trailing penalty text first (everything from the word `Penalty` on).

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_recover_targets_parses_incomplete_pass_text_and_prefers_feed_rows -v`
Expected: PASS

- [ ] **Step 5: Measure real coverage**

Run (scratch script, not committed): load each season 2020 through 2025 with `load_season_context` and print `target_coverage`. Record the per-season `uncovered` share in the commit message body.
Expected: uncovered share printed for six seasons; nothing written under `backend/data`.

- [ ] **Step 6: Commit**

```bash
git add backend/players/traits/targets.py tests/test_position_traits.py
git commit -m "Recover intended receivers from play text where the feed has no target"
```

---

### Task 4: Expectation tables from earlier seasons

**Files:**
- Create: `backend/players/traits/expectations.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `@dataclass ExpectationTables(rush_yards: pd.DataFrame, conversion: pd.DataFrame, field_goal: pd.DataFrame, source_seasons: list[int])`.
- Produces: `fit_expectation_tables(plays_by_season: dict[int, pd.DataFrame], before_season: int) -> ExpectationTables` using only `season < before_season`, and raising `ValueError` if no earlier season is given.
- Produces: `bucket(plays) -> pd.DataFrame` adding `down_bucket` (1, 2, 3, 4), `distance_bucket` (`1-2`, `3-6`, `7-10`, `11+`), `field_bucket` (`own: ytg>=80`, `mid: 40-79`, `plus: 20-39`, `red: <20`), and for field goals `fg_bucket` by distance in 5-yard bins from the play text's `NN yd FG`.
- Produces: `expected_rush_yards(tables, plays) -> pd.Series`, `expected_conversion(tables, plays) -> pd.Series`, `expected_field_goal(tables, plays) -> pd.Series`, each aligned to `plays.index`, falling back to the table's global mean for an unseen bucket.
- Conversion means `is_success` on the play, with the table keyed by the three buckets.

- [ ] **Step 1: Write the failing test**

```python
def test_expectation_tables_use_only_earlier_seasons_and_fall_back_to_the_mean():
    prior = pd.DataFrame({"season": 2021, "down": [1, 1, 3], "distance": [10, 10, 2], "yards_to_goal": [75, 75, 25],
                          "is_rush": True, "is_competitive": True, "yards_gained": [2.0, 6.0, 1.0], "is_success": [False, True, True],
                          "play_type": "Rush", "play_text": ""})
    current = prior.assign(season=2022, yards_gained=99.0)
    tables = expectations.fit_expectation_tables({2021: prior, 2022: current}, before_season=2022)
    assert tables.source_seasons == [2021]
    probe = pd.DataFrame({"down": [1, 4], "distance": [10, 1], "yards_to_goal": [75, 3]})
    expected = expectations.expected_rush_yards(tables, probe)
    assert expected.iloc[0] == pytest.approx(4.0)
    assert expected.iloc[1] == pytest.approx(3.0)  # unseen bucket: global mean of 2, 6, 1
    with pytest.raises(ValueError):
        expectations.fit_expectation_tables({2022: current}, before_season=2022)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_expectation_tables_use_only_earlier_seasons_and_fall_back_to_the_mean -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `expectations.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_expectation_tables_use_only_earlier_seasons_and_fall_back_to_the_mean -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/expectations.py tests/test_position_traits.py
git commit -m "Fit rush, conversion and field goal expectations on earlier seasons"
```

---

### Task 5: Opponent adjustment for rate traits

**Files:**
- Create: `backend/players/traits/adjust.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `opponent_rates(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame` where `events` has `game_id, opponent, numerator, denominator` and the result has `model_week, opponent, allowed_rate` = the opponent's cumulative `numerator / denominator` over games with `model_week < model_week` of the row, one row per (week, opponent), NaN when the opponent has no prior exposure.
- Produces: `adjust_rate(events, games) -> pd.DataFrame` returning `events` with `numerator` replaced by `numerator - denominator * (allowed_rate - league_rate)` where `league_rate` is the league cumulative rate through the prior week, and unchanged where `allowed_rate` is NaN. This is the single adjustment used by every rate trait that is not already EPA-adjusted by v4.

- [ ] **Step 1: Write the failing test**

```python
def test_adjust_rate_subtracts_only_prior_week_opponent_deviation():
    games = pd.DataFrame({"game_id": [1, 2, 3], "model_week": [1, 2, 3]})
    events = pd.DataFrame([
        {"game_id": 1, "athlete_id": "a", "opponent": "D1", "numerator": 2.0, "denominator": 10.0},
        {"game_id": 1, "athlete_id": "b", "opponent": "D2", "numerator": 0.0, "denominator": 10.0},
        {"game_id": 2, "athlete_id": "c", "opponent": "D1", "numerator": 1.0, "denominator": 10.0},
    ])
    out = adjust.adjust_rate(events, games).set_index(["game_id", "athlete_id"])["numerator"]
    assert out.loc[(1, "a")] == 2.0  # week 1: no prior evidence, unchanged
    # week 2 vs D1: D1 allowed 0.2 through week 1, league 0.1, so subtract 10 * 0.1
    assert out.loc[(2, "c")] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_adjust_rate_subtracts_only_prior_week_opponent_deviation -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `adjust.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_adjust_rate_subtracts_only_prior_week_opponent_deviation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/adjust.py tests/test_position_traits.py
git commit -m "Adjust rate traits by the opponent's prior-week allowed rate"
```

---

### Task 6: Teammate-relative splits and the with-without estimator

**Files:**
- Create: `backend/players/traits/teammate.py`, `backend/players/traits/absence.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `relative_rate(events, player_key="athlete_id", context_keys=("game_id", "team"), numerator="numerator", denominator="denominator", min_context_denominator=5.0) -> pd.DataFrame` returning one row per (player, game) with `numerator = player_rate_in_game - others_rate_in_game` times the player's denominator, `denominator` the player's denominator, and NaN numerator where teammates in that context have less than `min_context_denominator` exposure. The context for a receiver is `("game_id", "passer_id")`; the caller passes the keys.
- Produces: `with_without(presence, unit_games, value_column, shrink_games: float) -> pd.DataFrame` with `athlete_id, present_games, absent_games, raw_difference, impact` where `raw_difference = mean(value present) - mean(value absent)` over the team's games, `impact = raw_difference * (present * absent) / (present * absent + shrink_games)`, NaN when `absent_games == 0` or `present_games < 3`. Sign is such that positive means the team allowed less when present (`value_column` is EPA allowed, so impact = absent mean − present mean).

- [ ] **Step 1: Write the failing tests**

```python
def test_relative_rate_compares_to_teammates_in_the_same_games():
    events = pd.DataFrame([
        {"game_id": 1, "team": "T", "athlete_id": "rb1", "numerator": 4.0, "denominator": 10.0},
        {"game_id": 1, "team": "T", "athlete_id": "rb2", "numerator": 1.0, "denominator": 10.0},
        {"game_id": 2, "team": "T", "athlete_id": "rb1", "numerator": 3.0, "denominator": 10.0},
    ])
    out = teammate.relative_rate(events).set_index(["athlete_id", "game_id"])
    assert out.loc[("rb1", 1), "numerator"] == pytest.approx(3.0)   # (0.4 - 0.1) * 10
    assert np.isnan(out.loc[("rb1", 2), "numerator"])                # no teammate carries

def test_with_without_needs_an_absence_and_shrinks_by_games():
    presence = pd.DataFrame({"athlete_id": "de1", "team": "T", "game_id": [1, 2, 3, 4], "present": [True, True, True, False]})
    unit = pd.DataFrame({"team": "T", "game_id": [1, 2, 3, 4], "epa_allowed": [-1.0, -1.0, -1.0, 2.0]})
    out = absence.with_without(presence, unit, "epa_allowed", shrink_games=3.0).iloc[0]
    assert out["raw_difference"] == pytest.approx(3.0)
    assert out["impact"] == pytest.approx(3.0 * 3 / (3 + 3))
    never_absent = presence.assign(present=True)
    assert np.isnan(absence.with_without(never_absent, unit, "epa_allowed", 3.0).iloc[0]["impact"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "relative_rate or with_without" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `teammate.py` and `absence.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "relative_rate or with_without" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/teammate.py backend/players/traits/absence.py tests/test_position_traits.py
git commit -m "Add teammate-relative and with-without estimators for player traits"
```

---

### Task 7: QB traits

**Files:**
- Create: `backend/players/traits/offense.py`
- Modify: `backend/players/traits/__init__.py` (register QB traits)
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Consumes: `SeasonContext`, `ExpectationTables`, `adjust.adjust_rate`, `teammate.relative_rate`, `targets.recover_targets`.
- Produces: `qb_traits(ctx: SeasonContext, tables: ExpectationTables, recovered_targets: pd.DataFrame) -> pd.DataFrame` in the trait frame contract, for players whose `position_group == "QB"`. A dropback is a credit row with role `passer` (Completion, Incompletion, Sack Taken, Interception Thrown kinds). Traits and their per-game numerator/denominator:

| trait | kind | numerator | denominator |
| --- | --- | --- | --- |
| `qb.drive_engine` | counted | Σ over late-down (down 3 or 4) or two-minute (period 2 or 4, clock ≤ 120 s) dropbacks and QB rushes of `is_success − expected_conversion`, then `adjust_rate` by opponent | those plays |
| `qb.explosive` | counted | completions gaining ≥ 20 yards, `adjust_rate` | dropbacks |
| `qb.ball_security` | counted | −Σ `wpa` on the passer's interception and fumble plays (positive is good), `adjust_rate` | dropbacks |
| `qb.floor` | counted, aggregate `quantile20` | adjusted EPA per dropback in that game (NaN under 10 dropbacks) | 1.0 |
| `qb.legs` | counted | Σ `is_success − expected_conversion` over rushes credited to a QB (sacks excluded by role), `adjust_rate` | QB rushes |
| `qb.pressure_handling` | inferred | `denominator × (baseline_sack_rate − sacks_taken_rate)` where baseline is the team's cumulative sack rate with other passers through the prior week when they have ≥ 30 dropbacks, else the league cumulative sack rate through the prior week; `adjust_rate` | dropbacks |
| `qb.receiver_independence` | inferred | Σ over the passer's targets of `adjusted EPA on the target − the receiver's adjusted EPA per target from other passers through the prior week`, counting only targets whose receiver has ≥ 10 such other-passer targets (NaN numerator for a game with none) | those targets |

Registers each in `TRAITS` with `higher_is_better=True`.

- [ ] **Step 1: Write the failing tests**

Build a fixture `qb_context()` in the test module: two games, two teams, one QB per team plus a backup on team Home with 2 dropbacks, plays with down, distance, clock, period, yards, `is_success`, `wpa`, `adjusted_epa`; a stub `ExpectationTables` whose `expected_conversion` returns a constant 0.4.

```python
def test_qb_traits_cover_every_registered_qb_trait_once_per_game():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, recovered_targets=pd.DataFrame(columns=["play_id", "athlete_id", "source"]))
    expected = {name for name, spec in TRAITS.items() if spec.position_group == "QB"}
    assert set(out["trait"]) == expected
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()

def test_qb_pressure_handling_falls_back_to_the_league_rate_with_one_passer():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, pd.DataFrame(columns=["play_id", "athlete_id", "source"]))
    away = out[out["trait"].eq("qb.pressure_handling") & out["athlete_id"].eq("qb_away")]
    assert away["numerator"].notna().all()
    assert (away["kind"] == "inferred").all()

def test_qb_floor_is_nan_under_ten_dropbacks():
    ctx, tables = qb_context()
    out = offense.qb_traits(ctx, tables, pd.DataFrame(columns=["play_id", "athlete_id", "source"]))
    backup = out[out["trait"].eq("qb.floor") & out["athlete_id"].eq("qb_backup")]
    assert backup["numerator"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "qb_" -v`
Expected: FAIL with `ImportError` on `offense`

- [ ] **Step 3: Implement `qb_traits` in `offense.py`**

Two-minute uses `plays.clock_seconds_remaining` from `classify_plays`. The explosive and legs traits derive yards from `plays.yards_gained`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "qb_" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/offense.py backend/players/traits/__init__.py tests/test_position_traits.py
git commit -m "Build the quarterback trait frame"
```

---

### Task 8: RB and WR/TE traits

**Files:**
- Modify: `backend/players/traits/offense.py`, `backend/players/traits/__init__.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `rb_traits(ctx, tables) -> pd.DataFrame` and `receiver_traits(ctx, recovered_targets) -> pd.DataFrame` (position groups `WR` and `TE`), trait frame contract.

RB (a carry is a credit row with role `rusher` and the player's group `RB`):

| trait | kind | numerator | denominator |
| --- | --- | --- | --- |
| `rb.yards_over_expected` | counted | Σ `yards_gained − expected_rush_yards`, `adjust_rate` | carries |
| `rb.stuff_avoidance` | inferred | `−relative_rate` of stuffed carries (`yards_gained ≤ 0`) with context `("game_id", "team")` | carries |
| `rb.breakaway` | counted | carries ≥ 10 yards, `adjust_rate` | carries |
| `rb.short_yardage` | counted | Σ `is_success − expected_conversion` on carries with `distance ≤ 2` and down 3 or 4, or `yards_to_goal ≤ 3`, `adjust_rate` | those carries |
| `rb.receiving` | counted | Σ adjusted EPA on targets (feed or recovered) | targets |
| `rb.ball_security` | counted | −Σ `wpa` on the player's fumble plays, `adjust_rate` | touches (carries + targets) |
| `rb.line_independence` | inferred | `relative_rate` of adjusted EPA per carry, context `("game_id", "team")` | carries |

WR/TE (a target is a feed `Target`/`Reception` row or a recovered target):

| trait | kind | numerator | denominator |
| --- | --- | --- | --- |
| `rec.target_share` | counted | player's targets | team's competitive targets in games the player appeared |
| `rec.value_per_target` | counted | Σ adjusted EPA on targets | targets |
| `rec.explosive` | counted | receptions ≥ 20 yards, `adjust_rate` | targets |
| `rec.late_down` | counted | Σ `is_success − expected_conversion` on late-down targets, `adjust_rate` | late-down targets |
| `rec.qb_independence` | inferred | `relative_rate` of adjusted EPA per target, context `("game_id", "passer_id")` | targets |

- [ ] **Step 1: Write the failing tests**

```python
def test_rb_line_independence_is_relative_to_backs_behind_the_same_line():
    ctx, tables = rb_context()  # two backs on Home in game 1: rb1 +0.5 EPA/carry, rb2 -0.5; one back in game 2
    out = offense.rb_traits(ctx, tables)
    li = out[out["trait"].eq("rb.line_independence")].set_index(["athlete_id", "game_id"])
    assert li.loc[("rb1", 1), "numerator"] / li.loc[("rb1", 1), "denominator"] == pytest.approx(1.0)
    assert np.isnan(li.loc[("rb1", 2), "numerator"])

def test_receiver_target_share_uses_recovered_targets():
    ctx, recovered = receiver_context()  # wr1: 2 feed targets; wr2: 1 recovered target; team total 3
    out = offense.receiver_traits(ctx, recovered)
    share = out[out["trait"].eq("rec.target_share")].groupby("athlete_id")[["numerator", "denominator"]].sum()
    assert share.loc["wr2"].tolist() == [1.0, 3.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "rb_ or receiver_" -v`
Expected: FAIL with `AttributeError` on `rb_traits`

- [ ] **Step 3: Implement `rb_traits` and `receiver_traits`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "rb_ or receiver_" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/offense.py backend/players/traits/__init__.py tests/test_position_traits.py
git commit -m "Build running back and receiver trait frames"
```

---

### Task 9: DL, LB and DB traits

**Files:**
- Create: `backend/players/traits/defense.py`
- Modify: `backend/players/traits/__init__.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Consumes: `ctx.box` (categories `defensive`, `interceptions`, `fumbles` with stat names `TOT, SOLO, TFL, SACKS, QB HUR, PD, TD, INT, FUM, LOST, REC`), `ctx.credit` roles `sacker` (for sack WPA), `ctx.presence`, `build_unit_games(ctx.plays)` for the team's EPA allowed per game, `absence.with_without`.
- Produces: `defense_traits(ctx) -> pd.DataFrame` for groups `DL`, `LB`, `DB`. Denominator is `games = 1.0` (FCS-weighted) unless stated. `adjust_rate` runs with the opponent being the offense faced.
- `WITH_WITHOUT_SHRINK_GAMES: float` module constant; starting value 6.0, to be set by the builder from split-half reliability on 2020 through 2024 and documented in the report.

| trait | groups | kind | numerator |
| --- | --- | --- | --- |
| `def.havoc` | DL, LB, DB | counted | `TFL + FF + PD + INT` in the game, `adjust_rate` (FF is `fumbles.FUM` forced where present, else the credit role `fumble_forcer`) |
| `def.pass_rush` | DL, LB | counted | Σ|wpa| of the player's tagged sacks + `QB HUR × median |wpa| of all tagged sacks through the prior week` |
| `def.finishing` | DL, LB | counted | `SACKS` with denominator `SACKS + QB HUR` |
| `def.run_stop` | DL, LB | counted | `TFL − SACKS` clipped at 0, `adjust_rate` |
| `def.every_week` | DL, LB, DB | counted, aggregate `share_of_games` | 1 if any of `TFL, SACKS, PD, INT, FF` > 0 else 0 |
| `def.tackle_share` | LB | counted | `TOT` with denominator the team's `TOT` in that game |
| `def.coverage` | LB | counted | `PD + INT`, `adjust_rate` |
| `def.ball_production` | DB | counted | `INT + PD`, `adjust_rate` |
| `def.run_support` | DB | counted | `SOLO`, `adjust_rate`; registered with `higher_is_better=None` so the fit decides the sign |
| `def.impact` | DL, LB, DB | inferred, aggregate `latest` | `with_without` on the team's competitive EPA allowed per play (rush + pass) in games present versus absent, recomputed through each week w over games with `model_week <= w`; emitted as one row per (player, week) with `game_id = NaN`, `model_week = w`, `numerator = impact`, `denominator = 1.0` |
| `def.pass_impact` | DB | inferred, aggregate `latest` | same on pass EPA allowed only |

- [ ] **Step 1: Write the failing tests**

```python
def test_defense_traits_treat_a_zero_stat_box_row_as_present_and_need_an_absence():
    ctx = defense_context()  # de1 present weeks 1-3 (week 3 all zeros), absent week 4; lb1 present every week
    out = defense.defense_traits(ctx)
    impact = out[out["trait"].eq("def.impact")].set_index("athlete_id")["numerator"]
    assert np.isfinite(impact.loc["de1"])
    assert np.isnan(impact.loc["lb1"])

def test_defense_every_week_counts_games_with_any_disruption():
    ctx = defense_context()
    out = defense.defense_traits(ctx)
    ew = out[out["trait"].eq("def.every_week") & out["athlete_id"].eq("de1")].sort_values("game_id")
    assert ew["numerator"].tolist() == [1.0, 1.0, 0.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "defense_" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `defense.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "defense_" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/defense.py backend/players/traits/__init__.py tests/test_position_traits.py
git commit -m "Build defensive trait frames with a with-without impact estimate"
```

---

### Task 10: OL unit and kicking traits

**Files:**
- Create: `backend/players/traits/line.py`
- Modify: `backend/players/traits/__init__.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `line_traits(ctx, tables) -> pd.DataFrame` with `athlete_id = f"unit:{team}"`, `position_group = "OL"`, kind `inferred` for every row. Decision for v1 (Claude's call): OL is published as one unit row per team; no individual lineman rows.
- Produces: `kicking_traits(ctx, tables) -> pd.DataFrame` for groups `K` and `P`, kind `counted`.

| trait | numerator | denominator |
| --- | --- | --- |
| `ol.protection` | `expected_sacks − sacks_allowed` where expected is team dropbacks × the league sack rate through the prior week (v1; a passer-specific expectation from the previous season is a follow-up noted in the module docstring), `adjust_rate` by the defense faced | team dropbacks |
| `ol.stuff_rate` | −stuffed carries, `adjust_rate` | team carries |
| `ol.short_yardage` | Σ `is_success − expected_conversion` on short-yardage carries, `adjust_rate` | those carries |
| `ol.line_yards` | Σ `_adjusted_line_yards(yards_gained)` from `backend/features/units.py`, `adjust_rate` | team carries |
| `k.field_goal` | Σ `made − expected_field_goal` | attempts |
| `p.punt_average` | Σ `AVG − league AVG through the prior week` × `NO` from box `punting` | punts |

- [ ] **Step 1: Write the failing test**

```python
def test_line_traits_emit_one_unit_row_per_team_game():
    ctx, tables = rb_context()
    out = line.line_traits(ctx, tables)
    assert out["athlete_id"].str.startswith("unit:").all()
    assert (out["kind"] == "inferred").all()
    assert not out.duplicated(["athlete_id", "game_id", "trait"]).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_line_traits_emit_one_unit_row_per_team_game -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `line.py`**

Note the v1 `ol.protection` expectation and its follow-up in the module docstring.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_line_traits_emit_one_unit_row_per_team_game -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/line.py backend/players/traits/__init__.py tests/test_position_traits.py
git commit -m "Build offensive line unit and kicking trait frames"
```

---

### Task 11: Weekly snapshots, percentiles and reliability

**Files:**
- Create: `backend/players/traits/snapshots.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Consumes: concatenated trait frames from Tasks 7 through 10, `ctx.positions`, `ctx.games`, `ctx.teams`.
- Produces: `weekly_trait_snapshots(frames: pd.DataFrame, positions, games, teams, season, as_of) -> pd.DataFrame` with columns `season, week, as_of, model_version, athlete_id, athlete_name, team, team_id, position_group, games, trait, value, percentile, kind, qualified`. One row per (player, week, trait) for every `model_week` in `games` from the first completed week to the last; a week with no completed games repeats the prior week's values. `value` follows the trait's aggregate (`rate`, `quantile20`, `share_of_games`, `latest`); `percentile` ranks `value` among qualified players at the position group that week, 0 to 100, oriented by `higher_is_better` (NaN when `None` or when the player is not qualified); `qualified` is `games >= max(1, min(value.QUALIFYING_GAMES, week // 2))`.
- Produces: `split_half_reliability(frames, positions, games_at=(4, 8, 12)) -> pd.DataFrame` with `position_group, trait, games, players, correlation`, odd/even game split with the first `games` games of each half.
- Produces: `TRAIT_SNAPSHOT_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshots_are_cumulative_and_carry_a_dead_week_forward():
    frames, positions, games, teams = snapshot_context()  # qb1: week 1 (2/10), week 2 (4/10); week 3 has no completed game
    out = snapshots.weekly_trait_snapshots(frames, positions, games, teams, 2024, AS_OF)
    qb = out[out["athlete_id"].eq("qb1") & out["trait"].eq("qb.explosive")].set_index("week")["value"]
    assert qb.loc[1] == pytest.approx(0.2) and qb.loc[2] == pytest.approx(0.3) and qb.loc[3] == pytest.approx(0.3)

def test_snapshot_percentiles_only_rank_qualified_players_and_keep_unrostered_players():
    frames, positions, games, teams = snapshot_context()
    out = snapshots.weekly_trait_snapshots(frames, positions.iloc[:0], games, teams, 2024, AS_OF)
    assert set(out["position_group"]) == {"QB"}           # group from dominant role
    week2 = out[out["week"].eq(2) & out["trait"].eq("qb.explosive")]
    assert week2.loc[~week2["qualified"], "percentile"].isna().all()

def test_split_half_reliability_reports_each_game_count():
    frames, positions, games, teams = snapshot_context()
    out = snapshots.split_half_reliability(frames, positions, games_at=(1,))
    assert set(out.columns) == {"position_group", "trait", "games", "players", "correlation"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "snapshot or split_half_reliability_reports" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `snapshots.py`**

Position group resolution mirrors `value._position_groups`: roster first, dominant role second, with `unit:` ids fixed to `OL`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "snapshot or split_half_reliability_reports" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/snapshots.py tests/test_position_traits.py
git commit -m "Build weekly position trait snapshots with percentiles and reliability"
```

---

### Task 12: Transfer matching and portability report

**Files:**
- Create: `backend/players/traits/transfers.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `match_transfers(portal, roster_to, roster_from) -> pd.DataFrame` with `athlete_id, season, origin, destination`: a portal row matches when normalized first and last name and `destination` match exactly one row of `roster_to` (the portal season's roster) and the same name and `origin` match exactly one row of `roster_from` (the previous season's roster) with the same athlete id. Rows failing either side are dropped and counted.
- Produces: `portability(snapshots_by_season: dict[int, pd.DataFrame], transfers: pd.DataFrame, final_week_only=True) -> pd.DataFrame` with `position_group, trait, stayers, movers, stay_slope, move_slope, move_slope_se, portable_share` where each slope regresses the season-t final-week `value` on the season-(t−1) final-week `value` among qualified players, separately for players who stayed and who moved, and `portable_share = move_slope / stay_slope` (NaN when `stay_slope <= 0` or movers < 20).

- [ ] **Step 1: Write the failing test**

```python
def test_match_transfers_requires_both_rosters_to_agree_on_one_player():
    portal = pd.DataFrame([{"season": 2022, "first_name": "Jon", "last_name": "Smith", "origin": "Troy", "destination": "Auburn"},
                           {"season": 2022, "first_name": "Al", "last_name": "Dupe", "origin": "Troy", "destination": "Auburn"}])
    roster_to = pd.DataFrame([{"id": "7", "first_name": "Jon", "last_name": "Smith", "team": "Auburn"},
                              {"id": "8", "first_name": "Al", "last_name": "Dupe", "team": "Auburn"},
                              {"id": "9", "first_name": "Al", "last_name": "Dupe", "team": "Auburn"}])
    roster_from = pd.DataFrame([{"id": "7", "first_name": "Jon", "last_name": "Smith", "team": "Troy"},
                                {"id": "8", "first_name": "Al", "last_name": "Dupe", "team": "Troy"}])
    out = transfers.match_transfers(portal, roster_to, roster_from)
    assert out["athlete_id"].tolist() == ["7"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_match_transfers_requires_both_rosters_to_agree_on_one_player -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `transfers.py`**

Reuse the name normalizer from Task 3 (move it to `targets.normalize_name` if it is private).

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_match_transfers_requires_both_rosters_to_agree_on_one_player -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/transfers.py backend/players/traits/targets.py tests/test_position_traits.py
git commit -m "Match portal transfers to rosters and measure trait portability"
```

---

### Task 13: Headline fit

**Files:**
- Create: `backend/players/traits/headline.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Consumes: trait snapshots (Task 11) for several seasons, v4 player values (`store.read_processed(*pipeline.player_values_artifact(season))`, column `shrunk_per_game`), `load_weekly_games(season)`.
- Produces: `design_matrix(snapshots, player_values, games, season) -> pd.DataFrame`: one row per completed FBS-versus-FBS game with `model_week >= 2`; columns `game_id, season, model_week, margin, home_field` (1 unless `neutral_site`), and one column per `(position_group, feature)` named `f"{group}:{feature}"` holding `home_aggregate − away_aggregate`, where `feature` is each registered trait name plus `epa` (v4 `shrunk_per_game`), and a team's aggregate is `Σ_p share_p × z_p` over its players at that group using the week `model_week − 1` snapshot: `share_p` is the player's `denominator`-exposure at the group through that week divided by the team total (OL unit share is 1), `z_p` is the league-wide standardized value among qualified players that week, 0 when missing.
- Produces: `fit_headline(matrix, dev_seasons: list[int]) -> HeadlineFit` with fields `weights: pd.Series` (index `group:feature`, plus `home_field`), `penalty: float` chosen from a log-spaced search of 25 values between 1e-2 and 1e3 by leave-one-season-out MAE on `dev_seasons`, `loso_sd: pd.Series` (sd of each weight across the leave-one-season-out refits), `dropped: list[str]` (features with `|weight| < 2 × loso_sd`, dropped and refit once), `dev_seasons`, `model_version`.
- Produces: `score_players(snapshot_week: pd.DataFrame, player_values_week: pd.DataFrame, fit: HeadlineFit) -> pd.DataFrame` with `athlete_id, position_group, headline, headline_percentile, uncertainty, replaces_value`; `headline = Σ_t w_{g,t} × z_t(player) − replacement_g`, where `replacement_g` is the `value.REPLACEMENT_PERCENTILE` quantile of the raw score among qualified players at the group; `uncertainty = sd_g × sqrt(1 − r_g(games))` with `r_g` the headline split-half correlation interpolated from the fit's reliability table (clipped to [0, 1], `r_g(0) = 0`); `replaces_value` from the fit's per-group flag (Task 14 sets it).
- Produces: `save_headline_fit(fit, path)` / `load_headline_fit(path) -> HeadlineFit` as JSON under `processed/players/models/position_traits_<cutoff_season>.json`, with `validate_headline_fit(fit, season)` raising `ValueError` unless `max(fit.dev_seasons) < season`.

- [ ] **Step 1: Write the failing tests**

```python
def test_design_matrix_uses_the_prior_week_snapshot_and_skips_week_one():
    snaps, values, games = matrix_context()  # 3 weeks, home QB explosive rises each week
    m = headline.design_matrix(snaps, values, games, 2024)
    assert m["model_week"].min() == 2
    row = m[m["game_id"].eq(games.loc[games["model_week"].eq(2), "game_id"].iloc[0])].iloc[0]
    assert row["QB:qb.explosive"] == pytest.approx(EXPECTED_WEEK1_Z_DIFFERENCE)

def test_fit_headline_recovers_a_planted_weight_and_drops_noise():
    m = synthetic_matrix(seed=0)  # margin = 3*QB:qb.explosive + 2*home_field + noise, plus a pure-noise column RB:rb.breakaway
    fit = headline.fit_headline(m, dev_seasons=[2020, 2021, 2022, 2023])
    assert fit.weights["QB:qb.explosive"] == pytest.approx(3.0, abs=0.3)
    assert "RB:rb.breakaway" in fit.dropped
    assert fit.penalty > 0

def test_headline_fit_refuses_to_score_its_own_seasons(tmp_path):
    fit = headline.fit_headline(synthetic_matrix(seed=1), dev_seasons=[2020, 2021, 2022, 2023])
    headline.save_headline_fit(fit, tmp_path / "fit.json")
    loaded = headline.load_headline_fit(tmp_path / "fit.json")
    headline.validate_headline_fit(loaded, 2025)
    with pytest.raises(ValueError):
        headline.validate_headline_fit(loaded, 2023)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "design_matrix or fit_headline or refuses_to_score" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `headline.py`**

Ridge by closed form on standardized columns with `scipy.linalg.solve`; the penalty does not apply to `home_field`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "design_matrix or fit_headline or refuses_to_score" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/headline.py tests/test_position_traits.py
git commit -m "Fit the position trait headline as a joint ridge on team margin"
```

---

### Task 14: Backtest command

**Files:**
- Create: `backend/players/traits/backtest.py`
- Modify: `backend/commands/players.py`, `backend/cli.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Consumes: Tasks 2 through 13.
- Produces: `build_season_traits(season) -> dict` returning `{"frames", "snapshots", "reliability", "target_coverage"}` for one season (context, expectation tables from every earlier season on disk from 2019, all builders, snapshots). Writes `processed/players/position_traits/<season>.parquet` (snapshots), `processed/players/position_traits/reliability_<season>.parquet`, `processed/players/position_traits/target_coverage_<season>.parquet`.
- Produces: `run_backtest(dev_seasons: list[int], validation_season: int) -> BacktestResult` with frames `weights` (feature, weight, loso_sd, dropped), `validation` (`variant` in `{"v4", "traits"}`, `week_bucket` in `{"2-5", "6-9", "10+", "all"}`, `games`, `mae`), `swap` (`position_group, mae_v4_only, mae_with_traits, improvement, reliability_6_v4, reliability_6_headline, replaces_value`), `reliability` (from Task 11 for each dev season plus the headline), `portability` (Task 12), `target_coverage`, and `draft_check` (`position_group, players, spearman_pick_v4, spearman_pick_headline, spearman_nfl_games_v4, spearman_nfl_games_headline`) computed from `backend/data/processed/draft/2027/historical_cohort.parquet` when it exists (final-week values of the season before each draft year, rank-correlated with overall pick and three-season NFL participation), otherwise an empty frame with a log line. This table is report-only and never feeds the fit or the replacement flags. `replaces_value` is `improvement > 0 and reliability_6_headline >= reliability_6_v4`.
- Produces: `write_backtest(result, fit, cutoff_season)` saving the fit JSON (with `replaces_value` per group and the headline reliability table inside it) to `processed/players/models/position_traits_<cutoff_season>.json` and the result frames to `processed/players/position_traits/evaluation_<cutoff_season>.parquet` (one long frame with a `table` column), and printing every table as Markdown to stdout.
- CLI: `position-traits-fit --dev-seasons 2020 2021 2022 2023 2024 --validate 2025`.

- [ ] **Step 1: Write the failing test**

```python
def test_replacement_rule_needs_both_a_better_mae_and_equal_reliability():
    swap = pd.DataFrame([
        {"position_group": "QB", "mae_v4_only": 13.0, "mae_with_traits": 12.8, "reliability_6_v4": 0.5, "reliability_6_headline": 0.55},
        {"position_group": "RB", "mae_v4_only": 13.0, "mae_with_traits": 12.8, "reliability_6_v4": 0.5, "reliability_6_headline": 0.4},
        {"position_group": "DL", "mae_v4_only": 13.0, "mae_with_traits": 13.1, "reliability_6_v4": 0.5, "reliability_6_headline": 0.9},
    ])
    out = backtest.replacement_flags(swap).set_index("position_group")["replaces_value"]
    assert out.to_dict() == {"QB": True, "RB": False, "DL": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_replacement_rule_needs_both_a_better_mae_and_equal_reliability -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `backtest.py` and the command**

The swap test fits, per position group, a ridge on the validation-free dev matrix with every other group represented by `epa` only and the tested group by its full trait set, then scores the validation season; `mae_v4_only` comes from the all-`epa` fit. Reliability at 6 games uses `split_half_reliability(..., games_at=(6,))` on the dev seasons pooled, for `epa` (v4 `adjusted_per_game`) and for the headline score.

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_replacement_rule_needs_both_a_better_mae_and_equal_reliability -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/traits/backtest.py backend/commands/players.py backend/cli.py tests/test_position_traits.py
git commit -m "Add the position trait backtest and headline fit command"
```

---

### Task 15: Weekly runner and scoring

**Files:**
- Modify: `backend/players/pipeline.py`, `backend/commands/players.py`, `backend/cli.py`
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Produces: `pipeline.run_position_traits(season: int, as_of: datetime) -> pd.DataFrame`: loads the latest `position_traits_<cutoff>.json` with `cutoff < season` (error if none), validates it, calls `backtest.build_season_traits(season)`, scores every week with `headline.score_players`, and writes `processed/players/position_traits/<season>.parquet` with the snapshot columns plus headline rows (`trait = "headline"`, `value` = headline points per game, `percentile`, `kind = "fitted"`, and columns `uncertainty`, `replaces_value`, `in_headline` where `in_headline` is true for trait rows whose feature was not dropped at that group).
- Produces: `pipeline.position_traits_artifact(season) -> tuple[str, ...]`.
- CLI: `position-traits --seasons <season>`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_position_traits_scores_with_a_frozen_fit_and_never_refits(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    headline.save_headline_fit(FROZEN_FIT_2024, tmp_path / "players" / "models" / "position_traits_2024.json")
    monkeypatch.setattr(pipeline.backtest, "build_season_traits", lambda season: FAKE_SEASON_2026)
    monkeypatch.setattr(pipeline.headline, "fit_headline", lambda *a, **k: pytest.fail("refit"))
    monkeypatch.setattr(pipeline.store, "read_processed", lambda *parts, **k: FAKE_V4_2026)
    out = pipeline.run_position_traits(2026, AS_OF)
    assert (out["model_version"] == "cfb_position_traits_v1").all()
    assert set(out.loc[out["trait"].eq("headline"), "kind"]) == {"fitted"}
    assert out["uncertainty"].notna().where(out["trait"].eq("headline"), True).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_run_position_traits_scores_with_a_frozen_fit_and_never_refits -v`
Expected: FAIL with `AttributeError: module 'backend.players.pipeline' has no attribute 'run_position_traits'`

- [ ] **Step 3: Implement the runner and command**

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_position_traits.py::test_run_position_traits_scores_with_a_frozen_fit_and_never_refits -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/players/pipeline.py backend/commands/players.py backend/cli.py tests/test_position_traits.py
git commit -m "Score weekly position traits from a frozen headline fit"
```

---

### Task 16: Serving table and publisher

**Files:**
- Create: `sql/<YYYYMMDDHHMMSS>_position_traits.sql` (timestamp from `date -u +%Y%m%d%H%M%S`)
- Modify: `backend/publish.py:815-870`, `backend/players/pipeline.py` (`read_player_artifacts`)
- Test: `tests/test_position_traits.py`

**Interfaces:**
- Table `cfb.position_traits`: `season int, week int, as_of timestamptz, model_version text, athlete_id text, athlete_name text, team text, team_id int, position_group text, games double precision, trait text, value double precision, percentile double precision, kind text, in_headline boolean, uncertainty double precision, replaces_value boolean`, primary key `(season, athlete_id, trait)`, index on `(season, position_group, trait)`. Grants and row-level security copied from `sql/022_draft_publications.sql`. No `site_revalidate` trigger.
- `publish_players` publishes `position_traits` by season, keeping only the latest `week` per season (the parquet keeps the history), and tolerates a missing artifact by skipping the table with a log line so the Heisman publish never blocks on it.
- `read_player_artifacts` returns the new frame under key `position_traits` filtered to its latest week, or an empty frame.

- [ ] **Step 1: Write the failing test**

```python
def test_player_artifacts_publish_only_the_latest_trait_week(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    write_fake_player_artifacts(tmp_path, season=2026)      # helper: v4 values, boards, history, meta
    pipeline.store.write_processed(TWO_WEEK_TRAITS, *pipeline.position_traits_artifact(2026))
    out = pipeline.read_player_artifacts(2026)["position_traits"]
    assert out["week"].unique().tolist() == [TWO_WEEK_TRAITS["week"].max()]

def test_migration_declares_the_trait_table_contract():
    sql = next(Path("sql").glob("*_position_traits.sql")).read_text()
    for column in ("replaces_value boolean", "in_headline boolean", "uncertainty double precision"):
        assert column in sql
    assert "primary key (season, athlete_id, trait)" in sql.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_position_traits.py -k "latest_trait_week or migration_declares" -v`
Expected: FAIL (`KeyError: 'position_traits'`, `StopIteration`)

- [ ] **Step 3: Write the migration and the publisher change**

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_position_traits.py -k "latest_trait_week or migration_declares" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sql/*_position_traits.sql backend/publish.py backend/players/pipeline.py tests/test_position_traits.py
git commit -m "Publish the latest position trait snapshot to cfb.position_traits"
```

---

### Task 17: Workflow step and draft board consumer

**Files:**
- Modify: `.github/workflows/weekly-update.yml:273-290`
- Modify: `backend/draft/research.py:53-133`
- Test: `tests/test_draft_publication.py` or `tests/test_position_traits.py`

**Interfaces:**
- Workflow: add `poetry run python -m backend position-traits --seasons "$SEASON"` between `player-values` and `heisman`; `processed/players/models/` is already in the preserved cache path, so the fit artifact travels with it. The step must not fail the job when no fit artifact exists yet: wrap it as `|| echo "position traits skipped: no fit artifact"` only until Task 18 installs the artifact, then remove the guard in Task 18.
- Draft: `watchlist(roster, catalog, values, as_of, traits=None)`; when `traits` is given, for players whose group has `replaces_value`, `value_above_replacement` and `position_rank` come from the headline row and `model_version` becomes the trait model version; other players keep v4. `build` reads `processed/players/position_traits/<season>.parquet` when present.

- [ ] **Step 1: Write the failing test**

```python
def test_watchlist_uses_the_headline_where_it_replaces_v4():
    roster, catalog, values = draft_fixture()      # qb1 (QB) and rb1 (RB) with v4 values 5.0 and 4.0
    traits = pd.DataFrame([
        {"athlete_id": "qb1", "week": 8, "as_of": values["as_of"].iloc[0], "trait": "headline", "value": 9.0, "percentile": 99.0,
         "position_group": "QB", "replaces_value": True, "model_version": "cfb_position_traits_v1"},
        {"athlete_id": "rb1", "week": 8, "as_of": values["as_of"].iloc[0], "trait": "headline", "value": 1.0, "percentile": 10.0,
         "position_group": "RB", "replaces_value": False, "model_version": "cfb_position_traits_v1"},
    ])
    board, _ = research.watchlist(roster, catalog, values, AS_OF, traits=traits)
    by_id = board.set_index("athlete_id")
    assert by_id.loc["qb1", "value_above_replacement"] == 9.0 and by_id.loc["qb1", "model_version"] == "cfb_position_traits_v1"
    assert by_id.loc["rb1", "value_above_replacement"] == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_position_traits.py::test_watchlist_uses_the_headline_where_it_replaces_v4 -v`
Expected: FAIL with `TypeError: watchlist() got an unexpected keyword argument 'traits'`

- [ ] **Step 3: Implement the workflow step and the watchlist change**

- [ ] **Step 4: Run the test and the existing draft tests**

Run: `poetry run pytest tests/test_position_traits.py::test_watchlist_uses_the_headline_where_it_replaces_v4 tests/test_draft_publication.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/weekly-update.yml backend/draft/research.py tests/test_position_traits.py
git commit -m "Run position traits weekly and let the draft board read the headline"
```

---

### Task 18: Rebuild v4 history, run the backtest, write the report, verify 2026

This task runs the real pipeline. It has no new unit tests; each step is a verification with an expected observation. Do not tune anything on 2026.

**Files:**
- Create: `docs/position_traits_v1.md`
- Modify: `.github/workflows/weekly-update.yml` (remove the Task 17 guard), `docs/superpowers/specs/2026-10-16-position-trait-ratings-design.md` (status line only)

- [ ] **Step 1: Rebuild player value v4 for the development and validation seasons**

Run: `poetry run python -m backend player-values --seasons 2020 2021 2022 2023 2024 2025`
Expected: six log lines `player values <season>: N players through week W`; `backend/data/processed/players/player_values/<season>.parquet` now carries `model_version == "cfb_player_value_v4"` for each (check with a one-line pandas read).

- [ ] **Step 2: Run the backtest**

Run: `poetry run python -m backend position-traits-fit --dev-seasons 2020 2021 2022 2023 2024 --validate 2025 2>&1 | tee output/position-traits-fit-$(date -u +%Y%m%d).log`
Expected: Markdown tables for weights, validation MAE by week bucket (both variants), the swap table with `replaces_value`, reliability at 4, 8 and 12 games per trait, portability, and target coverage; `backend/data/processed/players/models/position_traits_2024.json` exists.

- [ ] **Step 3: Sanity-read the result before writing anything**

Check, and note in the report if any fail: the `traits` variant's validation MAE is finite for every bucket; no weight exceeds 3 × its leave-one-season-out sd without a trait name you can explain; `qb.explosive` and `rb.yards_over_expected` have reliability at 8 games above 0.3; the portal match rate per season is above 60 percent of portal rows naming an FBS destination. Anything surprising goes in the report as a finding, not a fix.

- [ ] **Step 4: Write `docs/position_traits_v1.md`**

Pattern it on `docs/player_value_v4.md`: what the model is, how the weights were chosen (the search, the penalty, the drop rule), the per-position weight table with leave-one-season-out spread, the 2025 validation table versus v4, the swap table and which positions replace v4, the reliability tables, the portability table and what it says about the inferred traits, target coverage by season, the with-without shrink value and how it was chosen, and the limits (no tackles, no snaps, OL as a unit, 2019 unadjusted). Numbers come from Step 2's log; copy them, do not retype from memory.

- [ ] **Step 5: Score 2026 end to end**

Run: `poetry run python -m backend position-traits --seasons 2026`
Expected: `backend/data/processed/players/position_traits/2026.parquet` with weeks 1 through the current week; headline rows for every group; a quick read shows the top five QB headlines are recognizable 2026 starters.

- [ ] **Step 6: Dry-run the publisher against the artifact**

Run: `poetry run python -c "from backend.players.pipeline import read_player_artifacts; f = read_player_artifacts(2026)['position_traits']; print(f['week'].unique(), len(f))"`
Expected: one week printed; row count under 40,000.

- [ ] **Step 7: Remove the workflow guard and run the full test file and lint**

Run: `poetry run pytest tests/test_position_traits.py tests/test_player_values.py tests/test_heisman.py tests/test_draft_publication.py -q && poetry run ruff check`
Expected: all pass, no lint findings.

- [ ] **Step 8: Commit**

```bash
git add docs/position_traits_v1.md .github/workflows/weekly-update.yml docs/superpowers/specs/2026-10-16-position-trait-ratings-design.md
git commit -m "Report the position trait backtest and enable the weekly step"
```

Applying the migration to Supabase and the first production publish are not in this plan: they need Rene's approval at the time, in that order (push, then migration, then the weekly run), following the project's rollout convention.
