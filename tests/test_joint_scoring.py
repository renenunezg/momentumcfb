from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backend.etl import store
from backend.features.scoring import (
    build_scoring_games,
    build_weekly_scoring_games,
)
from backend.model.calibration import fbs_calibration_cohort
from backend.model.forecast_calibration import apply_forecast_calibration
from backend.model.joint_scoring import DEFAULT_CONFIG, fit_joint_scoring
from backend.model.market_history import market_history_margins
from backend.model.preseason import (
    score_noise_prior_from_fit,
    scoring_priors_from_ratings,
)
from backend.model.process import (
    FEATURE_COLUMNS,
    FEATURE_SCALE,
    build_process_prior,
    load_process_prior,
    process_margin_adjustments,
)
from backend.model.production_evaluation import (
    PregameSnapshot,
    replay_production_season,
)
from backend.model.weekly import (
    WeeklyForecastNotReady,
    resolve_forecast_week,
)


def _mini_season() -> pd.DataFrame:
    teams = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
    }
    matchups = [
        (1, "A", "B", 31, 17, 3.2, -1.0, False),
        (1, "C", "D", 20, 21, 0.5, 0.7, True),
        (2, "A", "C", 28, 14, 2.7, -0.6, False),
        (2, "D", "B", 24, 20, 1.1, 0.2, False),
        (3, "B", "C", 23, 20, 0.8, 0.4, False),
        (3, "D", "A", 17, 27, -0.3, 2.1, True),
    ]
    rows = []
    for game_id, matchup in enumerate(matchups, start=100):
        week, home, away, home_points, away_points, home_epa, away_epa, neutral = (
            matchup
        )
        rows.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "model_week": week,
                "home_team_id": teams[home],
                "home_team": home,
                "home_classification": "fbs",
                "away_team_id": teams[away],
                "away_team": away,
                "away_classification": "fbs",
                "neutral_site": neutral,
                "home_points": home_points,
                "away_points": away_points,
                "game_possessions": 12.0,
                "home_epa_per_possession": home_epa,
                "away_epa_per_possession": away_epa,
            }
        )
    return pd.DataFrame(rows)


def test_joint_model_is_leak_free_and_reconciles_outputs(tmp_path, monkeypatch):
    games = _mini_season()
    as_of = datetime(2026, 9, 15, tzinfo=timezone.utc)
    fitted = fit_joint_scoring(games, forecast_week=3, as_of=as_of)
    target = games[games["model_week"].eq(3)]
    ratings = fitted.ratings()
    projections = fitted.project(target)
    uncalibrated = replace(
        fitted, config=replace(fitted.config, matchup_total_calibration=False)
    ).project(target)
    for current, previous in zip(projections, uncalibrated):
        assert current.model_version == "joint_scoring_v13"
        assert previous.model_version == "joint_scoring_v13"
        assert current.home_margin == pytest.approx(previous.home_margin)
        assert current.model_total < previous.model_total
        assert current.to_record()["total_calibration_adjustment"] == pytest.approx(
            current.model_total - previous.model_total
        )
        assert current.margin_sd == previous.margin_sd
        assert current.total_sd == previous.total_sd

    changed_future = games.copy()
    changed_future.loc[
        changed_future["model_week"].eq(3),
        [
            "home_points",
            "away_points",
            "game_possessions",
            "home_epa_per_possession",
            "away_epa_per_possession",
        ],
    ] = [100, 200, 1000, 500, -500]
    refitted = fit_joint_scoring(changed_future, forecast_week=3, as_of=as_of)

    assert [rating.to_record() for rating in ratings] == [
        rating.to_record() for rating in refitted.ratings()
    ]
    assert [p.to_record() for p in projections] == [
        p.to_record()
        for p in refitted.project(changed_future[changed_future.model_week.eq(3)])
    ]
    rating_by_id = {rating.team_id: rating for rating in ratings}
    for rating in ratings:
        assert rating.power_rating == rating.offense_points + rating.defense_points
    # joint_scoring_v12: the published margin is half the joint fit's rating
    # margin and half the points-only rating margin, with the total and the
    # joint fit's margin SD (before the fixed scalar) untouched by the blend.
    unblended = replace(
        fitted,
        config=replace(fitted.config, srs_blend_weight=0.0, margin_sd_scale=1.0),
    ).project(target)
    for projection, joint in zip(projections, unblended):
        home = rating_by_id[projection.home_team_id]
        away = rating_by_id[projection.away_team_id]
        expected_margin = (
            home.power_rating - away.power_rating + projection.home_field_points
        )
        assert abs(joint.home_margin - expected_margin) < 1e-10
        assert joint.srs_home_margin is None
        assert projection.srs_home_margin is not None
        assert projection.home_margin == pytest.approx(
            0.5 * joint.home_margin + 0.5 * projection.srs_home_margin
        )
        assert projection.home_margin != pytest.approx(joint.home_margin)
        assert projection.model_total == pytest.approx(joint.model_total)
        assert projection.margin_sd == pytest.approx(0.915 * joint.margin_sd)
        assert projection.total_sd == joint.total_sd
        assert projection.model_total == (
            projection.expected_home_points + projection.expected_away_points
        )
        assert projection.margin_sd > 0
        assert projection.total_sd > 0
        assert projection.to_record()["expected_game_possessions"] == pytest.approx(12)
        assert -1 < projection.margin_total_correlation < 1

    fcs_games = games.copy()
    fcs_games.loc[fcs_games["home_team"].eq("D"), "home_classification"] = "fcs"
    fcs_games.loc[fcs_games["away_team"].eq("D"), "away_classification"] = "fcs"
    fcs_fit = fit_joint_scoring(fcs_games, forecast_week=3, as_of=as_of)
    fcs_ratings = pd.DataFrame(rating.to_record() for rating in fcs_fit.ratings())
    classifications = fcs_fit.teams.set_index("team_id")["classification"]
    fcs_ratings["classification"] = fcs_ratings["team_id"].map(classifications)
    assert set(fcs_ratings["classification"]) == {"fbs", "fcs"}
    assert len(fbs_calibration_cohort(fcs_games)) == 3
    assert (
        abs(
            fcs_ratings.loc[
                fcs_ratings["classification"].eq("fbs"), "power_rating"
            ].mean()
        )
        < 1e-10
    )

    prior_fit = fit_joint_scoring(
        games,
        forecast_week=3,
        as_of=as_of,
        strength_prior_means={1: (0.5, 0.5)},
    )
    prior_rating_by_id = {rating.team_id: rating for rating in prior_fit.ratings()}
    assert prior_rating_by_id[1].power_rating > rating_by_id[1].power_rating

    # The clamp guards the unblended joint margin; the points-only blend
    # cannot reach it because a loose prior lets four games outvote it.
    extreme_fit = fit_joint_scoring(
        games,
        forecast_week=3,
        as_of=as_of,
        config=replace(DEFAULT_CONFIG, srs_blend_weight=0.0),
        strength_prior_means={1: (3.0, 3.0)},
    )
    extreme_projections = extreme_fit.project(target)
    expected_scores = [
        score
        for projection in extreme_projections
        for score in (
            projection.expected_home_points,
            projection.expected_away_points,
        )
    ]
    assert min(expected_scores) == 0.0
    assert all(score >= 0.0 for score in expected_scores)

    # Teams awaiting their opener retain distinct preseason pace and uncertainty.
    opener = games.iloc[[-1]].copy()
    opener["game_id"] = 999
    opener[["home_team_id", "away_team_id"]] = [5, 6]
    opener[["home_team", "away_team"]] = ["E", "F"]
    preseason = pd.DataFrame(
        {
            "team_id": [1, 2, 3, 4, 5, 6],
            "offense_points": [0.0] * 6,
            "defense_points": [0.0] * 6,
            "expected_possessions": [12.0, 12.0, 12.0, 12.0, 10.0, 14.0],
            "power_rating_sd": [6.0, 6.0, 6.0, 6.0, 3.0, 9.0],
        }
    )
    with_opener = pd.concat([games, opener], ignore_index=True)
    full_prior_fit = fit_joint_scoring(
        with_opener,
        forecast_week=3,
        as_of=as_of,
        priors=scoring_priors_from_ratings(preseason),
    )
    full_ratings = {rating.team_id: rating for rating in full_prior_fit.ratings()}
    assert full_ratings[5].expected_possessions == pytest.approx(10.0)
    assert full_ratings[6].expected_possessions == pytest.approx(14.0)
    assert full_ratings[5].power_rating_sd < full_ratings[6].power_rating_sd
    wider = preseason.copy()
    wider.loc[wider["team_id"].eq(5), "power_rating_sd"] = 9.0
    wider_fit = fit_joint_scoring(
        with_opener,
        forecast_week=3,
        as_of=as_of,
        priors=scoring_priors_from_ratings(wider),
    )
    assert (
        wider_fit.project(opener)[0].margin_sd
        > full_prior_fit.project(opener)[0].margin_sd
    )

    # A small opening slate must not discard the previous season's noise state.
    noise = score_noise_prior_from_fit(
        replace(
            fitted,
            season=2025,
            as_of=datetime(2026, 1, 20, tzinfo=timezone.utc),
            training_games=1000,
            score_residual_covariance=np.eye(2) * 100,
        )
    )
    noise_path = tmp_path / "score_noise_prior.parquet"
    noise.to_parquet(noise_path, index=False)
    noise = pd.read_parquet(noise_path)
    carried = scoring_priors_from_ratings(preseason, noise)
    carried_fit = fit_joint_scoring(with_opener, 3, as_of, priors=carried)
    before = full_prior_fit.project(opener)[0]
    after = carried_fit.project(opener)[0]
    assert after.home_margin == before.home_margin
    assert after.model_total == pytest.approx(before.model_total)
    assert after.margin_sd > before.margin_sd
    # Scoring support moves totals without changing strength, spread risk,
    # or the existing matchup correction, including a clipped extreme total.
    for baseline_ppp in (0.01, carried.base_ppp + 0.5):
        stabilized = replace(carried_fit, preseason_base_ppp=baseline_ppp)
        reference = replace(
            stabilized, config=replace(stabilized.config, scoring_prior_games=0)
        ).project(opener)[0]
        projection = stabilized.project(opener)[0]
        assert projection.model_version == "joint_scoring_v13"
        assert projection.home_margin == pytest.approx(reference.home_margin)
        assert projection.margin_sd == reference.margin_sd
        assert projection.total_sd == reference.total_sd
        assert projection.total_calibration_adjustment == pytest.approx(
            reference.total_calibration_adjustment
        )
        assert projection.model_total != pytest.approx(reference.model_total)
        assert (
            min(projection.expected_home_points, projection.expected_away_points) >= 0
        )
        assert projection.scoring_baseline_adjustment == pytest.approx(
            projection.model_total - reference.model_total
        )
    np.testing.assert_allclose(
        carried_fit.score_residual_covariance,
        (
            1000 * np.eye(2) * 100
            + full_prior_fit.training_games * full_prior_fit.score_residual_covariance
        )
        / (1000 + full_prior_fit.training_games),
    )
    with pytest.raises(ValueError, match="previous season"):
        fit_joint_scoring(
            with_opener,
            3,
            as_of,
            priors=replace(carried, score_noise_season=2026),
        )
    with pytest.raises(ValueError, match="forecast cutoff"):
        fit_joint_scoring(
            with_opener,
            3,
            as_of,
            priors=replace(carried, score_noise_as_of=as_of),
        )
    one_game = with_opener.copy()
    one_game["completed"] = False
    one_game.loc[one_game.index[0], "completed"] = True
    one_game_fit = fit_joint_scoring(one_game, 3, as_of, priors=carried)
    assert np.isfinite(one_game_fit.project(opener)[0].margin_sd)
    np.testing.assert_allclose(one_game_fit.score_residual_covariance, np.eye(2) * 100)

    replay_games = games.copy()
    replay_games["completed"] = True
    replay_games["season_type"] = "regular"
    replay_games["start_date"] = pd.to_datetime(
        [f"2026-09-{week * 7:02d}T18:00:00Z" for week in games["week"]], utc=True
    )
    prior_ratings = preseason[preseason["team_id"].le(4)].copy()
    prior_ratings["model_version"] = "preseason_test"
    snapshot = PregameSnapshot(
        ratings=prior_ratings,
        projections=pd.DataFrame(
            p.to_record() for p in fitted.project(replay_games[games["week"].eq(1)])
        ),
        as_of=pd.Timestamp("2026-09-01T00:00:00Z"),
        path=Path("synthetic-pregame-snapshot"),
        digest="synthetic-test-digest",
        score_noise_prior=noise,
    )
    replay = replay_production_season(
        2026, games=replay_games, snapshots=[snapshot], engine_only=True
    )
    prediction_columns = [
        "game_id",
        "expected_home_points",
        "expected_away_points",
        "home_margin",
        "model_total",
        "margin_sd",
        "total_sd",
    ]
    for week in (2, 3):
        weekly_target = replay_games[replay_games["week"].eq(week)]
        cutoff = weekly_target["start_date"].min() - pd.Timedelta(microseconds=1)
        direct = fit_joint_scoring(
            replay_games,
            week,
            cutoff.to_pydatetime(),
            priors=scoring_priors_from_ratings(prior_ratings, noise),
        )
        expected = pd.DataFrame(p.to_record() for p in direct.project(weekly_target))
        actual = replay[replay["model_week"].eq(week)]
        assert_frame_equal(
            actual[prediction_columns].reset_index(drop=True),
            expected[prediction_columns].reset_index(drop=True),
        )
    changed_results = replay_games.copy()
    changed_results.loc[changed_results["week"].eq(3), "home_points"] = 100
    changed_replay = replay_production_season(
        2026, games=changed_results, snapshots=[snapshot], engine_only=True
    )
    assert_frame_equal(
        replay.loc[replay["model_week"].eq(3), prediction_columns],
        changed_replay.loc[changed_replay["model_week"].eq(3), prediction_columns],
    )

    # Exercise the same frozen process artifact and final calibration as a fresh
    # weekly runner, including future features already present in a replay cache.
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")
    feature_rows = []
    for game in replay_games.itertuples():
        for side in ("home", "away"):
            feature_rows.append(
                {
                    "game_id": game.game_id,
                    "season": 2026,
                    "team": getattr(game, f"{side}_team"),
                    **dict(
                        zip(
                            FEATURE_COLUMNS,
                            FEATURE_SCALE
                            * (
                                0.2 * getattr(game, f"{side}_team_id")
                                + 0.01 * game.week
                            ),
                        )
                    ),
                }
            )
    team_games = pd.DataFrame(feature_rows)
    previous_features = team_games.assign(season=2025)
    previous_schedule = replay_games.rename(columns={"game_id": "id"}).assign(
        season=2025,
        start_date=replay_games["start_date"] - pd.DateOffset(years=1),
    )
    store.write_processed(previous_features, "team_games", "2025.parquet")
    schedule_path = store.raw_path("games", 2025)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    previous_schedule.to_parquet(schedule_path, index=False)
    with pytest.raises(FileNotFoundError, match="process_prior is required"):
        load_process_prior(2026)
    process_prior = build_process_prior(2026)
    calibrated = apply_forecast_calibration(
        expected, replay_games, team_games, process_prior, 3, cutoff.to_pydatetime()
    )
    assert calibrated["model_version"].eq("joint_scoring_v14").all()
    covariance_columns = ["margin_sd", "total_sd", "margin_total_correlation"]
    assert_frame_equal(calibrated[covariance_columns], expected[covariance_columns])
    with pytest.raises(ValueError, match="unadjusted core projections"):
        apply_forecast_calibration(
            calibrated,
            replay_games,
            team_games,
            process_prior,
            3,
            cutoff.to_pydatetime(),
        )
    with pytest.raises(ValueError, match="incompatible season"):
        apply_forecast_calibration(
            expected,
            replay_games,
            team_games,
            process_prior.assign(source_season=2026),
            3,
            cutoff.to_pydatetime(),
        )
    raw = process_margin_adjustments(
        expected, replay_games, team_games, process_prior, 3, cutoff
    )
    swapped = expected.assign(
        home_team=expected["away_team"], away_team=expected["home_team"]
    )
    reversed_raw = process_margin_adjustments(
        swapped, replay_games, team_games, process_prior, 3, cutoff
    )
    np.testing.assert_allclose(raw, -reversed_raw, rtol=0, atol=1e-12)
    assert raw.abs().max() > 0
    future = replay_games.iloc[[-1]].assign(
        game_id=10000, model_week=4, start_date=cutoff + pd.Timedelta(days=7)
    )
    future_features = team_games.iloc[[-2, -1]].assign(game_id=10000)
    with_future = pd.concat([replay_games, future], ignore_index=True)
    changed_features = pd.concat([team_games, future_features], ignore_index=True)
    future_ids = with_future.loc[with_future["model_week"].ge(3), "game_id"]
    changed_features.loc[
        changed_features["game_id"].isin(future_ids), list(FEATURE_COLUMNS)
    ] = 1e6
    np.testing.assert_allclose(
        raw,
        process_margin_adjustments(
            expected, with_future, changed_features, process_prior, 3, cutoff
        ),
        rtol=0,
        atol=1e-12,
    )

    # A large process correction cannot create negative scores or change risk.
    extreme_prior = process_prior.copy()
    extreme_prior.loc[
        extreme_prior["team"].eq(expected.iloc[0]["home_team"]), FEATURE_COLUMNS[0]
    ] = -1e6
    bounded = apply_forecast_calibration(
        expected, replay_games, team_games, extreme_prior, 3, cutoff.to_pydatetime()
    )
    assert bounded[["expected_home_points", "expected_away_points"]].ge(0).all().all()
    assert bounded.iloc[0]["expected_away_points"] == 0.0
    assert bounded.iloc[0]["process_margin_raw_adjustment"] != pytest.approx(
        bounded.iloc[0]["process_margin_adjustment"]
    )
    np.testing.assert_allclose(
        bounded["expected_home_points"] + bounded["expected_away_points"],
        bounded["model_total"],
    )
    np.testing.assert_allclose(
        bounded["expected_home_points"] - bounded["expected_away_points"],
        bounded["home_margin"],
    )
    np.testing.assert_allclose(bounded["home_spread"], -bounded["home_margin"])
    assert_frame_equal(bounded[covariance_columns], expected[covariance_columns])

    production_replay = replay_production_season(
        2026,
        games=replay_games,
        snapshots=[snapshot],
        process_prior=process_prior,
        team_games=team_games,
    )
    frozen_columns = [*prediction_columns, "model_version"]
    assert_frame_equal(
        production_replay.loc[
            production_replay["model_week"].eq(1), frozen_columns
        ].reset_index(drop=True),
        snapshot.projections[frozen_columns].reset_index(drop=True),
    )
    calibrated_columns = [
        *prediction_columns,
        "model_version",
        "process_margin_raw_adjustment",
        "process_margin_adjustment",
        "median_total_adjustment",
    ]
    for week in (2, 3):
        weekly_target = replay_games[replay_games["model_week"].eq(week)]
        weekly_cutoff = weekly_target["start_date"].min() - pd.Timedelta(microseconds=1)
        weekly_fit = fit_joint_scoring(
            replay_games,
            week,
            weekly_cutoff.to_pydatetime(),
            priors=scoring_priors_from_ratings(prior_ratings, noise),
        )
        direct_core = pd.DataFrame(
            p.to_record() for p in weekly_fit.project(weekly_target)
        )
        direct_calibrated = apply_forecast_calibration(
            direct_core,
            replay_games,
            team_games,
            process_prior,
            week,
            weekly_cutoff.to_pydatetime(),
        )
        actual = production_replay[production_replay["model_week"].eq(week)]
        assert actual["process_prior_source"].eq("supplied_process_prior").all()
        assert_frame_equal(
            actual[calibrated_columns].reset_index(drop=True),
            direct_calibrated[calibrated_columns].reset_index(drop=True),
        )
    changed_production_replay = replay_production_season(
        2026,
        games=changed_results,
        snapshots=[snapshot],
        process_prior=process_prior,
        team_games=changed_features[changed_features["game_id"].ne(10000)],
    )
    assert_frame_equal(
        production_replay.loc[
            production_replay["model_week"].eq(3), prediction_columns
        ],
        changed_production_replay.loc[
            changed_production_replay["model_week"].eq(3), prediction_columns
        ],
    )
    with pytest.raises(ValueError, match="no preseason snapshot before cutoff"):
        replay_production_season(
            2026,
            games=replay_games,
            snapshots=[replace(snapshot, as_of=replay_games["start_date"].min())],
        )


def test_weekly_frame_retains_future_games_without_training_on_them():
    games = _mini_season()
    games["completed"] = games["model_week"].lt(3)
    games["season_type"] = "regular"
    games["start_date"] = pd.to_datetime(
        [f"2026-09-{week * 7:02d}T18:00:00Z" for week in games["week"]],
        utc=True,
    )
    schedule = games.rename(
        columns={
            "game_id": "id",
            "home_team_id": "home_id",
            "away_team_id": "away_id",
        }
    )
    team_game_rows = []
    for game in games[games["completed"]].itertuples():
        for team, epa in (
            (game.home_team, game.home_epa_per_possession),
            (game.away_team, game.away_epa_per_possession),
        ):
            team_game_rows.append(
                {
                    "game_id": game.game_id,
                    "team": team,
                    "offense_possessions": game.game_possessions,
                    "offense_competitive_possessions": game.game_possessions,
                    "offense_epa_total": epa * game.game_possessions,
                    "game_possessions": game.game_possessions,
                }
            )
    team_games = pd.DataFrame(team_game_rows)

    weekly = build_weekly_scoring_games(schedule, team_games)
    completed = build_scoring_games(schedule, team_games)
    target = weekly[weekly["model_week"].eq(3)]

    assert len(weekly) == 6
    assert len(completed) == 4
    assert len(target) == 2
    assert target["game_possessions"].isna().all()

    as_of = datetime(2026, 9, 15, tzinfo=timezone.utc)
    fitted = fit_joint_scoring(weekly, forecast_week=3, as_of=as_of)
    assert len(fitted.project(target)) == 2

    opening_schedule = schedule.copy()
    opening_schedule["completed"] = False
    opening_schedule.loc[opening_schedule.index[0], "completed"] = True
    opening_schedule["start_date"] = pd.to_datetime(
        [
            "2026-08-30T02:00:00Z",
            "2026-09-03T04:00:00Z",
            "2026-09-10T23:00:00Z",
            "2026-09-11T00:00:00Z",
            "2026-09-17T23:00:00Z",
            "2026-09-18T00:00:00Z",
        ],
        utc=True,
    )
    opening = build_weekly_scoring_games(opening_schedule, team_games)
    source_week_one = opening[opening["week"].eq(1)].sort_values("start_date")

    assert source_week_one["model_week"].tolist() == [0, 1]
    gap_as_of = datetime(2026, 8, 31, 18, 17, tzinfo=timezone.utc)
    assert resolve_forecast_week(opening, None, gap_as_of) == 1
    with pytest.raises(WeeklyForecastNotReady, match="has started"):
        resolve_forecast_week(
            opening,
            None,
            datetime(2026, 9, 3, 5, tzinfo=timezone.utc),
        )


def _two_division_season() -> pd.DataFrame:
    """Four FBS and four FCS teams, level within division, crossover margin 28."""
    rows = []
    game_id = 500

    def add(week, home, away, home_points, away_points):
        nonlocal game_id
        rows.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": week,
                "model_week": week,
                "home_team_id": home,
                "home_team": f"T{home}",
                "home_classification": "fbs" if home <= 4 else "fcs",
                "away_team_id": away,
                "away_team": f"T{away}",
                "away_classification": "fbs" if away <= 4 else "fcs",
                "neutral_site": True,
                "home_points": home_points,
                "away_points": away_points,
                "game_possessions": 12.0,
                "home_epa_per_possession": (home_points - 24) / 12.0,
                "away_epa_per_possession": (away_points - 24) / 12.0,
            }
        )
        game_id += 1

    for week, (home, away) in enumerate(
        [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)], 1
    ):
        add(week, home, away, 24, 24)
        add(week, home + 4, away + 4, 24, 24)
    for week, fbs in enumerate([1, 2, 3, 4], 7):
        add(week, fbs, fbs + 4, 38, 10)
    return pd.DataFrame(rows)


def test_fcs_pool_is_anchored_to_crossover_margins():
    games = _two_division_season()
    fit = fit_joint_scoring(
        games,
        forecast_week=11,
        as_of=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )
    projected = fit.project(games[games["model_week"] >= 7])
    crossover_margin = sum(
        projection.expected_home_points - projection.expected_away_points
        for projection in projected
    ) / len(projected)
    # Without pool anchoring the per-team prior holds the FCS pool near the
    # FBS level and the fitted crossover margin lands close to half the truth.
    assert crossover_margin > 26.0


def test_market_history_never_reads_the_forecast_week_lines():
    games = _mini_season()
    games["start_date"] = pd.Timestamp("2026-09-05", tz="UTC") + pd.to_timedelta(
        7 * games["model_week"], unit="D"
    )
    prior = pd.DataFrame({"team": ["A"], "market_rating": [3.0]})
    target = games[games["model_week"].eq(3)]
    as_of = target["start_date"].min()

    def lines(target_week_spread):
        spreads = {100: -10.0, 101: 2.0, 102: -7.0, 103: -1.0}
        spreads.update(dict.fromkeys(target["game_id"], target_week_spread))
        return pd.DataFrame(
            {
                "game_id": list(spreads),
                "lines": [[{"spread": value}] for value in spreads.values()],
            }
        )

    first = market_history_margins(games, lines(-30.0), 3, as_of, target, prior)
    second = market_history_margins(games, lines(30.0), 3, as_of, target, prior)
    assert list(first.index) == list(target["game_id"])
    assert np.isfinite(first).all()
    pd.testing.assert_series_equal(first, second)
