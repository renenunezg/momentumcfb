"""Weekly Heisman forecasts preserve chronology and candidate-pool misses."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backend.players import artifacts, heisman, pipeline


def _synthetic_seasons(seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 1.0, len(heisman.FEATURES))
    frames = []
    for season in range(2010, 2018):
        design = rng.normal(0.0, 1.0, (40, len(heisman.FEATURES)))
        logits = 2.5 * (design @ truth)
        share = np.exp(logits - logits.max())
        share /= share.sum()
        frame = pd.DataFrame(design, columns=heisman.FEATURES)
        frame["season"] = season
        frame["athlete_id"] = [f"{season}-{index}" for index in range(40)]
        frame["athlete_name"] = [
            f"Player {chr(65 + index // 26)}{chr(65 + index % 26)}"
            for index in range(40)
        ]
        frame["team"] = "T"
        frame["share"] = share
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), truth


def test_predicted_shares_form_a_ballot_and_recover_the_signal(monkeypatch, tmp_path):
    rows, truth = _synthetic_seasons()
    model = heisman.fit_share_model(rows)
    rows["predicted_share"] = model.predict(rows)
    sums = rows.groupby("season")["predicted_share"].sum()
    assert np.allclose(sums, 1.0)
    assert np.corrcoef(model.beta, truth)[0, 1] > 0.9
    rows["points"] = rows["share"] * 1000
    seed = rows[["season", "athlete_name", "team", "points"]].rename(
        columns={"athlete_name": "player", "team": "school"}
    )
    snapshots = pd.concat([rows.assign(week=1), rows.assign(week=2)], ignore_index=True)
    final_season = rows["season"].max()
    winner = rows[rows["season"].eq(final_season)].sort_values("share").iloc[-1]
    snapshots = snapshots[
        ~(snapshots["athlete_id"].eq(winner["athlete_id"]) & snapshots["week"].eq(1))
    ].reset_index(drop=True)
    fit = heisman.fit_share_model
    training_cutoffs = []

    def fit_spy(train):
        training_cutoffs.append(int(train["season"].max()))
        return fit(train)

    monkeypatch.setattr(heisman, "fit_share_model", fit_spy)
    history = heisman.chronological_weekly_evaluation(snapshots, seed)
    seasons = sorted(rows["season"].unique())
    assert training_cutoffs == seasons[:-1]
    assert history.groupby("season")["week"].agg(list).tolist() == [[1, 2]] * 7
    missing = history[history["season"].eq(final_season) & history["week"].eq(1)].iloc[
        0
    ]
    assert not missing["winner_in_pool"]
    assert not missing["winner_hit"] and not missing["top_three_hit"]
    assert missing["actual_winner_predicted_share"] == 0
    assert pd.isna(missing["actual_winner_predicted_rank"])
    assert 0 < missing["ballot_share_covered"] < 1
    complete = history[history["week"].eq(2)]
    assert complete["winner_in_pool"].all()
    snapshots["predicted_share"] = model.predict(snapshots)
    assert np.allclose(
        snapshots.groupby(["season", "week"])["predicted_share"].sum(), 1
    )

    # A fresh runner can serve the same weekly evaluation without old raw data.
    monkeypatch.setattr(artifacts, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pipeline.store, "PROCESSED_DIR", tmp_path)
    as_of = datetime(2018, 9, 1, tzinfo=timezone.utc)
    artifacts.save_heisman_model(
        model,
        seasons,
        history,
        cutoff_season=2018,
        created_at=as_of,
        source_provenance={
            str(year): {"manifest_sha256": "a" * 64, "games_sha256": "b" * 64}
            for year in seasons
        },
    )
    restored, _, _ = artifacts.load_heisman_model(2018)
    assert np.allclose(restored.predict(snapshots), model.predict(snapshots))

    def no_training(*args, **kwargs):
        raise AssertionError("fresh-runner inference accessed historical raw sources")

    monkeypatch.setattr(heisman, "training_rows", no_training)
    monkeypatch.setattr(heisman, "load_seed", no_training)
    monkeypatch.setattr(
        heisman,
        "build_board",
        lambda model, season, week, player_values, as_of: pd.DataFrame(
            [{"season": season, "week": week}]
        ),
    )
    pipeline.store.write_processed(
        pd.DataFrame(), "players", "reliability", "2018.parquet"
    )
    board = pipeline.run_heisman(2018, 1, as_of)
    assert board["week"].tolist() == [1]
    served = pipeline.store.read_processed(*pipeline.HISTORY_ARTIFACT)
    assert len(served) == 7
    assert not served[served["season"].eq(final_season)]["winner_hit"].iloc[0]
    meta = pipeline.store.read_processed(*pipeline.META_ARTIFACT).iloc[0]
    assert meta["heisman_evaluation_week"] == 1
    assert meta["heisman_evaluation_kind"] == heisman.EVALUATION_KIND
    assert (
        meta["heisman_winner_hit_rate"]
        == history[history["week"].eq(1)]["winner_hit"].mean()
    )
    with pytest.raises(ValueError, match="before the forecast"):
        artifacts.save_heisman_model(
            model,
            [2018],
            history,
            cutoff_season=2018,
            created_at=as_of,
            source_provenance={
                "2018": {"manifest_sha256": "a" * 64, "games_sha256": "b" * 64}
            },
        )


def test_unmatched_ballot_row_raises(monkeypatch):
    players = pd.DataFrame(
        {
            "athlete_id": ["1"],
            "athlete_name": ["Real Player"],
            "team": ["State"],
            "games": [12.0],
        }
    )
    seed = pd.DataFrame(
        [
            {"season": 2024, "player": "Real Player", "school": "State", "points": 100},
            {
                "season": 2024,
                "player": "Nobody Here III",
                "school": "State",
                "points": 50,
            },
        ]
    )
    with pytest.raises(ValueError, match="Nobody Here"):
        heisman._match_seed(seed, players, 2024)
    matched, unchanged = heisman._match_seed(seed, players, 2024, strict=False)
    assert matched["athlete_id"].tolist() == ["1"]
    pd.testing.assert_frame_equal(unchanged, players)

    # CFBD labels both opening slates week 1; a model-week-0 board must
    # exclude the following week's stats while matching player-value cutoffs.
    games = pd.DataFrame(
        {
            "id": [1, 2],
            "season": 2026,
            "week": 1,
            "season_type": "regular",
            "home_id": 1,
            "away_id": 2,
            "home_team": "A",
            "away_team": "B",
            "home_classification": "fbs",
            "away_classification": "fbs",
            "completed": True,
            "home_points": 21,
            "away_points": 14,
            "start_date": ["2026-08-29T16:00:00Z", "2026-09-05T16:00:00Z"],
        }
    )
    box = pd.DataFrame(
        {
            "game_id": [1, 2],
            "week": 1,
            "season_type": "regular",
            "athlete_id": "1",
            "athlete_name": "Real Player",
            "team": "A",
            "category": "passing",
            "stat_name": "YDS",
            "stat": ["100", "900"],
        }
    )
    sources = {
        "teams": pd.DataFrame(
            {"school": ["A", "B"], "conference": "SEC", "classification": "fbs"}
        ),
        "roster": pd.DataFrame({"id": ["1"], "position": ["QB"]}),
        "rankings": pd.DataFrame(
            columns=["poll", "season_type", "week", "school", "rank"]
        ),
    }
    monkeypatch.setattr(heisman.store, "read_games", lambda season: games)
    monkeypatch.setattr(heisman, "read_weekly", lambda season, kind: box)
    monkeypatch.setattr(
        heisman, "read_season_source", lambda season, name: sources[name]
    )
    opening = heisman.featured_players(2026, 0)
    assert opening["games"].tolist() == [1]
    assert opening["pass_yards"].tolist() == [100]
