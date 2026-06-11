import pandas as pd

from backend.backtest.metrics import ats_summary, calibration_table, mae


def picks_frame():
    return pd.DataFrame(
        {
            "model_margin": [10.0, -1.0, 3.0, 7.0],
            "closing_spread": [-7.0, -2.0, -3.0, -7.0],
            "actual_margin": [10.0, 1.0, 3.0, 7.0],
        }
    )


def test_mae():
    df = pd.DataFrame({"model_margin": [3.0, -2.0], "actual_margin": [7.0, -2.0]})
    assert mae(df) == 2.0


def test_ats_summary_counts_wins_and_pushes():
    out = ats_summary(picks_frame(), thresholds=(2.0,))
    row = out.iloc[0]
    # edges: +3 (win: cover +3), -3 (win: cover -1), 0 (no pick), 0 (no pick)
    assert row.picks == 2
    assert row.wins == 2
    assert row.losses == 0


def test_push_excluded():
    df = pd.DataFrame(
        {"model_margin": [10.0], "closing_spread": [-7.0], "actual_margin": [7.0]}
    )
    out = ats_summary(df, thresholds=(2.0,))
    assert out.iloc[0].picks == 1
    assert out.iloc[0].wins == 0
    assert out.iloc[0].losses == 0


def test_calibration_buckets():
    out = calibration_table(picks_frame(), edges=(2.0, 3.0))
    assert "bucket" in out.columns
