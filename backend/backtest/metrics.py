import numpy as np
import pandas as pd


def mae(df: pd.DataFrame) -> float:
    return float((df["model_margin"] - df["actual_margin"]).abs().mean())


def _edges_and_covers(df: pd.DataFrame):
    d = df.dropna(subset=["closing_spread", "model_margin", "actual_margin"])
    edge = d["model_margin"] + d["closing_spread"]
    cover = d["actual_margin"] + d["closing_spread"]
    return edge, cover


def ats_summary(df: pd.DataFrame, thresholds=(2.0, 3.0, 4.0)) -> pd.DataFrame:
    edge, cover = _edges_and_covers(df)
    rows = []
    for t in thresholds:
        mask = edge.abs() >= t
        e, c = edge[mask], cover[mask]
        decided = c != 0
        wins = int((((e > 0) & (c > 0)) | ((e < 0) & (c < 0)))[decided].sum())
        n = int(decided.sum())
        rows.append(
            {
                "threshold": t,
                "picks": int(mask.sum()),
                "wins": wins,
                "losses": n - wins,
                "win_pct": wins / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, edges=(2.0, 3.0, 4.0, 6.0)) -> pd.DataFrame:
    edge, cover = _edges_and_covers(df)
    bins = list(edges) + [np.inf]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (edge.abs() >= lo) & (edge.abs() < hi)
        e, c = edge[mask], cover[mask]
        decided = c != 0
        wins = int((((e > 0) & (c > 0)) | ((e < 0) & (c < 0)))[decided].sum())
        n = int(decided.sum())
        rows.append(
            {
                "bucket": f"[{lo}, {hi})",
                "picks": int(mask.sum()),
                "win_pct": wins / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)
