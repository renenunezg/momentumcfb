import numpy as np
import pandas as pd

from backend.config import FCS_LABEL

DEFAULT_WEIGHTS = {"prev": 0.55, "talent": 3.5, "returning": 6.0}
PRIOR_SD = 6.0
FCS_NET = -25.0
FCS_SD = 3.0
RETURNING_BASELINE = 0.6


def preseason_prior(
    teams,
    talent: pd.Series,
    returning: pd.Series,
    prev_final: pd.Series | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    fbs = [t for t in teams if t != FCS_LABEL]
    df = pd.DataFrame(index=pd.Index(fbs, name="team"))

    t = talent.reindex(df.index)
    df["talent_z"] = ((t - t.mean()) / t.std(ddof=0)).fillna(-1.0)
    df["ret_c"] = returning.reindex(df.index).fillna(RETURNING_BASELINE) - RETURNING_BASELINE

    net = w["talent"] * df["talent_z"] + w["returning"] * df["ret_c"]
    if prev_final is not None:
        prev = prev_final.reindex(df.index)
        blended = w["prev"] * prev + (1 - w["prev"]) * net
        net = blended.where(prev.notna(), net)

    df["net_prior"] = net - net.mean()
    df["sd"] = PRIOR_SD

    if FCS_LABEL in teams:
        df.loc[FCS_LABEL] = {
            "talent_z": np.nan,
            "ret_c": np.nan,
            "net_prior": FCS_NET,
            "sd": FCS_SD,
        }

    df["off_prior"] = df["net_prior"] / 2
    df["def_prior"] = df["net_prior"] / 2
    return df


def fit_prior_weights(rows: pd.DataFrame) -> dict:
    """rows: prev_final, talent_z, ret_c, final (observed end-of-season net)."""
    x = rows[["prev_final", "talent_z", "ret_c"]].to_numpy()
    y = rows["final"].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    # convert the additive fit into the blend form preseason_prior uses,
    # so blended priors reproduce the fitted model for teams with history
    prev = float(np.clip(coef[0], 0.0, 0.9))
    return {
        "prev": prev,
        "talent": float(coef[1] / (1.0 - prev)),
        "returning": float(coef[2] / (1.0 - prev)),
    }
