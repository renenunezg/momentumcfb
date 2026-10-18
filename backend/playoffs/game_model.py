"""Sample from the existing calibrated Momentum predictive distribution."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t

from backend.model.distributions import marginal_scale
from backend.model.forecast_calibration import apply_forecast_calibration


@dataclass
class ForecastState:
    """Frozen cutoff fit with the production predictive distribution intact."""

    year: int
    cutoff: pd.Timestamp
    known: pd.DataFrame
    fit: object
    team_games: pd.DataFrame
    process_prior: pd.DataFrame

    def project(self, schedule):
        raw = pd.DataFrame(p.to_record() for p in self.fit.project(schedule))
        return apply_forecast_calibration(
            raw,
            self.known,
            self.team_games,
            self.process_prior,
            2,
            self.cutoff.to_pydatetime(),
        )

    def draw_state(self, rng, n):
        """Shared uncertainty replaces the parameter part of predictive variance."""
        values, vectors = np.linalg.eigh(self.fit.parameter_covariance)
        if values.min() < -1e-8:
            raise ValueError("Parameter covariance is not positive semidefinite")
        root = vectors * np.sqrt(np.maximum(values, 0))
        parameter = rng.normal(size=(n, len(values))) @ root.T
        df = self.fit.config.student_t_degrees_of_freedom
        multiplier = np.sqrt((df - 2) / rng.chisquare(df, size=n))
        return parameter, multiplier

    def sample(self, projections, rng, state, *, per_path=False, independent=False):
        fit = self.fit
        parameter, multiplier = state
        index = fit.team_index
        h = projections.home_team_id.map(index).to_numpy(int)
        a = projections.away_team_id.map(index).to_numpy(int)
        neutral = projections.neutral_site.to_numpy(bool)
        k = len(fit.teams)
        scale = fit.config.score_covariance_scale * fit.config.margin_sd_scale
        residual = fit.score_residual_covariance
        residual_sd = np.sqrt(residual[0, 0] + residual[1, 1] - 2 * residual[0, 1])
        if independent:
            shape = len(h) if per_path else (len(parameter), len(h))
            return projections.home_margin.to_numpy() + rng.standard_t(
                fit.config.student_t_degrees_of_freedom, size=shape
            ) * marginal_scale(projections.margin_sd, projections.degrees_of_freedom)
        if per_path:
            ix = np.arange(len(parameter))
            latent = (
                parameter[ix, h]
                - parameter[ix, a]
                + parameter[ix, k + h]
                - parameter[ix, k + a]
            )
            latent += (~neutral) * parameter[:, -1]
            noise = (
                fit.base_possessions * latent + rng.normal(size=len(h)) * residual_sd
            ) * scale
            return projections.home_margin.to_numpy() + noise * multiplier
        latent = (
            parameter[:, h]
            - parameter[:, a]
            + parameter[:, k + h]
            - parameter[:, k + a]
        )
        latent += (~neutral)[None, :] * parameter[:, -1, None]
        noise = (
            fit.base_possessions * latent + rng.normal(size=latent.shape) * residual_sd
        ) * scale
        return projections.home_margin.to_numpy()[None, :] + noise * multiplier[:, None]

    def validate_marginal_variance(self, projections):
        fit = self.fit
        k = len(fit.teams)
        design = np.zeros((len(projections), 2 * k + 1))
        for j, row in enumerate(projections.itertuples()):
            h, a = fit.team_index[row.home_team_id], fit.team_index[row.away_team_id]
            design[j, [h, a, k + h, k + a]] = [1, -1, 1, -1]
            design[j, -1] = not row.neutral_site
        design *= fit.base_possessions
        residual = fit.score_residual_covariance
        noise_var = residual[0, 0] + residual[1, 1] - 2 * residual[0, 1]
        variance = noise_var + np.einsum(
            "ij,jk,ik->i", design, fit.parameter_covariance, design
        )
        variance *= (
            fit.config.score_covariance_scale * fit.config.margin_sd_scale
        ) ** 2
        np.testing.assert_allclose(
            variance, projections.margin_sd.to_numpy() ** 2, rtol=1e-10
        )


def probability(projections):
    return t.cdf(
        projections.home_margin.to_numpy()
        / marginal_scale(projections.margin_sd, projections.degrees_of_freedom),
        projections.degrees_of_freedom.to_numpy(),
    )
