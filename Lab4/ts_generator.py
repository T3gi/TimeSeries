from typing import Dict
import numpy as np
from scipy import stats


class SyntheticTSGenerator:
    def __init__(self, trend_degree: int = 2, cycle_period: float = 11.0):
        self.trend_degree = trend_degree
        self.cycle_period = cycle_period
        self.poly_coeffs = None
        self.harm_coeffs = None
        self.phi1 = 0.0
        self.sigma_eps = 1.0

    @staticmethod
    def _compute_acf(y: np.ndarray, nlags: int) -> np.ndarray:
        yc = y - np.mean(y)
        denom = np.dot(yc, yc)
        return np.array([1.0 if k == 0 else np.dot(yc[:-k], yc[k:]) / denom for k in range(nlags + 1)])

    def fit(self, y: np.ndarray) -> "SyntheticTSGenerator":
        t = np.arange(len(y), dtype=float)
        self.poly_coeffs = np.polyfit(t, y, self.trend_degree)
        resid_trend = y - np.polyval(self.poly_coeffs, t)

        omega = 2.0 * np.pi / self.cycle_period
        x_harm = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
        self.harm_coeffs, *_ = np.linalg.lstsq(x_harm, resid_trend, rcond=None)
        ar_resid = resid_trend - (x_harm @ self.harm_coeffs)

        # Оцінка параметрів AR(1) методом найменших квадратів
        x_lag = ar_resid[:-1]
        y_curr = ar_resid[1:]
        self.phi1 = float(np.dot(x_lag, y_curr) / np.dot(x_lag, x_lag))
        innovations = y_curr - self.phi1 * x_lag
        self.sigma_eps = float(np.std(innovations, ddof=1))
        return self

    def generate(self, n_steps: int, n_simulations: int = 500, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.arange(n_steps, dtype=float)
        trend = np.polyval(self.poly_coeffs, t)
        omega = 2.0 * np.pi / self.cycle_period
        cycle = np.column_stack([np.sin(omega * t), np.cos(omega * t)]) @ self.harm_coeffs

        simulations = np.zeros((n_simulations, n_steps))
        init_std = self.sigma_eps / np.sqrt(max(1e-6, 1.0 - self.phi1 ** 2))

        for i in range(n_simulations):
            eps = rng.normal(0.0, self.sigma_eps, n_steps)
            ar_proc = np.zeros(n_steps)
            ar_proc[0] = rng.normal(0.0, init_std)
            for step in range(1, n_steps):
                ar_proc[step] = self.phi1 * ar_proc[step - 1] + eps[step]
            simulations[i] = trend + cycle + ar_proc

        return simulations

    @classmethod
    def validate_adequacy(cls, real_y: np.ndarray, sim_matrix: np.ndarray) -> Dict[str, float]:
        rep_synth = sim_matrix[0]
        ks_stat, ks_p = stats.ks_2samp(real_y, rep_synth)
        w_dist = stats.wasserstein_distance(real_y, rep_synth)

        acf_real = cls._compute_acf(real_y, 10)
        acf_sim = cls._compute_acf(rep_synth, 10)
        acf_rmse = float(np.sqrt(np.mean((acf_real - acf_sim) ** 2)))

        low_95 = np.percentile(sim_matrix, 2.5, axis=0)
        high_95 = np.percentile(sim_matrix, 97.5, axis=0)
        coverage_95 = float(np.mean((real_y >= low_95) & (real_y <= high_95)) * 100.0)

        return {
            "real_mean": float(np.mean(real_y)),
            "ensemble_mean": float(np.mean(sim_matrix)),
            "real_std": float(np.std(real_y, ddof=1)),
            "ensemble_std": float(np.std(sim_matrix, ddof=1)),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "wasserstein_dist": float(w_dist),
            "acf_rmse": acf_rmse,
            "pi95_coverage_pct": coverage_95,
        }