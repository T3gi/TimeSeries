from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@dataclass
class TSAnalysisReport:
    descriptive: Dict[str, float]
    stationarity: Dict[str, float]
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray
    trend_strength: float
    seasonal_strength: float
    acf_raw: np.ndarray
    pacf_raw: np.ndarray
    pacf_detrended: np.ndarray
    acf_detrended: np.ndarray
    ljung_box: pd.DataFrame
    clusters: np.ndarray
    silhouette: float


class TimeSeriesAnalyzer:
    def __init__(self, series: pd.Series, cycle_period: int = 11):
        self.ts = series.dropna()
        self.y = self.ts.values.astype(float)
        self.t = np.arange(len(self.y), dtype=float)
        self.cycle_period = cycle_period

    @staticmethod
    def compute_hurst(y: np.ndarray, max_lag: int = 30) -> float:
        lags = range(2, min(max_lag, len(y) // 3))
        tau = [np.std(np.subtract(y[lag:], y[:-lag])) for lag in lags]
        return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])

    @staticmethod
    def compute_acf(y: np.ndarray, nlags: int) -> np.ndarray:
        y_centered = y - np.mean(y)
        denom = np.dot(y_centered, y_centered)
        return np.array([
            1.0 if k == 0 else np.dot(y_centered[:-k], y_centered[k:]) / denom
            for k in range(nlags + 1)
        ])

    @classmethod
    def compute_pacf(cls, y: np.ndarray, nlags: int) -> np.ndarray:
        r = cls.compute_acf(y, nlags)
        pacf_vals = np.zeros(nlags + 1)
        pacf_vals[0] = 1.0
        for k in range(1, nlags + 1):
            R = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
            phi = np.linalg.solve(R, r[1 : k + 1])
            pacf_vals[k] = phi[-1]
        return pacf_vals

    @classmethod
    def ljung_box_test(cls, y: np.ndarray, lags: list) -> pd.DataFrame:
        n = len(y)
        r = cls.compute_acf(y, max(lags))
        rows = []
        for h in lags:
            q_stat = n * (n + 2) * sum((r[k] ** 2) / (n - k) for k in range(1, h + 1))
            p_val = float(1.0 - stats.chi2.cdf(q_stat, df=h))
            rows.append({"lag": h, "lb_stat": q_stat, "lb_pvalue": p_val})
        return pd.DataFrame(rows).set_index("lag")

    @staticmethod
    def adf_test_simple(y: np.ndarray) -> Dict[str, float]:
        dy = np.diff(y)
        y_lag = y[:-1]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
        resid = dy - X @ beta
        s2 = np.sum(resid ** 2) / (len(dy) - 2)
        se_gamma = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        t_stat = float(beta[1] / se_gamma)
        return {"adf_stat": t_stat, "is_stationary_5pct": float(t_stat < -2.89)}

    def decompose_additive(self):
        p = self.cycle_period
        trend = pd.Series(self.y).rolling(window=p, center=True, min_periods=1).mean().values
        detrended = self.y - trend
        seasonal = np.zeros_like(self.y)
        for i in range(p):
            idx = np.arange(i, len(self.y), p)
            seasonal[idx] = np.mean(detrended[idx])
        seasonal -= np.mean(seasonal)
        resid = self.y - trend - seasonal
        return trend, seasonal, resid

    def run_full_analysis(self, n_clusters: int = 3, nlags: int = 15) -> TSAnalysisReport:
        desc = {
            "count": len(self.y),
            "mean": float(np.mean(self.y)),
            "median": float(np.median(self.y)),
            "std": float(np.std(self.y, ddof=1)),
            "min": float(np.min(self.y)),
            "max": float(np.max(self.y)),
            "iqr": float(np.percentile(self.y, 75) - np.percentile(self.y, 25)),
            "skew": float(stats.skew(self.y)),
            "kurtosis": float(stats.kurtosis(self.y)),
            "hurst": self.compute_hurst(self.y),
        }

        stat_tests = self.adf_test_simple(self.y)
        trend, seasonal, resid = self.decompose_additive()
        f_t = max(0.0, 1.0 - np.var(resid) / np.var(trend + resid))
        f_s = max(0.0, 1.0 - np.var(resid) / np.var(seasonal + resid))

        poly2 = np.polyfit(self.t, self.y, 2)
        detrended = self.y - np.polyval(poly2, self.t)
        acf_raw = self.compute_acf(self.y, nlags)
        pacf_raw = self.compute_pacf(self.y, nlags)
        acf_det = self.compute_acf(detrended, nlags)
        pacf_det = self.compute_pacf(detrended, nlags)
        lb = self.ljung_box_test(self.y, [1, 5, 10, nlags])

        features = pd.DataFrame({
            "val": self.ts,
            "roll_mean": self.ts.rolling(5, min_periods=1).mean(),
            "roll_std": self.ts.rolling(5, min_periods=1).std().fillna(0.0),
        })
        x_scaled = StandardScaler().fit_transform(features)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        raw_labels = km.fit_predict(x_scaled)
        order = np.argsort([self.y[raw_labels == c].mean() for c in range(n_clusters)])
        mapping = {old: new for new, old in enumerate(order)}
        clusters = np.array([mapping[c] for c in raw_labels])
        sil = float(silhouette_score(x_scaled, clusters))

        return TSAnalysisReport(
            descriptive=desc,
            stationarity=stat_tests,
            trend=trend,
            seasonal=seasonal,
            resid=resid,
            trend_strength=float(f_t),
            seasonal_strength=float(f_s),
            acf_raw=acf_raw,
            pacf_raw=pacf_raw,
            pacf_detrended=pacf_det,
            acf_detrended=acf_det,
            ljung_box=lb,
            clusters=clusters,
            silhouette=sil,
        )