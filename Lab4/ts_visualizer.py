from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class TimeSeriesVisualizer:
    def __init__(self, style_font: str = "DejaVu Sans"):
        plt.rcParams["font.sans-serif"] = style_font
        plt.rcParams["axes.unicode_minus"] = False

    @staticmethod
    def _format_xticks(ax: plt.Axes, labels: np.ndarray, max_ticks: int = 10) -> None:
        n = len(labels)
        if n <= max_ticks:
            pos = np.arange(n)
        else:
            step = max(1, n // max_ticks)
            pos = np.arange(0, n, step)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels[pos], rotation=25 if isinstance(labels[0], str) else 0)

    def plot_decomposition(
        self,
        ts: pd.Series,
        trend: np.ndarray,
        seasonal: np.ndarray,
        resid: np.ndarray,
        poly_trend: Optional[np.ndarray] = None,
        title: str = "Декомпозиція часового ряду",
        ylabel: str = "Значення",
    ) -> plt.Figure:
        """Графік 1: 4-панельна декомпозиція (Ряд, Тренд, Цикл/Сезонність, Залишки)."""
        x = np.arange(len(ts))
        labels = ts.index.values
        y = ts.values

        fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)

        axes[0].plot(x, y, color="#1f77b4", lw=1.4, label="Реальний ряд")
        if poly_trend is not None:
            axes[0].plot(x, poly_trend, color="#d62728", ls="--", lw=1.5, label="Поліноміальний тренд")
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(title, fontweight="bold")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        axes[1].plot(x, trend, color="#d62728", lw=1.8)
        axes[1].set_ylabel("Тренд")
        axes[1].grid(alpha=0.3)

        axes[2].plot(x, seasonal, color="#2ca02c", lw=1.2)
        axes[2].set_ylabel("Цикл / Сезон")
        axes[2].grid(alpha=0.3)

        axes[3].bar(x, resid, color="#7f7f7f", width=0.7, alpha=0.85)
        axes[3].axhline(0, color="black", lw=0.8)
        axes[3].set_ylabel("Залишки")
        axes[3].set_xlabel("Період")
        axes[3].grid(alpha=0.3)

        self._format_xticks(axes[3], labels)
        plt.tight_layout()
        return fig

    def plot_clusters_and_correlations(
        self,
        ts: pd.Series,
        clusters: np.ndarray,
        acf_raw: np.ndarray,
        pacf_raw: np.ndarray,
        acf_detrended: Optional[np.ndarray] = None,
        pacf_detrended: Optional[np.ndarray] = None,
        cluster_names: Optional[List[str]] = None,
        ylabel: str = "Значення",
    ) -> plt.Figure:
        """Графік 2: 3-панельний аналіз кластерів режимів, ACF та PACF."""
        x = np.arange(len(ts))
        labels = ts.index.values
        y = ts.values
        n_clusters = len(np.unique(clusters))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        palette = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981"]

        if cluster_names is None:
            cluster_names = [f"Кластер {c}" for c in range(n_clusters)]

        for c in range(n_clusters):
            mask = clusters == c
            axes[0].scatter(
                x[mask], y[mask], color=palette[c % len(palette)], label=cluster_names[c], s=32, zorder=3
            )
        axes[0].plot(x, y, color="gray", alpha=0.35, lw=1)
        axes[0].set_title(f"Кластеризація режимів (K-Means, k={n_clusters})", fontweight="bold", fontsize=10)
        axes[0].set_xlabel("Період")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8, loc="best")
        axes[0].grid(alpha=0.3)
        self._format_xticks(axes[0], labels, max_ticks=7)

        lags = np.arange(len(acf_raw))
        conf = 1.96 / np.sqrt(len(y))

        axes[1].bar(lags, acf_raw, width=0.42, color="#2563eb", label="Вихідний ряд")
        if acf_detrended is not None:
            axes[1].plot(
                np.arange(len(acf_detrended)),
                acf_detrended,
                "o--",
                color="#dc2626",
                markersize=4,
                label="Після детрендингу",
            )
        axes[1].axhline(conf, ls=":", color="black")
        axes[1].axhline(-conf, ls=":", color="black")
        axes[1].set_title("Автокореляційна функція (ACF)", fontweight="bold", fontsize=10)
        axes[1].set_xlabel("Лаг")
        axes[1].set_ylabel("ACF")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        axes[2].bar(lags, pacf_raw, width=0.42, color="#059669", label="Вихідний ряд")
        if pacf_detrended is not None:
            axes[2].plot(
                np.arange(len(pacf_detrended)),
                pacf_detrended,
                "o--",
                color="#d97706",
                markersize=4,
                label="Після детрендингу",
            )
        axes[2].axhline(conf, ls=":", color="black")
        axes[2].axhline(-conf, ls=":", color="black")
        axes[2].set_ylim(-0.7, 1.1)
        axes[2].set_title("Часткова автокореляція (PACF)", fontweight="bold", fontsize=10)
        axes[2].set_xlabel("Лаг")
        axes[2].set_ylabel("PACF")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_synthetic_validation(
        self,
        ts: pd.Series,
        sim_matrix: np.ndarray,
        metrics: Dict[str, float],
        ylabel: str = "Значення",
    ) -> plt.Figure:
        """Графік 3: 2-панельна валідація синтетики (Монте-Карло + гістограма KDE)."""
        x = np.arange(len(ts))
        labels = ts.index.values
        y = ts.values
        y_synth = sim_matrix[0]

        low_95 = np.percentile(sim_matrix, 2.5, axis=0)
        high_95 = np.percentile(sim_matrix, 97.5, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})

        axes[0].fill_between(
            x, low_95, high_95, color="#93c5fd", alpha=0.45, label=f"95% довірчий інтервал ({len(sim_matrix)} симуляцій)"
        )
        axes[0].plot(x, y, color="#111827", lw=1.6, label="Реальні дані")
        axes[0].plot(x, y_synth, color="#dc2626", lw=1.25, alpha=0.88, label="Синтетична реалізація")
        axes[0].set_title(
            f"Порівняння реального та модельного рядів (покриття 95% PI = {metrics['pi95_coverage_pct']:.1f}%)",
            fontweight="bold",
            fontsize=10,
        )
        axes[0].set_xlabel("Період")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8.5, loc="best")
        axes[0].grid(alpha=0.3)
        self._format_xticks(axes[0], labels)

        axes[1].hist(y, bins=14, density=True, alpha=0.5, color="#111827", label="Реальні дані")
        axes[1].hist(y_synth, bins=14, density=True, alpha=0.5, color="#dc2626", label="Синтетичний ряд")
        kde_r = stats.gaussian_kde(y)
        kde_s = stats.gaussian_kde(y_synth)
        grid_x = np.linspace(min(y.min(), y_synth.min()) * 0.95, max(y.max(), y_synth.max()) * 1.05, 200)
        axes[1].plot(grid_x, kde_r(grid_x), color="#111827", lw=2)
        axes[1].plot(grid_x, kde_s(grid_x), color="#dc2626", lw=2, ls="--")
        axes[1].set_title(
            f"Щільність розподілу (KS p={metrics['ks_pvalue']:.2f}, W={metrics['wasserstein_dist']:.2f})",
            fontweight="bold",
            fontsize=10,
        )
        axes[1].set_xlabel(ylabel)
        axes[1].set_ylabel("Щільність")
        axes[1].legend(fontsize=8.5)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_summary_dashboard(
        self,
        ts: pd.Series,
        trend: np.ndarray,
        clusters: np.ndarray,
        acf_raw: np.ndarray,
        pacf_raw: np.ndarray,
        sim_matrix: np.ndarray,
        metrics: Dict[str, float],
        dataset_title: str = "Часовий ряд",
        ylabel: str = "Значення",
        cluster_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Графік 4: Зведений 4-панельний дашборд (Тренд + Кластери + ACF/PACF + Синтетика)."""
        x = np.arange(len(ts))
        labels = ts.index.values
        y = ts.values
        y_synth = sim_matrix[0]
        n_clusters = len(np.unique(clusters))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

        # 1. Динаміка та тренд
        axes[0, 0].plot(x, y, color="#111827", lw=1.6, label="Реальні дані")
        axes[0, 0].plot(x, trend, color="#dc2626", lw=2, ls="--", label="Тренд декомпозиції")
        axes[0, 0].set_title(f"1. Динаміка та тренд: {dataset_title}", fontweight="bold", fontsize=10)
        axes[0, 0].set_ylabel(ylabel)
        axes[0, 0].legend(fontsize=8.5)
        axes[0, 0].grid(alpha=0.3)
        self._format_xticks(axes[0, 0], labels)

        # 2. Кластеризація
        palette = ["#dc2626", "#f59e0b", "#16a34a", "#2563eb"]
        if cluster_names is None:
            cluster_names = [f"Кластер {c}" for c in range(n_clusters)]
        for c in range(n_clusters):
            m = clusters == c
            axes[0, 1].scatter(x[m], y[m], color=palette[c % len(palette)], label=cluster_names[c], s=38, zorder=3)
        axes[0, 1].plot(x, y, color="gray", alpha=0.35, lw=1)
        axes[0, 1].set_title(f"2. Кластеризація режимів (K-Means, k={n_clusters})", fontweight="bold", fontsize=10)
        axes[0, 1].set_ylabel(ylabel)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)
        self._format_xticks(axes[0, 1], labels)

        # 3. ACF та PACF
        lags = np.arange(len(acf_raw))
        conf = 1.96 / np.sqrt(len(y))
        axes[1, 0].bar(lags - 0.15, acf_raw, width=0.3, color="#2563eb", label="ACF")
        axes[1, 0].bar(lags + 0.15, pacf_raw, width=0.3, color="#059669", label="PACF")
        axes[1, 0].axhline(conf, ls=":", color="black")
        axes[1, 0].axhline(-conf, ls=":", color="black")
        axes[1, 0].set_ylim(-0.7, 1.1)
        axes[1, 0].set_title("3. Корелограма ACF та PACF", fontweight="bold", fontsize=10)
        axes[1, 0].set_xlabel("Лаг")
        axes[1, 0].set_ylabel("Кореляція")
        axes[1, 0].legend(fontsize=8.5)
        axes[1, 0].grid(alpha=0.3)

        # 4. Валідація синтетичної моделі
        low_95 = np.percentile(sim_matrix, 2.5, axis=0)
        high_95 = np.percentile(sim_matrix, 97.5, axis=0)
        axes[1, 1].fill_between(x, low_95, high_95, color="#93c5fd", alpha=0.45, label="95% довірчий інтервал")
        axes[1, 1].plot(x, y, color="#111827", lw=1.6, label="Реальні дані")
        axes[1, 1].plot(x, y_synth, color="#dc2626", lw=1.3, ls="--", label="Синтетична модель")
        axes[1, 1].set_title(
            f"4. Валідація моделі (KS p={metrics['ks_pvalue']:.2f}, 95% PI={metrics['pi95_coverage_pct']:.1f}%)",
            fontweight="bold",
            fontsize=10,
        )
        axes[1, 1].set_ylabel(ylabel)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.3)
        self._format_xticks(axes[1, 1], labels)

        plt.tight_layout()
        return fig