from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# Вспомогательные функции

def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    logger.info("График сохранён: %s", path)


def _get_series(df: pd.DataFrame) -> pd.Series:
    """Возвращает pd.Series с DatetimeIndex из DataFrame формата [unique_id, ds, y]."""
    s = df.set_index("ds")["y"].sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


# 1. Базовые визуализации

def plot_series_overview(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Временной ряд + скользящие среднее и стд.

    Args:
        df: DataFrame с колонками [ds, y].
        save: Сохранить ли PNG.

    Returns:
        Figure matplotlib.
    """
    s = _get_series(df)
    window = min(12, len(s) // 4)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(s.index, s.values, color="#2C7BB6", linewidth=1.5, label="Исходный ряд")
    axes[0].set_title("Месячный пассажиропоток", fontsize=13, fontweight="bold")
    axes[0].legend()

    rolling_mean = s.rolling(window=window).mean()
    rolling_std = s.rolling(window=window).std()

    axes[1].plot(s.index, s.values, color="#AAAAAA", linewidth=1, alpha=0.6)
    axes[1].plot(rolling_mean.index, rolling_mean.values,
                 color="#D7191C", linewidth=2, label=f"Rolling Mean (w={window})")
    axes[1].legend()
    axes[1].set_title("Скользящее среднее")

    axes[2].plot(rolling_std.index, rolling_std.values,
                 color="#1A9641", linewidth=1.5, label=f"Rolling Std (w={window})")
    axes[2].legend()
    axes[2].set_title("Скользящее стандартное отклонение")

    for ax in axes:
        ax.set_ylabel("Значение")
    axes[2].set_xlabel("Дата")

    plt.tight_layout()
    if save:
        _save(fig, "01_series_overview")
    return fig


def plot_boxplot_by_month(df: pd.DataFrame, save: bool = True) -> plt.Figure:

    s = _get_series(df)
    frame = pd.DataFrame({"value": s.values, "month": s.index.month})
    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                   "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=frame, x="month", y="value", ax=ax, palette="Blues")
    ax.set_xticklabels(month_names[:frame["month"].max()])
    ax.set_title("Распределение по месяцам (сезонный паттерн)", fontweight="bold")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Пассажиропоток")
    plt.tight_layout()
    if save:
        _save(fig, "02_boxplot_month")
    return fig


def plot_seasonal_plot(df: pd.DataFrame, save: bool = True) -> plt.Figure:

    s = _get_series(df)
    frame = pd.DataFrame({
        "value": s.values,
        "month": s.index.month,
        "year": s.index.year,
    })
    years = sorted(frame["year"].unique())
    palette = sns.color_palette("tab10", len(years))

    fig, ax = plt.subplots(figsize=(12, 5))
    for color, year in zip(palette, years):
        subset = frame[frame["year"] == year].sort_values("month")
        ax.plot(subset["month"], subset["value"], marker="o",
                color=color, label=str(year), linewidth=1.5)

    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                   "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Seasonal Plot (по годам)", fontweight="bold")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Пассажиропоток")
    ax.legend(title="Год", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save:
        _save(fig, "03_seasonal_plot")
    return fig


def plot_lag_plot(df: pd.DataFrame, lags: list[int] | None = None, save: bool = True) -> plt.Figure:

    if lags is None:
        lags = [1, 3, 6, 12]

    s = _get_series(df)
    n = len(lags)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, lag in zip(axes, lags):
        x = s.iloc[:-lag].values
        y = s.iloc[lag:].values
        ax.scatter(x, y, alpha=0.5, s=20, color="#2C7BB6")
        ax.set_xlabel(f"y(t)")
        ax.set_ylabel(f"y(t+{lag})")
        ax.set_title(f"Lag {lag}")
        corr = np.corrcoef(x, y)[0, 1]
        ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=10, color="red")

    fig.suptitle("Lag Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "04_lag_plots")
    return fig


def plot_acf_pacf(df: pd.DataFrame, lags: int = 36, save: bool = True) -> plt.Figure:

    s = _get_series(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(s, lags=min(lags, len(s) // 2 - 1), ax=axes[0], color="#2C7BB6")
    axes[0].set_title("Автокорреляционная функция (ACF)", fontweight="bold")
    plot_pacf(s, lags=min(lags, len(s) // 2 - 1), ax=axes[1], color="#D7191C", method="ywm")
    axes[1].set_title("Частная автокорреляционная функция (PACF)", fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "05_acf_pacf")
    return fig


# 2. Декомпозиция

def plot_stl_decomposition(
    df: pd.DataFrame,
    period: int = 12,
    save: bool = True,
) -> plt.Figure:

    s = _get_series(df)
    stl = STL(s, period=period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    components = [
        (s.values, "Исходный ряд", "#2C7BB6"),
        (result.trend, "Тренд", "#D7191C"),
        (result.seasonal, "Сезонность", "#1A9641"),
        (result.resid, "Остатки", "#756BB1"),
    ]
    for ax, (data, title, color) in zip(axes, components):
        ax.plot(s.index, data, color=color, linewidth=1.5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Значение")

    axes[-1].set_xlabel("Дата")
    fig.suptitle("STL Декомпозиция", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "06_stl_decomposition")
    return fig


def plot_classical_decomposition(
    df: pd.DataFrame,
    model: str = "multiplicative",
    period: int = 12,
    save: bool = True,
) -> plt.Figure:

    s = _get_series(df)
    # multiplicative требует положительных значений
    if model == "multiplicative" and (s <= 0).any():
        logger.warning("Обнаружены неположительные значения — переключаемся на additive.")
        model = "additive"

    result = seasonal_decompose(s, model=model, period=period)
    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"Классическая декомпозиция ({model})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "07_classical_decomposition")
    return fig


# 3. Тесты на стационарность

def adf_test(series: pd.Series, verbose: bool = True) -> dict:

    result = adfuller(series.dropna(), autolag="AIC")
    output = {
        "test_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "stationary": result[1] < 0.05,
    }
    if verbose:
        print("\n=== ADF Test ===")
        print(f"Test Statistic : {output['test_statistic']:.4f}")
        print(f"p-value        : {output['p_value']:.4f}")
        for key, val in output["critical_values"].items():
            print(f"  Critical ({key}): {val:.4f}")
        verdict = "СТАЦИОНАРЕН ✓" if output["stationary"] else "НЕСТАЦИОНАРЕН ✗"
        print(f"Вывод          : {verdict} (p {'<' if output['stationary'] else '>'} 0.05)")
    return output


def kpss_test(series: pd.Series, regression: str = "c", verbose: bool = True) -> dict:

    result = kpss(series.dropna(), regression=regression, nlags="auto")
    output = {
        "test_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "critical_values": result[3],
        "stationary": result[1] > 0.05,
    }
    if verbose:
        print("\n=== KPSS Test ===")
        print(f"Test Statistic : {output['test_statistic']:.4f}")
        print(f"p-value        : {output['p_value']:.4f}")
        for key, val in output["critical_values"].items():
            print(f"  Critical ({key}): {val:.4f}")
        verdict = "СТАЦИОНАРЕН ✓" if output["stationary"] else "НЕСТАЦИОНАРЕН ✗"
        print(f"Вывод          : {verdict} (p {'>' if output['stationary'] else '<'} 0.05)")
    return output


def make_stationary(
    df: pd.DataFrame,
    max_diff: int = 2,
    seasonal_period: int = 12,
) -> tuple[pd.DataFrame, int, bool]:

    s = _get_series(df)
    n_diff = 0
    seasonal_diff = False

    # Проверяем исходный ряд
    result = adfuller(s.dropna(), autolag="AIC")
    if result[1] < 0.05:
        logger.info("Ряд уже стационарен.")
        return df, 0, False

    # Пробуем сезонное дифференцирование
    s_sdiff = s.diff(seasonal_period).dropna()
    result_s = adfuller(s_sdiff.dropna(), autolag="AIC")
    if result_s[1] < 0.05:
        seasonal_diff = True
        logger.info("После сезонного дифференцирования (lag=%d): стационарен (p=%.4f)",
                    seasonal_period, result_s[1])
        out = pd.DataFrame({"unique_id": "train_occupancy",
                            "ds": s_sdiff.index, "y": s_sdiff.values})
        return out, 1, True

    # Обычное дифференцирование
    s_work = s.copy()
    for i in range(1, max_diff + 1):
        s_work = s_work.diff().dropna()
        n_diff += 1
        result = adfuller(s_work.dropna(), autolag="AIC")
        logger.info("Дифференцирование %d: p=%.4f", i, result[1])
        if result[1] < 0.05:
            break

    out = pd.DataFrame({"unique_id": "train_occupancy",
                        "ds": s_work.index, "y": s_work.values})
    return out, n_diff, seasonal_diff


# 4. Анализ остатков

def plot_residuals(residuals: np.ndarray | pd.Series, model_name: str = "Model",
                   save: bool = True) -> plt.Figure:

    res = pd.Series(residuals).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Анализ остатков: {model_name}", fontsize=14, fontweight="bold")

    # Residual plot
    axes[0, 0].plot(res.values, color="#2C7BB6", linewidth=1)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Остатки")
    axes[0, 0].set_xlabel("Наблюдение")

    # Histogram
    axes[0, 1].hist(res.values, bins=20, color="#2C7BB6", edgecolor="white", density=True)
    xmin, xmax = axes[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, res.mean(), res.std()),
                    color="red", linewidth=2, label="Normal PDF")
    axes[0, 1].set_title("Гистограмма остатков")
    axes[0, 1].legend()

    # QQ plot
    stats.probplot(res.values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("QQ Plot")
    axes[1, 0].get_lines()[1].set_color("red")

    # Ljung-Box
    max_lags = min(20, len(res) // 2 - 1)
    lb_result = acorr_ljungbox(res, lags=range(1, max_lags + 1), return_df=True)
    axes[1, 1].bar(lb_result.index, lb_result["lb_pvalue"], color="#2C7BB6", alpha=0.8)
    axes[1, 1].axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="p=0.05")
    axes[1, 1].set_title("Ljung-Box p-values")
    axes[1, 1].set_xlabel("Лаг")
    axes[1, 1].set_ylabel("p-value")
    axes[1, 1].legend()

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    if save:
        _save(fig, f"residuals_{safe_name}")
    return fig


# 5. Полный EDA pipeline

def run_full_eda(df: pd.DataFrame) -> dict:

    logger.info("=== Запуск полного EDA ===")

    plot_series_overview(df)
    plot_boxplot_by_month(df)
    plot_seasonal_plot(df)
    plot_lag_plot(df)
    plot_acf_pacf(df)
    plot_stl_decomposition(df)
    plot_classical_decomposition(df)

    s = _get_series(df)
    adf_result = adf_test(s)
    kpss_result = kpss_test(s)

    logger.info("=== EDA завершён ===")
    return {"adf": adf_result, "kpss": kpss_result}
