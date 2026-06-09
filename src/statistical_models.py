from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    HistoricAverage,
    Naive,
    SeasonalNaive,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Конфигурация моделей

def get_models(season_length: int = 12) -> list:

    return [
        Naive(),
        SeasonalNaive(season_length=season_length),
        AutoARIMA(season_length=season_length, approximation=False),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
        HistoricAverage(),
    ]


# Обучение и прогноз

def train_statistical_models(
    df: pd.DataFrame,
    horizon: int = 24,
    season_length: int = 12,
    freq: str = "MS",
) -> tuple[StatsForecast, pd.DataFrame]:

    models = get_models(season_length)
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1,
        verbose=True,
    )
    logger.info("Обучение статистических моделей (горизонт=%d мес.)...", horizon)
    sf.fit(df)

    forecasts = sf.predict(h=horizon, level=[80, 95])
    logger.info("Прогноз построен: %s", forecasts.shape)
    return sf, forecasts


# Визуализация

def plot_forecasts(
    train_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    model_names: list[str] | None = None,
    horizon: int = 24,
    save: bool = True,
) -> plt.Figure:

    if model_names is None:
        # Все колонки, кроме unique_id и ds
        model_names = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
        # Только базовые прогнозы (без -lo/-hi)
        model_names = [c for c in model_names if "-" not in c]

    n_models = len(model_names)
    cols = 2
    rows = (n_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    train_series = train_df.set_index("ds")["y"].sort_index()

    for i, model in enumerate(model_names):
        ax = axes[i]
        # История
        ax.plot(train_series.index[-36:], train_series.values[-36:],
                color="#2C7BB6", linewidth=1.5, label="История")

        fc_df = forecasts[["ds", model]].set_index("ds")
        ax.plot(fc_df.index, fc_df[model], color="#D7191C",
                linewidth=2, linestyle="--", label="Прогноз")

        # Доверительные интервалы (если есть)
        lo_col = f"{model}-lo-95"
        hi_col = f"{model}-hi-95"
        if lo_col in forecasts.columns and hi_col in forecasts.columns:
            lo = forecasts.set_index("ds")[lo_col]
            hi = forecasts.set_index("ds")[hi_col]
            ax.fill_between(fc_df.index, lo, hi, alpha=0.2, color="#D7191C", label="CI 95%")

        ax.set_title(model, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    # Скрыть лишние оси
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Прогнозы статистических моделей", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "08_statistical_forecasts.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig


def plot_prediction_intervals(
    train_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    best_model: str,
    save: bool = True,
) -> plt.Figure:

    train_series = train_df.set_index("ds")["y"].sort_index()
    fc_df = forecasts.set_index("ds")

    fig, ax = plt.subplots(figsize=(14, 6))

    # История (последние 3 года)
    ax.plot(train_series.index[-36:], train_series.values[-36:],
            color="#2C7BB6", linewidth=2, label="История")

    # Точечный прогноз
    ax.plot(fc_df.index, fc_df[best_model], color="#D7191C",
            linewidth=2.5, linestyle="--", label="Прогноз")

    # PI 95%
    if f"{best_model}-lo-95" in fc_df.columns:
        ax.fill_between(
            fc_df.index,
            fc_df[f"{best_model}-lo-95"],
            fc_df[f"{best_model}-hi-95"],
            alpha=0.15, color="#D7191C", label="PI 95%",
        )
    # PI 80%
    if f"{best_model}-lo-80" in fc_df.columns:
        ax.fill_between(
            fc_df.index,
            fc_df[f"{best_model}-lo-80"],
            fc_df[f"{best_model}-hi-80"],
            alpha=0.25, color="#D7191C", label="PI 80%",
        )

    ax.set_title(f"Вероятностный прогноз: {best_model}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Пассажиропоток")
    ax.legend()
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f"09_probabilistic_{best_model.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig


# Метрики

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE — исключаем нули
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_vals = np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
    smape = np.mean(smape_vals) * 100

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2), "sMAPE": round(smape, 2)}


# Бектестинг

def run_backtesting(
    df: pd.DataFrame,
    horizon: int = 24,
    n_windows: int = 5,
    season_length: int = 12,
    freq: str = "MS",
) -> pd.DataFrame:

    models = get_models(season_length)
    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)

    logger.info(
        "Бектестинг: %d окон, горизонт=%d мес. ...", n_windows, horizon
    )
    cv_df = sf.cross_validation(
        df=df,
        h=horizon,
        n_windows=n_windows,
        step_size=1,
        refit=True,
    )
    logger.info("Бектестинг завершён: %s строк", len(cv_df))
    return cv_df


def summarize_backtesting(cv_df: pd.DataFrame) -> pd.DataFrame:

    model_cols = [c for c in cv_df.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    rows = []
    for model in model_cols:
        m = compute_metrics(cv_df["y"].values, cv_df[model].values)
        m["Model"] = model
        rows.append(m)

    summary = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
    return summary
