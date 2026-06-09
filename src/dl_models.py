from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Конфигурация моделей

def get_dl_models(
    horizon: int = 24,
    input_size: int = 36,
    max_steps: int = 500,
) -> list:

    from neuralforecast.models import LSTM, NBEATS, NHITS

    models = [
        NBEATS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            scaler_type="standard",
            random_seed=42,
            logger_name=None,
            enable_progress_bar=True,
        ),
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            scaler_type="standard",
            random_seed=42,
            logger_name=None,
            enable_progress_bar=True,
        ),
        LSTM(
            h=horizon,
            input_size=input_size,
            inference_input_size=input_size,
            encoder_hidden_size=128,
            encoder_n_layers=2,
            max_steps=max_steps,
            scaler_type="standard",
            random_seed=42,
            logger_name=None,
            enable_progress_bar=True,
        ),
    ]
    logger.info("DL-модели: %s", [type(m).__name__ for m in models])
    return models


# Обучение и прогноз

def train_dl_models(
    df: pd.DataFrame,
    horizon: int = 24,
    input_size: int = 36,
    max_steps: int = 500,
    freq: str = "MS",
) -> tuple:

    from neuralforecast import NeuralForecast

    models = get_dl_models(horizon, input_size, max_steps)
    nf = NeuralForecast(models=models, freq=freq)

    logger.info("Обучение DL-моделей (max_steps=%d)...", max_steps)
    nf.fit(df)

    forecasts = nf.predict()
    logger.info("DL прогноз построен: %s", forecasts.shape)
    return nf, forecasts


# Бектестинг DL

def run_dl_backtesting(
    df: pd.DataFrame,
    horizon: int = 24,
    n_windows: int = 3,
    input_size: int = 36,
    max_steps: int = 300,
    freq: str = "MS",
) -> pd.DataFrame:

    from neuralforecast import NeuralForecast

    models = get_dl_models(horizon, input_size, max_steps)
    nf = NeuralForecast(models=models, freq=freq)

    logger.info("DL бектестинг: %d окон...", n_windows)
    cv_df = nf.cross_validation(df=df, n_windows=n_windows, step_size=1)
    logger.info("DL бектестинг завершён: %d строк", len(cv_df))
    return cv_df


# Метрики

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2), "sMAPE": round(smape, 2)}


def summarize_dl_backtesting(cv_df: pd.DataFrame) -> pd.DataFrame:

    model_cols = [c for c in cv_df.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    rows = []
    for model in model_cols:
        m = compute_metrics(cv_df["y"].values, cv_df[model].values)
        m["Model"] = model
        rows.append(m)

    return pd.DataFrame(rows).set_index("Model").sort_values("RMSE")


# Визуализация

def plot_dl_forecasts(
    train_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:

    model_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
    train_series = train_df.set_index("ds")["y"].sort_index()

    fig, axes = plt.subplots(1, len(model_cols), figsize=(6 * len(model_cols), 5))
    if len(model_cols) == 1:
        axes = [axes]

    colors = ["#D7191C", "#1A9641", "#756BB1"]
    for ax, model, color in zip(axes, model_cols, colors):
        ax.plot(train_series.index[-36:], train_series.values[-36:],
                color="#2C7BB6", linewidth=1.5, label="История")
        fc = forecasts.set_index("ds")[model]
        ax.plot(fc.index, fc.values, color=color,
                linewidth=2, linestyle="--", label="Прогноз")
        ax.set_title(model, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Прогнозы DL-моделей", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "12_dl_forecasts.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig


def plot_training_curves(nf, save: bool = True) -> None:

    for model in nf.models:
        model_name = type(model).__name__
        if hasattr(model, "loss_logger") and model.loss_logger is not None:
            logger.info("Кривые потерь для %s недоступны в текущей конфигурации.", model_name)
        else:
            logger.info("Кривые потерь для %s: не логируются.", model_name)
