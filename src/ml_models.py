from __future__ import annotations

import logging
from pathlib import Path
from mlforecast.target_transforms import Differences
from window_ops.rolling import rolling_mean, rolling_std

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)



def get_date_features() -> list:

    from mlforecast.feature_engineering import DateFeatures
    import utilsforecast.processing as ufp

    date_features = ["month", "quarter", "year"]
    return date_features


def build_mlforecast(
    season_length: int = 12,
    freq: str = "MS",
) -> MLForecast:

    models = [
        LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    ]

    mlf = MLForecast(
        models=models,
        freq=freq,
        lags=[1, 3, 6, 12],
        lag_transforms={
            1: [
                (rolling_mean, 3),
                (rolling_mean, 6),
                (rolling_mean, 12),
                (rolling_std, 3),
                (rolling_std, 12),
            ],
        },
        date_features=["month", "quarter", "year"],
        target_transforms=[Differences([1])],
    )
    logger.info("MLForecast создан с моделями: %s",
                [type(m).__name__ for m in models])
    return mlf


# Обучение и прогноз

def train_ml_models(
    df: pd.DataFrame,
    horizon: int = 24,
    season_length: int = 12,
    freq: str = "MS",
) -> tuple[MLForecast, pd.DataFrame]:

    mlf = build_mlforecast(season_length, freq)
    logger.info("Обучение ML-моделей...")
    mlf.fit(df)

    forecasts = mlf.predict(horizon)
    logger.info("Прогноз ML-моделей построен: %s", forecasts.shape)
    return mlf, forecasts


# Бектестинг ML

def run_ml_backtesting(
    df: pd.DataFrame,
    horizon: int = 24,
    n_windows: int = 5,
    season_length: int = 12,
    freq: str = "MS",
) -> pd.DataFrame:

    mlf = build_mlforecast(season_length, freq)
    logger.info("Бектестинг ML-моделей: %d окон, горизонт=%d...", n_windows, horizon)

    cv_df = mlf.cross_validation(
        df=df,
        h=horizon,
        n_windows=n_windows,
        step_size=1,
        refit=True,
    )
    logger.info("Бектестинг ML завершён: %d строк", len(cv_df))
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


def summarize_ml_backtesting(cv_df: pd.DataFrame) -> pd.DataFrame:

    model_cols = [c for c in cv_df.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    rows = []
    for model in model_cols:
        m = compute_metrics(cv_df["y"].values, cv_df[model].values)
        m["Model"] = model
        rows.append(m)

    return pd.DataFrame(rows).set_index("Model").sort_values("RMSE")


# Визуализация

def plot_ml_forecasts(
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

    fig.suptitle("Прогнозы ML-моделей", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "10_ml_forecasts.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig


def plot_feature_importance(mlf: MLForecast, save: bool = True) -> plt.Figure:

    model = mlf.models_[list(mlf.models_.keys())[0]]
    if not hasattr(model, "feature_importances_"):
        logger.warning("Модель не поддерживает feature_importances_")
        return None

    # Имена признаков
    if hasattr(mlf, "ts") and hasattr(mlf.ts, "features_order_"):
        feature_names = mlf.ts.features_order_
    else:
        feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]

    importance = pd.Series(
        model.feature_importances_, index=feature_names[:len(model.feature_importances_)]
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color="#2C7BB6", edgecolor="white")
    ax.set_title("Feature Importance (LightGBM)", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "11_feature_importance.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig
