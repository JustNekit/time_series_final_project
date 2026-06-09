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


# Утилиты метрик

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan}

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nonzero = y_true != 0
    mape = (np.mean(np.abs(
        (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
    )) * 100) if nonzero.any() else np.nan
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 2),
        "sMAPE": round(float(smape), 2),
    }


# Агрегация результатов бектестинга

def extract_metrics_from_cv(
    cv_df: pd.DataFrame,
    category: str,
) -> pd.DataFrame:

    model_cols = [c for c in cv_df.columns
                  if c not in ("unique_id", "ds", "cutoff", "y")
                  and "-lo-" not in c and "-hi-" not in c]
    rows = []
    for model in model_cols:
        m = compute_metrics(cv_df["y"].values, cv_df[model].values)
        m["Model"] = model
        m["Category"] = category
        rows.append(m)
    return pd.DataFrame(rows)


def build_comparison_table(
    stat_cv: pd.DataFrame | None = None,
    ml_cv: pd.DataFrame | None = None,
    dl_cv: pd.DataFrame | None = None,
    extra_rows: list[dict] | None = None,
) -> pd.DataFrame:

    frames = []

    if stat_cv is not None:
        # Разделяем Naive-модели (Baseline) от остальных
        baseline_models = {"Naive", "SeasonalNaive", "HistoricAverage"}
        stat_cols = [c for c in stat_cv.columns
                     if c not in ("unique_id", "ds", "cutoff", "y")
                     and "-lo-" not in c and "-hi-" not in c]

        for model in stat_cols:
            cat = "Baseline" if model in baseline_models else "Statistical"
            m = compute_metrics(stat_cv["y"].values, stat_cv[model].values)
            m["Model"] = model
            m["Category"] = cat
            frames.append(m)

    if ml_cv is not None:
        frames.extend(extract_metrics_from_cv(ml_cv, "ML").to_dict("records"))

    if dl_cv is not None:
        frames.extend(extract_metrics_from_cv(dl_cv, "DL").to_dict("records"))

    if extra_rows:
        frames.extend(extra_rows)

    comparison = pd.DataFrame(frames)
    comparison = comparison[["Model", "Category", "MAE", "RMSE", "MAPE", "sMAPE"]]
    comparison = comparison.sort_values("RMSE").reset_index(drop=True)
    return comparison


# Визуализация сравнения

def plot_comparison_table(comparison: pd.DataFrame, save: bool = True) -> plt.Figure:

    import seaborn as sns

    metrics = ["MAE", "RMSE", "MAPE", "sMAPE"]
    heat_data = comparison.set_index("Model")[metrics]

    # Нормализация для цвета (по столбцу)
    normalized = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, max(5, len(comparison) * 0.5)))
    sns.heatmap(
        normalized,
        annot=heat_data.round(2),
        fmt="g",
        cmap="RdYlGn_r",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Нормализованное значение"},
    )
    ax.set_title("Сравнение всех моделей (меньше = лучше)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Модель")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "13_model_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig


def plot_metric_bars(
    comparison: pd.DataFrame,
    metric: str = "RMSE",
    save: bool = True,
) -> plt.Figure:

    category_colors = {
        "Baseline": "#AAAAAA",
        "Statistical": "#2C7BB6",
        "ML": "#1A9641",
        "DL": "#D7191C",
    }

    sorted_df = comparison.sort_values(metric)
    colors = [category_colors.get(cat, "#333333") for cat in sorted_df["Category"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(sorted_df["Model"], sorted_df[metric], color=colors, edgecolor="white")
    ax.set_xlabel(metric)
    ax.set_title(f"Сравнение моделей по {metric}", fontweight="bold")

    # Легенда по категориям
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in category_colors.items()
                       if k in sorted_df["Category"].values]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f"14_comparison_{metric.lower()}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("График сохранён: %s", path)

    return fig


# Анализ остатков (delegated import)

def get_residuals_for_model(
    cv_df: pd.DataFrame,
    model_name: str,
) -> np.ndarray:

    if model_name not in cv_df.columns:
        raise ValueError(f"Модель '{model_name}' не найдена в CV DataFrame.")
    return (cv_df["y"] - cv_df[model_name]).values
