from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"


# Шаг 1: Загрузка и препроцессинг

def step_load_and_prepare(filepath: str | Path) -> pd.DataFrame:

    from src.data_preparation import prepare_data

    logger.info("[Pipeline] Шаг 1: Загрузка данных...")
    df = prepare_data(filepath, save=True)
    logger.info("[Pipeline] Загружено %d месячных точек.", len(df))
    return df


# Шаг 2: Feature Engineering (вспомогательные признаки для анализа)

def step_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("[Pipeline] Шаг 2: Feature Engineering...")
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["year"] = df["ds"].dt.year

    df["lag_1"] = df["y"].shift(1)
    df["lag_3"] = df["y"].shift(3)
    df["lag_6"] = df["y"].shift(6)
    df["lag_12"] = df["y"].shift(12)

    df["rolling_mean_3"] = df["y"].rolling(3).mean()
    df["rolling_mean_6"] = df["y"].rolling(6).mean()
    df["rolling_mean_12"] = df["y"].rolling(12).mean()
    df["rolling_std_3"] = df["y"].rolling(3).std()
    df["rolling_std_12"] = df["y"].rolling(12).std()

    # Сезонные индикаторы
    df["is_q1"] = (df["quarter"] == 1).astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

    logger.info("[Pipeline] Признаки сгенерированы: %s", df.columns.tolist())
    return df


# Шаг 3: Обучение лучшей модели

def step_train_best_model(
    df: pd.DataFrame,
    best_model_type: str = "auto",
    horizon: int = 24,
    freq: str = "MS",
    season_length: int = 12,
) -> Any:

    logger.info("[Pipeline] Шаг 3: Обучение модели '%s'...", best_model_type)

    base_df = df[["unique_id", "ds", "y"]].copy()

    if best_model_type in ("auto", "statistical"):
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        sf = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=freq,
            n_jobs=-1,
        )
        sf.fit(base_df)
        logger.info("[Pipeline] AutoARIMA обучен.")
        return sf

    elif best_model_type == "ml":
        from src.ml_models import build_mlforecast

        mlf = build_mlforecast(season_length, freq)
        mlf.fit(base_df)
        logger.info("[Pipeline] LightGBM (MLForecast) обучен.")
        return mlf

    elif best_model_type == "dl":
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        nf = NeuralForecast(
            models=[NBEATS(h=horizon, input_size=36, max_steps=500,
                           scaler_type="standard", random_seed=42)],
            freq=freq,
        )
        nf.fit(base_df)
        logger.info("[Pipeline] NBEATS обучен.")
        return nf

    else:
        raise ValueError(f"Неизвестный тип модели: {best_model_type}")


# Шаг 4: Прогноз

def step_predict(model: Any, horizon: int = 24) -> pd.DataFrame:

    logger.info("[Pipeline] Шаг 4: Прогноз на %d месяцев...", horizon)

    # StatsForecast
    if hasattr(model, "predict") and hasattr(model, "fitted_"):
        forecasts = model.predict(h=horizon, level=[80, 95])
    # MLForecast
    elif hasattr(model, "predict") and hasattr(model, "models_"):
        forecasts = model.predict(horizon)
    # NeuralForecast
    elif hasattr(model, "predict") and hasattr(model, "models"):
        forecasts = model.predict()
    else:
        raise TypeError(f"Неизвестный тип модели: {type(model)}")

    logger.info("[Pipeline] Прогноз: %s", forecasts.shape)
    return forecasts


# Шаг 5: Сохранение результатов

def step_save_results(
    forecasts: pd.DataFrame,
    filename: str = "final_forecast.csv",
) -> Path:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    forecasts.to_csv(out_path, index=False)
    logger.info("[Pipeline] Результаты сохранены: %s", out_path)
    return out_path


# Тестирование устойчивости

def test_robustness(
    df: pd.DataFrame,
    missing_fraction: float = 0.05,
    outlier_fraction: float = 0.05,
    horizon: int = 24,
    freq: str = "MS",
    season_length: int = 12,
) -> dict:

    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    results = {}

    # Тест 1: устойчивость к пропускам
    df_missing = df.copy()
    np.random.seed(42)
    idx = np.random.choice(df_missing.index, size=max(1, int(len(df_missing) * missing_fraction)),
                           replace=False)
    df_missing.loc[idx, "y"] = np.nan
    df_missing["y"] = df_missing["y"].interpolate(method="linear").fillna(method="bfill")

    try:
        sf = StatsForecast(models=[AutoARIMA(season_length=season_length)],
                           freq=freq, n_jobs=1)
        sf.fit(df_missing[["unique_id", "ds", "y"]])
        fc = sf.predict(h=horizon)
        results["missing_data"] = {
            "status": "OK",
            "missing_rows": len(idx),
            "forecast_rows": len(fc),
        }
    except Exception as e:
        results["missing_data"] = {"status": "ERROR", "error": str(e)}

    # Тест 2: устойчивость к выбросам
    df_outliers = df.copy()
    outlier_idx = np.random.choice(df_outliers.index,
                                   size=max(1, int(len(df_outliers) * outlier_fraction)),
                                   replace=False)
    df_outliers.loc[outlier_idx, "y"] *= 10  # Экстремальные выбросы

    try:
        sf2 = StatsForecast(models=[AutoARIMA(season_length=season_length)],
                            freq=freq, n_jobs=1)
        sf2.fit(df_outliers[["unique_id", "ds", "y"]])
        fc2 = sf2.predict(h=horizon)
        results["outlier_data"] = {
            "status": "OK",
            "outlier_rows": len(outlier_idx),
            "forecast_rows": len(fc2),
        }
    except Exception as e:
        results["outlier_data"] = {"status": "ERROR", "error": str(e)}

    logger.info("[Pipeline] Тест устойчивости: %s", results)
    return results


# Полный пайплайн

def run_pipeline(
    filepath: str | Path,
    best_model_type: str = "auto",
    horizon: int = 24,
    freq: str = "MS",
    season_length: int = 12,
    run_robustness: bool = True,
) -> dict:

    pipeline_start = time.time()
    results: dict[str, Any] = {}

    try:
        # Шаг 1
        t0 = time.time()
        df = step_load_and_prepare(filepath)
        results["step1_load"] = {"rows": len(df), "time_sec": round(time.time() - t0, 2)}

        # Шаг 2
        t0 = time.time()
        df_features = step_feature_engineering(df)
        results["step2_features"] = {
            "n_features": len(df_features.columns),
            "time_sec": round(time.time() - t0, 2),
        }

        # Шаг 3
        t0 = time.time()
        model = step_train_best_model(df, best_model_type, horizon, freq, season_length)
        results["step3_train"] = {
            "model_type": best_model_type,
            "time_sec": round(time.time() - t0, 2),
        }

        # Шаг 4
        t0 = time.time()
        forecasts = step_predict(model, horizon)
        results["step4_forecast"] = {
            "rows": len(forecasts),
            "time_sec": round(time.time() - t0, 2),
        }

        # Шаг 5
        t0 = time.time()
        out_path = step_save_results(forecasts)
        results["step5_save"] = {"path": str(out_path), "time_sec": round(time.time() - t0, 2)}

        # Тест устойчивости
        if run_robustness:
            results["robustness"] = test_robustness(df, horizon=horizon,
                                                    freq=freq, season_length=season_length)

        results["total_time_sec"] = round(time.time() - pipeline_start, 2)
        results["status"] = "SUCCESS"
        logger.info("[Pipeline] Завершён за %.1f сек.", results["total_time_sec"])

    except Exception:
        results["status"] = "ERROR"
        results["traceback"] = traceback.format_exc()
        logger.error("[Pipeline] ОШИБКА:\n%s", results["traceback"])

    # Сохраняем лог пайплайна
    log_path = PROCESSED_DIR / "pipeline_log.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("[Pipeline] Лог сохранён: %s", log_path)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_csv> [model_type]")
        sys.exit(1)

    fp = sys.argv[1]
    mtype = sys.argv[2] if len(sys.argv) > 2 else "auto"

    res = run_pipeline(fp, best_model_type=mtype)
    print(json.dumps(res, indent=2, default=str))
