from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Константы
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Загрузка

def load_raw_data(filepath: str | Path) -> pd.DataFrame:

    filepath = Path(filepath)
    logger.info("Загрузка данных из %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Загружено строк: %d, колонок: %d", len(df), len(df.columns))
    logger.info("Колонки: %s", df.columns.tolist())
    return df


# Анализ качества

def quality_report(df: pd.DataFrame) -> dict:

    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    logger.info("Отчёт о качестве: %s", report)
    return report


# Определение временного столбца

def detect_datetime_column(df: pd.DataFrame) -> str:

    # Явные кандидаты по имени
    date_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ("date", "time", "datetime", "timestamp", "dt")
    )]
    if date_candidates:
        logger.info("Обнаружен временной столбец: %s", date_candidates[0])
        return date_candidates[0]

    # Пробуем преобразовать каждый object-столбец
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(20))
            logger.info("Временной столбец определён по содержимому: %s", col)
            return col
        except Exception:
            continue

    raise ValueError(
        "Не удалось автоматически определить временной столбец. "
        "Укажите его явно через параметр date_col."
    )


def detect_target_column(df: pd.DataFrame, date_col: str) -> str:

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Приоритет — колонки с «occupancy», «count», «passengers» в названии
    priority = [c for c in numeric_cols if any(
        kw in c.lower() for kw in ("occupan", "count", "passeng", "load", "ridership", "demand")
    )]
    if priority:
        logger.info("Целевой столбец: %s", priority[0])
        return priority[0]

    # Первая числовая колонка, не являющаяся датой
    candidates = [c for c in numeric_cols if c != date_col]
    if candidates:
        logger.info("Целевой столбец (первый числовой): %s", candidates[0])
        return candidates[0]

    raise ValueError("Не удалось определить целевой столбец.")


# Очистка

def clean_data(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    outlier_z_threshold: float = 3.5,
) -> pd.DataFrame:

    df = df.copy()

    # Парсинг дат
    # Обработка нестандартного формата 1999M01
    if df[date_col].astype(str).str.match(r'^\d{4}M\d{2}$').any():
        df[date_col] = pd.to_datetime(
            df[date_col].astype(str).str.replace('M', '-', regex=False),
            format='%Y-%m'
        )
    else:
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    logger.info("Диапазон дат: %s — %s", df[date_col].min(), df[date_col].max())

    # Удаление дубликатов
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Удалено дубликатов: %d", n_before - len(df))

    # Целевой столбец — числовой
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Заполнение пропусков интерполяцией
    n_missing = df[target_col].isnull().sum()
    if n_missing > 0:
        df[target_col] = df[target_col].interpolate(method="time")
        logger.info("Заполнено пропусков интерполяцией: %d", n_missing)

    # Обнаружение и ограничение выбросов (winsorizing)
    z = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
    n_outliers = (z > outlier_z_threshold).sum()
    if n_outliers > 0:
        lower = df[target_col].quantile(0.01)
        upper = df[target_col].quantile(0.99)
        df[target_col] = df[target_col].clip(lower, upper)
        logger.info(
            "Ограничено выбросов (z > %.1f): %d → winsorized [%.2f, %.2f]",
            outlier_z_threshold, n_outliers, lower, upper,
        )

    # Сортировка
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


# Агрегация до месячного уровня

def aggregate_to_monthly(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    agg_func: str = "sum",
) -> pd.DataFrame:

    df = df.copy()
    df["month_period"] = df[date_col].dt.to_period("M")

    monthly = (
        df.groupby("month_period")[target_col]
        .agg(agg_func)
        .reset_index()
        .rename(columns={"month_period": "ds", target_col: "y"})
    )

    # Конвертируем Period → Timestamp (начало месяца)
    monthly["ds"] = monthly["ds"].dt.to_timestamp()
    monthly["unique_id"] = "train_occupancy"

    monthly = monthly.sort_values("ds").reset_index(drop=True)
    logger.info(
        "После агрегации (%s): %d месячных точек (%s — %s)",
        agg_func, len(monthly), monthly["ds"].min().date(), monthly["ds"].max().date(),
    )
    return monthly[["unique_id", "ds", "y"]]


# Сохранение

def save_processed(df: pd.DataFrame, filename: str = "monthly_series.csv") -> Path:

    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Сохранено: %s", out_path)
    return out_path


# Единая точка входа

def prepare_data(
    filepath: str | Path,
    date_col: str | None = None,
    target_col: str | None = None,
    agg_func: str = "sum",
    outlier_z_threshold: float = 3.5,
    save: bool = True,
) -> pd.DataFrame:

    raw = load_raw_data(filepath)
    _ = quality_report(raw)

    if date_col is None:
        date_col = detect_datetime_column(raw)
    if target_col is None:
        target_col = detect_target_column(raw, date_col)

    cleaned = clean_data(raw, date_col, target_col, outlier_z_threshold)
    monthly = aggregate_to_monthly(cleaned, date_col, target_col, agg_func)

    if save:
        save_processed(monthly)

    return monthly


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_preparation.py <path_to_csv>")
        sys.exit(1)

    result = prepare_data(sys.argv[1])
    print(result.head(10))
    print(f"\nИтого: {len(result)} месяцев")
