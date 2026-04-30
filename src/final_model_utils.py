"""Helpers for the final model comparison, forecasting, and diagnostics workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


LOGGER = logging.getLogger(__name__)
SKEW_THRESHOLD = 1.5


@dataclass(slots=True)
class FinalFeatureBuildResult:
    frame: pd.DataFrame
    categorical_features: list[str]
    numeric_features: list[str]
    engineered_feature_notes: list[str]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def ensure_final_output_dirs(base_dir: Path, report_subdir: str | None = None) -> dict[str, Path]:
    reports_dir = base_dir / "reports"
    if report_subdir:
        reports_dir = reports_dir / report_subdir
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {"reports": reports_dir, "figures": figures_dir}


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def load_final_analysis_frame(base_dir: Path, dataset_path: Path | None = None) -> pd.DataFrame:
    desired_columns = [
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "route",
        "Distance",
        "scheduled_departure_hour_local",
        "scheduled_arrival_hour_local",
        "day_of_week",
        "month",
        "dep_delay_minutes",
        "arr_delay_minutes",
        "dep_delayed_15",
        "previous_leg_delay_available",
        "previous_leg_arr_delay_minutes",
        "origin_hourly_avg_dep_delay_prior",
        "carrier_route_avg_dep_delay_prior",
        "origin_hourly_congestion_proxy_prior",
        "origin_temp_c",
        "origin_wind_speed_mps",
        "origin_visibility_m",
        "origin_precip_mm",
        "origin_dew_point_c",
        "origin_humidity_pct",
        "origin_ceiling_m",
        "dest_temp_c",
        "dest_wind_speed_mps",
        "dest_visibility_m",
        "dest_precip_mm",
        "dest_dew_point_c",
        "dest_humidity_pct",
        "dest_ceiling_m",
        "sched_dep_utc",
        "origin_weather_timestamp_utc",
    ]
    path = dataset_path if dataset_path is not None else base_dir / "data" / "modeling_dataset_fm15_strict_top25.parquet"
    LOGGER.info("Loading final analysis dataset from %s", path)
    available_columns = pq.ParquetFile(path).schema.names
    read_columns = [column for column in desired_columns if column in available_columns]
    frame = pd.read_parquet(path, columns=read_columns)

    if "route" not in frame.columns and {"Origin", "Dest"}.issubset(frame.columns):
        frame["route"] = frame["Origin"].astype("string").fillna("UNK") + "_" + frame["Dest"].astype("string").fillna("UNK")
    if "sched_dep_utc" not in frame.columns:
        if "origin_weather_timestamp_utc" in frame.columns:
            frame["sched_dep_utc"] = pd.to_datetime(frame["origin_weather_timestamp_utc"], utc=True, errors="coerce")
        else:
            frame["sched_dep_utc"] = pd.NaT
    if "scheduled_arrival_hour_local" not in frame.columns and "scheduled_departure_hour_local" in frame.columns:
        frame["scheduled_arrival_hour_local"] = frame["scheduled_departure_hour_local"]
    for missing_numeric in [
        "arr_delay_minutes",
        "previous_leg_delay_available",
        "previous_leg_arr_delay_minutes",
        "dest_temp_c",
        "dest_wind_speed_mps",
        "dest_visibility_m",
        "dest_precip_mm",
        "dest_dew_point_c",
        "dest_humidity_pct",
        "dest_ceiling_m",
        "origin_dew_point_c",
        "month",
        "day_of_week",
    ]:
        if missing_numeric not in frame.columns:
            frame[missing_numeric] = np.nan
    if "month" in frame.columns:
        frame["month"] = frame["month"].fillna(pd.to_datetime(frame["FlightDate"], errors="coerce").dt.month)
    if "day_of_week" in frame.columns:
        frame["day_of_week"] = frame["day_of_week"].fillna(pd.to_datetime(frame["FlightDate"], errors="coerce").dt.dayofweek + 1)

    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="coerce")
    frame["sched_dep_utc"] = pd.to_datetime(frame["sched_dep_utc"], utc=True, errors="coerce")
    numeric_columns = [column for column in desired_columns if column in frame.columns and column not in {"FlightDate", "Reporting_Airline", "Origin", "Dest", "route", "sched_dep_utc", "origin_weather_timestamp_utc"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["FlightDate", "sched_dep_utc", "dep_delay_minutes", "dep_delayed_15"]).copy()
    frame["dep_delayed_15"] = frame["dep_delayed_15"].astype(int)
    frame["dep_delay_minutes_late_only"] = frame["dep_delay_minutes"].clip(lower=0)
    return frame


def _season_from_month(month: pd.Series) -> pd.Series:
    mapping = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
    }
    return month.map(mapping).astype("string")


def _add_previous_hour_features(frame: pd.DataFrame, entity_column: str, prefix: str, include_rolling_rate: bool) -> pd.DataFrame:
    hourly = (
        frame[[entity_column, "dep_hour", "dep_delay_minutes", "dep_delayed_15"]]
        .dropna(subset=[entity_column, "dep_hour"])
        .groupby([entity_column, "dep_hour"], as_index=False)
        .agg(avg_delay=("dep_delay_minutes", "mean"), delay_rate=("dep_delayed_15", "mean"))
        .sort_values([entity_column, "dep_hour"])
    )
    hourly[f"{prefix}_avg_dep_delay_prev_hour"] = hourly.groupby(entity_column, dropna=False)["avg_delay"].shift(1)
    if include_rolling_rate:
        previous_rates = hourly.groupby(entity_column, dropna=False)["delay_rate"].shift(1)
        hourly[f"{prefix}_delay_rate_prev_3h"] = (
            previous_rates.groupby(hourly[entity_column], dropna=False)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    keep_columns = [entity_column, "dep_hour", f"{prefix}_avg_dep_delay_prev_hour"]
    if include_rolling_rate:
        keep_columns.append(f"{prefix}_delay_rate_prev_3h")
    return frame.merge(hourly[keep_columns], on=[entity_column, "dep_hour"], how="left")


def build_final_feature_frame(frame: pd.DataFrame) -> FinalFeatureBuildResult:
    work = frame.copy().sort_values("sched_dep_utc").reset_index(drop=True)
    work["dep_hour"] = work["sched_dep_utc"].dt.floor("h")
    work["is_weekend"] = work["day_of_week"].isin([6, 7]).astype(int)
    work["peak_hour"] = work["scheduled_departure_hour_local"].between(6, 9, inclusive="both").fillna(False).astype(int)
    holidays = USFederalHolidayCalendar().holidays(
        start=work["FlightDate"].min().normalize(),
        end=work["FlightDate"].max().normalize(),
    )
    work["is_holiday"] = work["FlightDate"].dt.normalize().isin(holidays).astype(int)
    work["season"] = _season_from_month(work["month"])

    work = _add_previous_hour_features(work, "Origin", "origin", include_rolling_rate=True)
    work = _add_previous_hour_features(work, "Reporting_Airline", "carrier", include_rolling_rate=True)
    work = _add_previous_hour_features(work, "route", "route", include_rolling_rate=False)

    airport_carrier_keys = ["Origin", "Reporting_Airline"]
    airport_carrier_count = work.groupby(airport_carrier_keys, dropna=False).cumcount()
    airport_carrier_cumsum = work.groupby(airport_carrier_keys, dropna=False)["dep_delay_minutes"].cumsum() - work["dep_delay_minutes"]
    work["airport_carrier_avg_dep_delay_prior"] = airport_carrier_cumsum / airport_carrier_count.replace(0, np.nan)

    work["origin_precip_mm"] = work["origin_precip_mm"].fillna(0.0)
    work["precip_peak_interaction"] = work["origin_precip_mm"] * work["peak_hour"]

    categorical_features = ["Reporting_Airline", "Origin", "Dest", "season"]
    numeric_features = [
        "scheduled_departure_hour_local",
        "scheduled_arrival_hour_local",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "peak_hour",
        "Distance",
        "previous_leg_delay_available",
        "previous_leg_arr_delay_minutes",
        "origin_hourly_avg_dep_delay_prior",
        "carrier_route_avg_dep_delay_prior",
        "origin_hourly_congestion_proxy_prior",
        "origin_temp_c",
        "origin_wind_speed_mps",
        "origin_visibility_m",
        "origin_precip_mm",
        "origin_dew_point_c",
        "origin_humidity_pct",
        "origin_ceiling_m",
        "dest_temp_c",
        "dest_wind_speed_mps",
        "dest_visibility_m",
        "dest_precip_mm",
        "dest_dew_point_c",
        "dest_humidity_pct",
        "dest_ceiling_m",
        "origin_avg_dep_delay_prev_hour",
        "carrier_avg_dep_delay_prev_hour",
        "route_avg_dep_delay_prev_hour",
        "origin_delay_rate_prev_3h",
        "carrier_delay_rate_prev_3h",
        "airport_carrier_avg_dep_delay_prior",
        "precip_peak_interaction",
    ]
    notes = [
        "origin_avg_dep_delay_prev_hour: average departure delay at the same origin during the prior clock hour, shifted so only past hours are used.",
        "carrier_avg_dep_delay_prev_hour: average departure delay for the same carrier during the prior clock hour.",
        "route_avg_dep_delay_prev_hour: average departure delay for the same route during the prior clock hour.",
        "origin_delay_rate_prev_3h: rolling three-hour mean of departure-delay rate for the origin, computed from prior hours only.",
        "carrier_delay_rate_prev_3h: rolling three-hour mean of departure-delay rate for the carrier, computed from prior hours only.",
        "airport_carrier_avg_dep_delay_prior: cumulative average departure delay for the origin-carrier pair using only prior flights in time order.",
        "precip_peak_interaction: origin precipitation multiplied by a peak-hour flag to capture weather effects during heavy traffic windows.",
        "is_weekend, is_holiday, and season: calendar features derived from FlightDate.",
    ]
    return FinalFeatureBuildResult(frame=work, categorical_features=categorical_features, numeric_features=numeric_features, engineered_feature_notes=notes)


def temporal_train_test_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(frame["FlightDate"]).dt.normalize().dropna().unique().tolist())
    split_index = max(int(len(unique_dates) * (1 - test_fraction)), 1)
    split_date = pd.Timestamp(unique_dates[split_index])
    train = frame[frame["FlightDate"] < split_date].copy()
    test = frame[frame["FlightDate"] >= split_date].copy()
    LOGGER.info("Temporal split at %s produced %s train rows and %s test rows", split_date.date(), f"{len(train):,}", f"{len(test):,}")
    return train, test, split_date


def build_one_hot_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def build_ordinal_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
        ]
    )


def build_pca_logistic_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=0.95, svd_solver="full")),
                    ]
                ),
                numeric,
            ),
        ]
    )


def fit_numeric_pca_diagnostics(train_frame: pd.DataFrame, numeric_features: list[str]) -> tuple[np.ndarray, int, float]:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
        ]
    )
    numeric_pipeline.fit(train_frame[numeric_features])
    pca = numeric_pipeline.named_steps["pca"]
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    selected_components = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    explained_variance = float(cumulative_variance[selected_components - 1])
    return cumulative_variance, selected_components, explained_variance


def apply_log_transforms(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    transformed_train = train_frame.copy()
    transformed_test = test_frame.copy()
    created_features: list[str] = []
    for feature in numeric_features:
        train_series = pd.to_numeric(transformed_train[feature], errors="coerce")
        if train_series.dropna().empty:
            continue
        if train_series.min(skipna=True) < 0:
            continue
        if float(train_series.skew(skipna=True)) < SKEW_THRESHOLD:
            continue
        new_feature = f"log1p_{feature}"
        transformed_train[new_feature] = np.log1p(train_series)
        transformed_test[new_feature] = np.log1p(pd.to_numeric(transformed_test[feature], errors="coerce").clip(lower=0))
        created_features.append(new_feature)
    return transformed_train, transformed_test, created_features
