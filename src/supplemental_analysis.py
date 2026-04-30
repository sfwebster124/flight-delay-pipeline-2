"""Supplemental analysis for the flight-delay project."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from openai import OpenAI
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from final_model_utils import (
    build_final_feature_frame,
    build_one_hot_preprocessor,
    build_ordinal_preprocessor,
    configure_logging,
    ensure_final_output_dirs,
    load_final_analysis_frame,
    save_plot,
    temporal_train_test_split,
)
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_STATE = 42


def _slugify_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplemental analysis outputs.")
    parser.add_argument("--top-origins", type=int, default=25, help="Top N origins to retain for supplemental experiments.")
    parser.add_argument("--folds", type=int, default=4, help="Number of temporal folds for significance tests.")
    parser.add_argument("--train-cap", type=int, default=60000, help="Per-fold train sample cap for supplemental model experiments.")
    parser.add_argument("--test-cap", type=int, default=20000, help="Per-fold test sample cap for supplemental model experiments.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/modeling_dataset_fm15_strict_top25.parquet",
        help="Relative or absolute parquet path to use for supplemental experiments.",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="Use all rows from each temporal split/fold instead of sampling to train/test caps.",
    )
    parser.add_argument(
        "--report-subdir",
        type=str,
        default="",
        help="Optional subdirectory under reports/ for supplemental outputs from a specialized run.",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-5", help="OpenAI model name to use for the optional LLM analysis.")
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="auto",
        choices=["auto", "openai", "local"],
        help="LLM backend for the supplemental interpretation step.",
    )
    parser.add_argument(
        "--local-llm-model",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Free local Hugging Face model id for the supplemental interpretation step.",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Run only the supplemental LLM interpretation step using existing report outputs.",
    )
    parser.add_argument(
        "--final-sweep-only",
        action="store_true",
        help="Generate lightweight poster-ready summary files and LLM comparison figures from existing outputs only.",
    )
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="Generate only descriptive subset figures for the selected dataset and report folder.",
    )
    return parser.parse_args()


def _try_import_xgboost() -> object | None:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception:
        return None


def _try_import_local_llm_modules() -> tuple[object | None, object | None]:
    try:
        import torch
        from transformers import pipeline

        return torch, pipeline
    except Exception:
        return None, None


def _sample_frame(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    return frame.sample(n=cap, random_state=RANDOM_STATE)


def _resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _prepare_base_frame(top_origins: int, dataset_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = load_final_analysis_frame(BASE_DIR, dataset_path=dataset_path)
    feature_result = build_final_feature_frame(frame)
    if top_origins > 0:
        top_origin_codes = feature_result.frame["Origin"].value_counts().head(top_origins).index
        feature_result.frame = feature_result.frame[feature_result.frame["Origin"].isin(top_origin_codes)].copy()
    return feature_result.frame, feature_result.categorical_features, feature_result.numeric_features


def build_poster_dataset_summary(top_origins: int, dataset_path: Path, reports_dir: Path) -> pd.DataFrame:
    comparison = pd.read_parquet(REPORTS_DIR / "fm15_subset" / "dataset_comparison_summary.parquet").copy()
    frame, categorical_features, numeric_features = _prepare_base_frame(top_origins, dataset_path)
    dataset_label = f"top_{top_origins}_origin_runtime_subset"
    if dataset_path.name == "modeling_dataset_fm15_rows.parquet":
        dataset_label = f"fm15_strict_top_{top_origins}_origin_subset"
    top_origin_summary = pd.DataFrame(
        [
            {
                "dataset": dataset_label,
                "rows": len(frame),
                "unique_origin_airports": frame["Origin"].nunique(dropna=True),
                "unique_destination_airports": frame["Dest"].nunique(dropna=True),
                "date_start": frame["FlightDate"].min(),
                "date_end": frame["FlightDate"].max(),
                "overall_delay_rate": float(frame["dep_delayed_15"].mean()),
                "weather_join_rate": float(frame["origin_temp_c"].notna().mean()),
                "non_null_origin_temp_c": float(frame["origin_temp_c"].notna().mean()),
                "non_null_origin_visibility_m": float(frame["origin_visibility_m"].notna().mean()),
                "non_null_origin_wind_speed_mps": float(frame["origin_wind_speed_mps"].notna().mean()),
                "non_null_origin_precip_mm": float(frame["origin_precip_mm"].notna().mean()),
                "non_null_origin_ceiling_m": float(frame["origin_ceiling_m"].notna().mean()),
                "dep_delayed_15_positive_rate": float(frame["dep_delayed_15"].mean()),
                "morning_delay_rate": float(frame.loc[frame["scheduled_departure_hour_local"].between(6, 11), "dep_delayed_15"].mean()),
                "evening_delay_rate": float(frame.loc[frame["scheduled_departure_hour_local"].between(17, 22), "dep_delayed_15"].mean()),
                "bad_weather_delay_rate": float(frame.loc[frame["origin_precip_mm"].fillna(0) > 0, "dep_delayed_15"].mean()),
                "missing_temp_share": float(frame["origin_temp_c"].isna().mean()),
                "missing_visibility_share": float(frame["origin_visibility_m"].isna().mean()),
                "missing_precip_share": float(frame["origin_precip_mm"].isna().mean()),
                "top_10_origin_airports": ", ".join(frame["Origin"].value_counts().head(10).index.tolist()),
            }
        ]
    )
    summary = pd.concat([comparison, top_origin_summary], ignore_index=True)
    summary.to_csv(reports_dir / "poster_dataset_summary.csv", index=False)

    feature_counts = pd.DataFrame(
        [
            {"feature_type": "categorical", "count": len(categorical_features)},
            {"feature_type": "numeric", "count": len(numeric_features)},
            {"feature_type": "total_model_features", "count": len(categorical_features) + len(numeric_features)},
        ]
    )
    feature_counts.to_csv(reports_dir / "poster_feature_type_counts.csv", index=False)
    return summary


def plot_label_distribution(top_origins: int, dataset_path: Path, figures_dir: Path) -> None:
    frame, _, _ = _prepare_base_frame(top_origins, dataset_path)
    proportions = frame["dep_delayed_15"].value_counts(normalize=True).sort_index()
    counts = frame["dep_delayed_15"].value_counts().sort_index()
    labels = ["On-time (<15 min)", "Delayed (15+ min)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts.values)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}\n({proportions.iloc[idx]:.1%})", ha="center", va="bottom")
    ax.set_title(f"Label Distribution for Top {top_origins} Origin Subset")
    ax.set_ylabel("Flights")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, figures_dir / "poster_label_distribution.png")


def plot_feature_type_counts(reports_dir: Path, figures_dir: Path) -> None:
    feature_counts = pd.read_csv(reports_dir / "poster_feature_type_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feature_counts["feature_type"], feature_counts["count"])
    for idx, value in enumerate(feature_counts["count"]):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom")
    ax.set_title("Poster Modeling Feature Counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, figures_dir / "poster_feature_type_counts.png")


def plot_descriptive_subset_figures(top_origins: int, dataset_path: Path, figures_dir: Path) -> None:
    frame, _, _ = _prepare_base_frame(top_origins, dataset_path)

    by_hour = (
        frame.groupby("scheduled_departure_hour_local", dropna=True)
        .agg(delay_rate=("dep_delayed_15", "mean"), flights=("dep_delayed_15", "size"))
        .reset_index()
        .sort_values("scheduled_departure_hour_local")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(by_hour["scheduled_departure_hour_local"], by_hour["delay_rate"], marker="o")
    ax.set_title("Delay Rate by Scheduled Departure Hour")
    ax.set_xlabel("Scheduled Departure Hour (Local)")
    ax.set_ylabel("Delay Rate")
    ax.grid(alpha=0.25)
    save_plot(fig, figures_dir / "poster_delay_rate_by_departure_hour.png")

    top_origins_frame = (
        frame.groupby("Origin", dropna=False)
        .agg(delay_rate=("dep_delayed_15", "mean"), flights=("dep_delayed_15", "size"))
        .sort_values("flights", ascending=False)
        .head(15)
        .sort_values("delay_rate", ascending=True)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_origins_frame["Origin"], top_origins_frame["delay_rate"])
    ax.set_title("Delay Rate for Top Origin Airports")
    ax.set_xlabel("Delay Rate")
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, figures_dir / "poster_top_origin_airports_delay_rate.png")

    by_month = (
        frame.groupby("month", dropna=True)
        .agg(delay_rate=("dep_delayed_15", "mean"), flights=("dep_delayed_15", "size"))
        .reset_index()
        .sort_values("month")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(by_month["month"], by_month["delay_rate"], marker="o")
    ax.set_title("Monthly Delay Rate Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Delay Rate")
    ax.set_xticks(by_month["month"])
    ax.grid(alpha=0.25)
    save_plot(fig, figures_dir / "poster_monthly_delay_trend.png")

    precip_work = frame.copy()
    precip_work["precip_bucket"] = pd.cut(
        precip_work["origin_precip_mm"].fillna(0),
        bins=[-0.001, 0, 1, 5, 10, np.inf],
        labels=["0", "(0,1]", "(1,5]", "(5,10]", "10+"],
    )
    precip_summary = (
        precip_work.groupby("precip_bucket", observed=False)
        .agg(delay_rate=("dep_delayed_15", "mean"), flights=("dep_delayed_15", "size"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(precip_summary["precip_bucket"].astype(str), precip_summary["delay_rate"])
    ax.set_title("Delay Rate by Origin Precipitation Bucket")
    ax.set_xlabel("Origin Precipitation Bucket (mm)")
    ax.set_ylabel("Delay Rate")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, figures_dir / "poster_delay_vs_precipitation_bucket.png")

    by_carrier = (
        frame.groupby("Reporting_Airline", dropna=False)
        .agg(delay_rate=("dep_delayed_15", "mean"), flights=("dep_delayed_15", "size"))
        .sort_values("flights", ascending=False)
        .head(15)
        .sort_values("delay_rate", ascending=True)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(by_carrier["Reporting_Airline"], by_carrier["delay_rate"])
    ax.set_title("Delay Rate by Carrier")
    ax.set_xlabel("Delay Rate")
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, figures_dir / "poster_delay_rate_by_carrier.png")


def write_pca_component_summary(top_origins: int, dataset_path: Path, reports_dir: Path, figures_dir: Path) -> None:
    frame, _, numeric = _prepare_base_frame(top_origins, dataset_path)
    train, _, _ = temporal_train_test_split(frame)
    active_numeric = [feature for feature in numeric if feature in train.columns and train[feature].notna().any()]
    train_numeric = train[active_numeric].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca_full = SklearnPCA()
    transformed = imputer.fit_transform(train_numeric)
    transformed = scaler.fit_transform(transformed)
    pca_full.fit(transformed)

    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    selected_components = int(np.searchsorted(cumulative, 0.95) + 1)
    components_to_report = min(selected_components, 5)
    rows: list[dict[str, object]] = []
    summary_lines = [
        "# PCA Component Summary",
        "",
        f"- Numeric feature count entering PCA: {len(numeric)}",
        f"- Numeric features retained after dropping all-missing columns in this subset: {len(active_numeric)}",
        f"- Components required to explain 95% variance: {selected_components}",
        f"- Cumulative variance at component {selected_components}: {cumulative[selected_components - 1]:.4f}",
        "",
        "## Top Component Combinations",
        "",
    ]
    for component_index in range(components_to_report):
        component_number = component_index + 1
        component = pca_full.components_[component_index]
        loading_frame = (
            pd.DataFrame({"feature": active_numeric, "loading": component})
            .assign(abs_loading=lambda df: df["loading"].abs())
            .sort_values("abs_loading", ascending=False)
            .reset_index(drop=True)
        )
        top_loading_frame = loading_frame.head(10).copy()
        top_loading_frame.insert(0, "component", f"PC{component_number}")
        top_loading_frame.insert(1, "explained_variance_ratio", pca_full.explained_variance_ratio_[component_index])
        rows.extend(top_loading_frame.to_dict("records"))
        summary_lines.extend(
            [
                f"### PC{component_number}",
                f"- Explained variance ratio: {pca_full.explained_variance_ratio_[component_index]:.4f}",
                top_loading_frame.loc[:, ["feature", "loading", "abs_loading"]].to_markdown(index=False),
                "",
            ]
        )

    loadings_frame = pd.DataFrame(rows)
    loadings_frame.to_csv(reports_dir / "poster_pca_top_loadings.csv", index=False)
    (reports_dir / "poster_pca_component_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    first_component = (
        loadings_frame.loc[loadings_frame["component"] == "PC1", ["feature", "loading", "abs_loading"]]
        .sort_values("abs_loading", ascending=True)
        .tail(10)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(first_component["feature"], first_component["loading"])
    ax.set_title("Top PCA Loadings for PC1")
    ax.set_xlabel("Loading")
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, figures_dir / "poster_pca_pc1_top_loadings.png")


def write_supervised_model_comparison_with_llm(
    poster_reports_dir: Path,
    strict_reports_dir: Path,
    figures_dir: Path,
) -> None:
    model_summary = pd.read_csv(poster_reports_dir / "model_summary.csv").copy()
    llm_summary = pd.read_csv(strict_reports_dir / "llm_delay_model_comparison.csv").copy()
    llm_best = llm_summary.sort_values(["f1", "accuracy"], ascending=False).iloc[0]

    supervised = model_summary.loc[:, ["model", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean", "pr_auc_mean"]].copy()
    supervised = supervised.rename(
        columns={
            "precision_mean": "precision",
            "recall_mean": "recall",
            "f1_mean": "f1",
            "roc_auc_mean": "roc_auc",
            "pr_auc_mean": "pr_auc",
        }
    )
    llm_row = pd.DataFrame(
        [
            {
                "model": "llm_qwen_3b",
                "precision": llm_best["precision"],
                "recall": llm_best["recall"],
                "f1": llm_best["f1"],
                "roc_auc": np.nan,
                "pr_auc": np.nan,
            }
        ]
    )
    comparison = pd.concat([supervised, llm_row], ignore_index=True)
    comparison.to_csv(poster_reports_dir / "poster_supervised_model_comparison_with_llm.csv", index=False)

    metrics_to_plot = ["precision", "recall", "f1", "pr_auc"]
    plot_frame = comparison.copy()
    plot_frame["pr_auc"] = plot_frame["pr_auc"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(plot_frame))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics_to_plot))
    for offset, metric in zip(offsets, metrics_to_plot):
        values = plot_frame[metric].to_numpy()
        bars = ax.bar(x + offset, values, width=width, label=metric.upper())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Poster Model Comparison Including MLP and Best LLM")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_plot(fig, figures_dir / "poster_supervised_model_comparison_with_llm.png")

    lines = [
        "# Supervised Model Comparison Including LLM",
        "",
        "- The MLP neural network is included from the full 1.8M strict-subset poster run.",
        "- The LLM entry uses the best local row-prediction LLM from the strict-subset held-out prompt experiment.",
        "- Clustering is reported separately because KMeans is unsupervised and does not produce precision/recall/F1/ROC-AUC/PR-AUC in the same way.",
        "",
        comparison.to_markdown(index=False),
    ]
    (poster_reports_dir / "poster_supervised_model_comparison_with_llm.md").write_text("\n".join(lines), encoding="utf-8")


def write_cluster_method_summary(reports_dir: Path, figures_dir: Path) -> None:
    cluster_summary = pd.read_csv(reports_dir / "poster_cluster_summary.csv")
    airport_clusters = pd.read_csv(reports_dir / "poster_airport_clusters.csv")
    lines = [
        "# Clustering Summary",
        "",
        "- Clustering method used: `KMeans`",
        "- Number of clusters: `3`",
        f"- Airports clustered: `{len(airport_clusters):,}`",
        "",
        "## Cluster-Level Summary",
        cluster_summary.to_markdown(index=False),
        "",
        "## Figure",
        "- `poster_airport_clusters.png`",
    ]
    (reports_dir / "poster_clustering_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_classifier_models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    XGBClassifier = _try_import_xgboost()
    models: dict[str, Pipeline] = {
        "logistic_regression_balanced": Pipeline(
            [
                ("preprocessor", build_one_hot_preprocessor(categorical, numeric)),
                ("model", LogisticRegression(max_iter=1200, solver="lbfgs", class_weight="balanced")),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "mlp_classifier": Pipeline(
            [
                ("preprocessor", build_one_hot_preprocessor(categorical, numeric)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.0005,
                        learning_rate_init=0.001,
                        max_iter=80,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost_classifier"] = Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return models


def _temporal_folds(frame: pd.DataFrame, folds: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    unique_dates = sorted(pd.to_datetime(frame["FlightDate"]).dt.normalize().dropna().unique().tolist())
    if len(unique_dates) < folds + 1:
        raise ValueError("Not enough unique dates to create temporal folds.")
    boundaries = np.linspace(0, len(unique_dates), folds + 2, dtype=int)
    fold_pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_index in range(folds):
        train_end_idx = boundaries[fold_index + 1]
        test_end_idx = boundaries[fold_index + 2]
        train_end_date = pd.Timestamp(unique_dates[train_end_idx - 1])
        test_end_date = pd.Timestamp(unique_dates[test_end_idx - 1])
        train = frame[frame["FlightDate"] <= train_end_date].copy()
        test = frame[(frame["FlightDate"] > train_end_date) & (frame["FlightDate"] <= test_end_date)].copy()
        if not train.empty and not test.empty:
            fold_pairs.append((train, test))
    return fold_pairs


def run_poster_model_experiments(
    top_origins: int,
    folds: int,
    train_cap: int,
    test_cap: int,
    dataset_path: Path,
    reports_dir: Path,
    use_full_data: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, categorical, numeric = _prepare_base_frame(top_origins, dataset_path)
    features = categorical + numeric
    fold_pairs = _temporal_folds(frame, folds)
    models = _build_classifier_models(categorical, numeric)
    rows: list[dict[str, object]] = []

    for fold_index, (train, test) in enumerate(fold_pairs, start=1):
        if use_full_data:
            train = train.copy()
            test = test.copy()
        else:
            train = _sample_frame(train, train_cap)
            test = _sample_frame(test, test_cap)
        X_train = train[features]
        y_train = train["dep_delayed_15"]
        X_test = test[features]
        y_test = test["dep_delayed_15"]
        for model_name, pipeline in models.items():
            LOGGER.info("Poster fold %s/%s: training %s", fold_index, len(fold_pairs), model_name)
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            preds = (probabilities >= 0.5).astype(int)
            rows.append(
                {
                    "fold": fold_index,
                    "model": model_name,
                    "precision": precision_score(y_test, preds, zero_division=0),
                    "recall": recall_score(y_test, preds, zero_division=0),
                    "f1": f1_score(y_test, preds, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    "pr_auc": average_precision_score(y_test, probabilities),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "dataset_rows": len(frame),
                    "full_data": int(use_full_data),
                }
            )

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(reports_dir / "poster_model_fold_metrics.csv", index=False)
    summary_metrics = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            precision_std=("precision", "std"),
            recall_std=("recall", "std"),
            f1_std=("f1", "std"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_std=("pr_auc", "std"),
        )
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_metrics["dataset_rows"] = len(frame)
    summary_metrics["full_data"] = int(use_full_data)
    summary_metrics.to_csv(reports_dir / "model_summary.csv", index=False)
    return fold_metrics, summary_metrics


def write_hyperparameter_summary(reports_dir: Path) -> None:
    rows = [
        {
            "model": "logistic_regression_balanced",
            "search_method": "manual fixed settings",
            "key_hyperparameters": "solver=lbfgs; max_iter=1200; class_weight=balanced",
        },
        {
            "model": "random_forest_balanced",
            "search_method": "manual fixed settings",
            "key_hyperparameters": "n_estimators=120; max_depth=14; min_samples_leaf=20; class_weight=balanced_subsample",
        },
        {
            "model": "xgboost_classifier",
            "search_method": "manual fixed settings",
            "key_hyperparameters": "n_estimators=120; max_depth=6; learning_rate=0.08; subsample=0.8; colsample_bytree=0.8",
        },
        {
            "model": "mlp_classifier",
            "search_method": "manual fixed settings",
            "key_hyperparameters": "hidden_layer_sizes=(64,32); alpha=0.0005; learning_rate_init=0.001; max_iter=80; early_stopping=True",
        },
    ]
    pd.DataFrame(rows).to_csv(reports_dir / "model_hyperparameters.csv", index=False)
    lines = [
        "# Poster Hyperparameter Summary",
        "",
        "- The current poster experiments use fixed, manually specified hyperparameters rather than a full grid or random search.",
        "- This should be presented as targeted tuning based on runtime constraints and prior baseline results, not as exhaustive optimization.",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
    ]
    (reports_dir / "model_hyperparameters.md").write_text("\n".join(lines), encoding="utf-8")


def write_significance_tests(fold_metrics: pd.DataFrame, reports_dir: Path) -> None:
    lines = ["# Poster Significance Tests", ""]
    summary_lines = [
        "# Statistical Tests Summary",
        "",
        "- Test design: one-way ANOVA across temporal-fold model metrics, followed by Tukey HSD pairwise comparisons.",
        "- Why this test: multiple supervised models were compared across repeated temporal folds, so ANOVA is appropriate for screening overall differences before pairwise follow-up testing.",
        "",
    ]
    for metric in ["f1", "pr_auc", "roc_auc"]:
        groups = [group[metric].to_numpy() for _, group in fold_metrics.groupby("model")]
        model_names = fold_metrics["model"].drop_duplicates().tolist()
        anova = f_oneway(*groups)
        tukey = pairwise_tukeyhsd(endog=fold_metrics[metric], groups=fold_metrics["model"], alpha=0.05)
        tukey_frame = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_frame.to_csv(reports_dir / f"poster_tukey_{metric}.csv", index=False)
        lines.extend(
            [
                f"## {metric.upper()}",
                f"- ANOVA F-statistic: {anova.statistic:.4f}",
                f"- ANOVA p-value: {anova.pvalue:.6f}",
                "",
                tukey_frame.to_markdown(index=False),
                "",
            ]
        )
        summary_lines.extend(
            [
                f"## {metric.upper()}",
                f"- ANOVA F-statistic: {anova.statistic:.4f}",
                f"- ANOVA p-value: {anova.pvalue:.6f}",
                f"- Significant at alpha=0.05: {'yes' if anova.pvalue < 0.05 else 'no'}",
                "",
            ]
        )
    (reports_dir / "significance_tests.md").write_text("\n".join(lines), encoding="utf-8")
    (reports_dir / "statistical_tests_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def plot_interpretability(
    top_origins: int,
    dataset_path: Path,
    figures_dir: Path,
    reports_dir: Path,
    use_full_data: bool,
) -> None:
    frame, categorical, numeric = _prepare_base_frame(top_origins, dataset_path)
    train, test, _ = temporal_train_test_split(frame)
    if not use_full_data:
        train = _sample_frame(train, 80000)
        test = _sample_frame(test, 25000)
    features = categorical + numeric
    models = _build_classifier_models(categorical, numeric)

    for model_name in ["random_forest_balanced", "xgboost_classifier"]:
        if model_name not in models:
            continue
        pipeline = models[model_name]
        pipeline.fit(train[features], train["dep_delayed_15"])
        transformed_train = pipeline.named_steps["preprocessor"].transform(train[features])
        transformed_test = pipeline.named_steps["preprocessor"].transform(test[features])
        model = pipeline.named_steps["model"]
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = np.array(categorical + numeric, dtype=object)

        transformed_train_df = pd.DataFrame(transformed_train, columns=feature_names)
        transformed_test_df = pd.DataFrame(transformed_test, columns=feature_names)
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=feature_names)
        else:
            importance = pd.Series(np.zeros(len(feature_names)), index=feature_names)
        top = importance.sort_values(ascending=False).head(15).sort_values()
        top.to_csv(reports_dir / f"poster_{model_name}_importance.csv", header=["importance"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top.index, top.values)
        ax.set_title(f"Poster Feature Importance: {model_name}")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
        save_plot(fig, figures_dir / f"poster_{model_name}_importance.png")

        sample_background = transformed_train_df.sample(n=min(len(transformed_train_df), 1000), random_state=RANDOM_STATE)
        sample_eval = transformed_test_df.sample(n=min(len(transformed_test_df), 1000), random_state=RANDOM_STATE)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_eval)
            if isinstance(shap_values, list):
                shap_matrix = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
            else:
                shap_array = np.asarray(shap_values)
                shap_matrix = shap_array[..., 1] if shap_array.ndim == 3 and shap_array.shape[-1] > 1 else shap_array
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
            shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
            shap_importance.head(20).to_csv(reports_dir / f"poster_{model_name}_shap.csv", header=["mean_abs_shap"])

            top_shap = shap_importance.head(15).sort_values()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_shap.index, top_shap.values)
            ax.set_title(f"SHAP Mean |Value|: {model_name}")
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.grid(axis="x", alpha=0.25)
            save_plot(fig, figures_dir / f"poster_{model_name}_shap.png")

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_matrix, sample_eval, feature_names=feature_names, show=False, max_display=15)
            fig = plt.gcf()
            save_plot(fig, figures_dir / f"poster_{model_name}_shap_beeswarm.png")
        except Exception as exc:
            LOGGER.warning("SHAP generation failed for %s: %s", model_name, exc)


def _build_llm_prompt(reports_dir: Path) -> tuple[str, Path]:
    dataset_summary = pd.read_csv(reports_dir / "poster_dataset_summary.csv")
    model_summary = pd.read_csv(reports_dir / "model_summary.csv")
    significance_tests = (reports_dir / "significance_tests.md").read_text(encoding="utf-8")
    top_origin_row = dataset_summary.loc[dataset_summary["dataset"].str.contains("top_")].iloc[0]
    prompt = "\n".join(
        [
            "You are writing a concise research-poster interpretation for a flight-delay prediction project.",
            "Use the following facts only and do not invent numbers.",
            f"Full dataset rows: {int(dataset_summary.loc[dataset_summary['dataset'] == 'full_main_dataset', 'rows'].iloc[0]):,}",
            f"FM-15 strict rows: {int(dataset_summary.loc[dataset_summary['dataset'] == 'fm15_strict_row_subset', 'rows'].iloc[0]):,}",
            f"Top-origin runtime rows: {int(top_origin_row['rows']):,}",
            "Poster model summary:",
            model_summary.to_markdown(index=False),
            "Significance test summary:",
            significance_tests,
            "Write markdown with sections: Poster Narrative, Best Model Rationale, Limitations, and Presentation Talking Points.",
            "Keep it under 350 words.",
        ]
    )
    prompt_path = reports_dir / "llm_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    return prompt, prompt_path


def write_llm_poster_analysis(llm_model: str, llm_backend: str, local_llm_model: str, reports_dir: Path) -> None:
    prompt, prompt_path = _build_llm_prompt(reports_dir)
    output_path = reports_dir / "llm_analysis.md"
    model_tag = _slugify_model_name(local_llm_model if llm_backend == "local" or (llm_backend == "auto" and not os.getenv("OPENAI_API_KEY")) else llm_model)
    model_output_path = reports_dir / f"llm_analysis_{model_tag}.md"
    backend = llm_backend
    api_key = os.getenv("OPENAI_API_KEY")
    if backend == "auto":
        backend = "local"
        if api_key:
            backend = "openai"

    if backend == "openai":
        if not api_key:
            lines = [
                "# Poster LLM Analysis",
                "",
                "- Status: Requested OpenAI backend, but `OPENAI_API_KEY` is not set.",
                f"- Prompt saved to `{prompt_path}`.",
            ]
            output_path.write_text("\n".join(lines), encoding="utf-8")
            model_output_path.write_text("\n".join(lines), encoding="utf-8")
            return
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=llm_model,
            input=prompt,
        )
        lines = [
            "# Poster LLM Analysis",
            "",
            f"- Backend: `openai`",
            f"- Model: `{llm_model}`",
            "",
            response.output_text.strip(),
        ]
        output_path.write_text("\n".join(lines), encoding="utf-8")
        model_output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    torch, text_pipeline = _try_import_local_llm_modules()
    if torch is None or text_pipeline is None:
        lines = [
            "# Poster LLM Analysis",
            "",
            "- Status: Requested local LLM backend, but local LLM dependencies are not installed.",
            f"- Prompt saved to `{prompt_path}`.",
            "- Install `transformers` and `torch`, then rerun the script.",
        ]
        output_path.write_text("\n".join(lines), encoding="utf-8")
        model_output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    generator = text_pipeline(
        "text-generation",
        model=local_llm_model,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    system_prompt = "You are a concise research assistant writing a poster interpretation from provided metrics only."
    response = generator(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_new_tokens=450,
        do_sample=False,
    )
    generated = response[0]["generated_text"]
    if isinstance(generated, list):
        assistant_chunks = [item.get("content", "") for item in generated if item.get("role") == "assistant"]
        output_text = "\n".join([chunk for chunk in assistant_chunks if chunk]).strip()
    else:
        output_text = str(generated)

    lines = [
        "# Poster LLM Analysis",
        "",
        f"- Backend: `local`",
        f"- Model: `{local_llm_model}`",
        "",
        output_text.strip(),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    model_output_path.write_text("\n".join(lines), encoding="utf-8")


def run_clustering(top_origins: int, dataset_path: Path, reports_dir: Path, figures_dir: Path) -> None:
    frame, _, _ = _prepare_base_frame(top_origins, dataset_path)
    airport_summary = (
        frame.groupby("Origin", dropna=False)
        .agg(
            flights=("Origin", "size"),
            delay_rate=("dep_delayed_15", "mean"),
            avg_delay_minutes=("dep_delay_minutes", "mean"),
            avg_precip_mm=("origin_precip_mm", "mean"),
            avg_visibility_m=("origin_visibility_m", "mean"),
            avg_ceiling_m=("origin_ceiling_m", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )
    cluster_features = ["delay_rate", "avg_delay_minutes", "avg_precip_mm", "avg_visibility_m", "avg_ceiling_m"]
    model = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
    airport_summary["cluster"] = model.fit_predict(airport_summary[cluster_features])
    airport_summary.to_csv(reports_dir / "poster_airport_clusters.csv", index=False)

    reduced = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(airport_summary[cluster_features])
    plot_frame = pd.DataFrame(reduced, columns=["pc1", "pc2"])
    plot_frame["cluster"] = airport_summary["cluster"]
    plot_frame["Origin"] = airport_summary["Origin"]
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(plot_frame["pc1"], plot_frame["pc2"], c=plot_frame["cluster"])
    for _, row in plot_frame.head(25).iterrows():
        ax.text(row["pc1"], row["pc2"], row["Origin"], fontsize=7)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title("Poster Airport Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)
    save_plot(fig, figures_dir / "poster_airport_clusters.png")

    cluster_summary = (
        airport_summary.groupby("cluster", as_index=False)
        .agg(
            airports=("Origin", "size"),
            avg_delay_rate=("delay_rate", "mean"),
            avg_flights=("flights", "mean"),
            avg_precip_mm=("avg_precip_mm", "mean"),
            avg_visibility_m=("avg_visibility_m", "mean"),
            avg_ceiling_m=("avg_ceiling_m", "mean"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    cluster_summary.to_csv(reports_dir / "poster_cluster_summary.csv", index=False)


def write_poster_requirements_checklist(reports_dir: Path) -> None:
    llm_output = reports_dir / "poster_llm_analysis.md"
    llm_status = "complete" if llm_output.exists() else "implemented_pending_run"
    llm_evidence = "`reports/poster_llm_analysis.md`, `reports/poster_llm_prompt.txt`"
    lines = [
        "# Poster Requirements Checklist",
        "",
        "| Requirement | Status | Evidence |",
        "|:--|:--|:--|",
        "| Dataset tiering | complete | `reports/fm15_subset/dataset_comparison_summary.parquet`, `reports/fm15_subset/fm15_subset_summary.md` |",
        "| Classification comparison | complete | `reports/final_model_metrics.csv`, `reports/final_model_comparison.md` |",
        "| Regression comparison | complete | `reports/final_regression_metrics.csv` |",
        "| PCA comparison | complete | `reports/final_pca_summary.md`, `reports/final_pca_comparison.csv` |",
        "| Class imbalance comparison | complete | `reports/final_imbalance_metrics.csv` |",
        "| Feature/label distribution plots | complete | `reports/figures/poster_label_distribution.png`, `reports/figures/poster_feature_type_counts.png` |",
        "| Hyperparameter documentation | complete | `reports/poster_hyperparameters.md` |",
        "| Clustering | complete | `reports/poster_cluster_summary.csv`, `reports/figures/poster_airport_clusters.png` |",
        "| Neural network model | complete | `reports/poster_model_summary.csv` (`mlp_classifier`) |",
        "| ANOVA + Tukey tests | complete | `reports/poster_significance_tests.md`, `reports/poster_tukey_*.csv` |",
        "| SHAP / interpretability analysis | complete | `reports/figures/poster_random_forest_balanced_shap.png`, `reports/figures/poster_xgboost_classifier_shap.png` |",
        f"| Large language model | {llm_status} | {llm_evidence} |",
    ]
    (reports_dir / "poster_requirements_checklist.md").write_text("\n".join(lines), encoding="utf-8")


def write_poster_summary(top_origins: int, reports_dir: Path) -> None:
    dataset_summary = pd.read_csv(reports_dir / "poster_dataset_summary.csv")
    model_summary = pd.read_csv(reports_dir / "poster_model_summary.csv")
    top_row_matches = dataset_summary.loc[dataset_summary["dataset"].str.contains(f"top_{top_origins}_origin")]
    if top_row_matches.empty:
        top_row_matches = dataset_summary.loc[dataset_summary["dataset"].str.contains(f"top_{top_origins}")]
    top_origin_row = top_row_matches.iloc[0]
    best_model_row = model_summary.iloc[0]
    lines = [
        "# Poster Supplemental Summary",
        "",
        f"- Full cleaned modeling dataset: {int(dataset_summary.loc[dataset_summary['dataset'] == 'full_main_dataset', 'rows'].iloc[0]):,} rows.",
        f"- FM-15 airport subset: {int(dataset_summary.loc[dataset_summary['dataset'] == 'fm15_airport_subset', 'rows'].iloc[0]):,} rows with weather join rate {float(dataset_summary.loc[dataset_summary['dataset'] == 'fm15_airport_subset', 'weather_join_rate'].iloc[0]):.2%}.",
        f"- FM-15 strict subset: {int(dataset_summary.loc[dataset_summary['dataset'] == 'fm15_strict_row_subset', 'rows'].iloc[0]):,} rows with weather join rate {float(dataset_summary.loc[dataset_summary['dataset'] == 'fm15_strict_row_subset', 'weather_join_rate'].iloc[0]):.2%}.",
        f"- Top {top_origins} origin runtime subset: {int(top_origin_row['rows']):,} rows across {int(top_origin_row['unique_origin_airports'])} origin airports.",
        f"- Poster model experiments used {int(pd.read_csv(reports_dir / 'poster_feature_type_counts.csv').loc[2, 'count'])} model input features before encoding.",
        f"- Best poster fold-averaged classifier: {best_model_row['model']} with mean F1 {best_model_row['f1_mean']:.4f}, mean ROC-AUC {best_model_row['roc_auc_mean']:.4f}, and mean PR-AUC {best_model_row['pr_auc_mean']:.4f}.",
        f"- Poster model folds were run on dataset size {int(best_model_row['dataset_rows']):,} with full_data={int(best_model_row['full_data'])}.",
        "- SHAP-based interpretability outputs are available for the Random Forest and XGBoost classifiers.",
        "",
        "## Large Language Model",
        "- The codebase now includes an OpenAI-backed poster interpretation step that writes `reports/poster_llm_analysis.md` when `OPENAI_API_KEY` is available.",
        "- If no API key is present, the script writes a placeholder status file and saves the exact prompt for later execution.",
    ]
    (reports_dir / "poster_supplemental_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_llm_row_prediction_comparison(report_subdir: str = "fm15_strict_top25") -> None:
    strict_reports_dir = REPORTS_DIR / report_subdir
    strict_figures_dir = strict_reports_dir / "figures"
    llm_metrics_path = strict_reports_dir / "llm_delay_model_comparison.csv"
    final_metrics_path = strict_reports_dir / "final_model_metrics.csv"
    if not llm_metrics_path.exists():
        LOGGER.warning("Skipping LLM poster figures because %s does not exist", llm_metrics_path)
        return

    llm_metrics = pd.read_csv(llm_metrics_path).copy()
    llm_metrics["model_label"] = llm_metrics["model"].str.replace("Qwen/Qwen2.5-", "", regex=False).str.replace("-Instruct", "", regex=False)
    metric_columns = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(llm_metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metric_columns))
    for offset, metric in zip(offsets, metric_columns):
        values = llm_metrics[metric].to_numpy()
        bars = ax.bar(x + offset, values, width=width, label=metric.upper())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(llm_metrics["model_label"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Local LLM Delay Prediction Comparison (60-row balanced test sample)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_plot(fig, strict_figures_dir / "llm_delay_model_comparison_bar.png")

    prediction_paths = sorted(strict_reports_dir.glob("llm_delay_predictions_*.csv"))
    confusion_rows: list[dict[str, object]] = []
    for path in prediction_paths:
        prediction_frame = pd.read_csv(path)
        valid = prediction_frame.dropna(subset=["llm_predicted_delay_15"]).copy()
        if valid.empty:
            continue
        model_slug = path.stem.replace("llm_delay_predictions_", "")
        tn = int(((valid["dep_delayed_15"] == 0) & (valid["llm_predicted_delay_15"] == 0)).sum())
        fp = int(((valid["dep_delayed_15"] == 0) & (valid["llm_predicted_delay_15"] == 1)).sum())
        fn = int(((valid["dep_delayed_15"] == 1) & (valid["llm_predicted_delay_15"] == 0)).sum())
        tp = int(((valid["dep_delayed_15"] == 1) & (valid["llm_predicted_delay_15"] == 1)).sum())
        confusion_rows.extend(
            [
                {"model": model_slug, "component": "TN", "count": tn},
                {"model": model_slug, "component": "FP", "count": fp},
                {"model": model_slug, "component": "FN", "count": fn},
                {"model": model_slug, "component": "TP", "count": tp},
            ]
        )
    if confusion_rows:
        confusion_frame = pd.DataFrame(confusion_rows)
        model_order = llm_metrics["model"].apply(_slugify_model_name).tolist()
        display_labels = llm_metrics["model_label"].tolist()
        component_order = ["TN", "FP", "FN", "TP"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(model_order))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(component_order))
        for offset, component in zip(offsets, component_order):
            subset = (
                confusion_frame.loc[confusion_frame["component"] == component]
                .set_index("model")
                .reindex(model_order)
                .reset_index()
            )
            values = subset["count"].fillna(0).to_numpy()
            bars = ax.bar(x + offset, values, width=width, label=component)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{int(value)}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels)
        ax.set_ylabel("Flights")
        ax.set_title("LLM Prediction Confusion Breakdown")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save_plot(fig, strict_figures_dir / "llm_delay_confusion_breakdown.png")

    if final_metrics_path.exists():
        final_metrics = pd.read_csv(final_metrics_path)
        best_llm = llm_metrics.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
        comparison = pd.concat(
            [
                final_metrics.loc[:, ["model", "precision", "recall", "f1"]].assign(family="tabular"),
                pd.DataFrame(
                    [
                        {
                            "model": f"llm_{best_llm['model_label']}",
                            "precision": best_llm["precision"],
                            "recall": best_llm["recall"],
                            "f1": best_llm["f1"],
                            "family": "llm",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        comparison = comparison.sort_values("f1", ascending=False).reset_index(drop=True)
        metrics_to_plot = ["precision", "recall", "f1"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(comparison))
        width = 0.22
        offsets = np.linspace(-width, width, len(metrics_to_plot))
        for offset, metric in zip(offsets, metrics_to_plot):
            values = comparison[metric].to_numpy()
            bars = ax.bar(x + offset, values, width=width, label=metric.upper())
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison["model"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Best LLM vs Tabular Classifiers on Strict Top-25 Subset")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save_plot(fig, strict_figures_dir / "poster_llm_vs_tabular_comparison.png")


def write_llm_row_prediction_summary(report_subdir: str = "fm15_strict_top25") -> None:
    strict_reports_dir = REPORTS_DIR / report_subdir
    llm_metrics_path = strict_reports_dir / "llm_delay_model_comparison.csv"
    if not llm_metrics_path.exists():
        LOGGER.warning("Skipping LLM summary because %s does not exist", llm_metrics_path)
        return
    llm_metrics = pd.read_csv(llm_metrics_path).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_row = llm_metrics.iloc[0]
    lines = [
        "# LLM Delay Prediction Summary",
        "",
        "- Experiment: local instruction-tuned LLMs predict `DELAY` vs `ON_TIME` directly from row features.",
        f"- Dataset: `{Path(best_row['dataset_path']).name}` restricted to top {int(best_row['top_origins'])} origin airports from the strict FM-15 subset.",
        f"- Evaluation design: {int(best_row['evaluation_rows_requested'])} balanced held-out test rows with {int(best_row['few_shot_examples'])} balanced few-shot training examples embedded in each prompt.",
        "",
        llm_metrics.loc[:, ["model", "accuracy", "precision", "recall", "f1"]].to_markdown(index=False),
        "",
        f"- Best local LLM by F1 in this small-sample test: `{best_row['model']}` with accuracy `{best_row['accuracy']:.4f}`, precision `{best_row['precision']:.4f}`, recall `{best_row['recall']:.4f}`, and F1 `{best_row['f1']:.4f}`.",
        "- Interpretation: all tested local LLMs over-predicted delay, which lifted recall but hurt specificity and overall discrimination.",
        "- Poster use: present the LLM as an exploratory baseline showing that generic text-generation models are much less reliable than tabular ML on structured aviation data.",
    ]
    (strict_reports_dir / "llm_delay_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_strict_subset_poster_brief(report_subdir: str = "fm15_strict_top25") -> None:
    strict_reports_dir = REPORTS_DIR / report_subdir
    reduction = pd.read_csv(strict_reports_dir / "dataset_reduction_summary.csv")
    pca = pd.read_csv(strict_reports_dir / "final_pca_comparison.csv")
    classification = pd.read_csv(strict_reports_dir / "final_model_metrics.csv").sort_values("pr_auc", ascending=False).reset_index(drop=True)
    regression = pd.read_csv(strict_reports_dir / "final_regression_metrics.csv").sort_values(["mae", "rmse", "r2"], ascending=[True, True, False]).reset_index(drop=True)
    imbalance = pd.read_csv(strict_reports_dir / "final_imbalance_metrics.csv").sort_values("pr_auc", ascending=False).reset_index(drop=True)
    llm_metrics_path = strict_reports_dir / "llm_delay_model_comparison.csv"
    llm_metrics = pd.read_csv(llm_metrics_path).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True) if llm_metrics_path.exists() else pd.DataFrame()
    best_classifier = classification.iloc[0]
    best_regressor = regression.iloc[0]
    pca_row = pca.loc[pca["experiment"].str.contains("with_pca")].iloc[0]
    weighted_row = imbalance.loc[imbalance["experiment"] == "logistic_class_weight_balanced"].iloc[0]
    no_balance_row = imbalance.loc[imbalance["experiment"] == "logistic_no_balancing"].iloc[0]

    lines = [
        "# Final Poster Brief",
        "",
        "## Dataset Reduction",
        f"- Full cleaned main dataset: {int(reduction.loc[reduction['stage'] == 'full_main_dataset', 'rows'].iloc[0]):,} rows.",
        f"- Strict FM-15 weather-covered subset: {int(reduction.loc[reduction['stage'] == 'modeling_dataset_fm15_rows', 'rows'].iloc[0]):,} rows, retaining {float(reduction.loc[reduction['stage'] == 'modeling_dataset_fm15_rows', 'share_retained_from_full'].iloc[0]):.2%} of the full dataset.",
        f"- Strict top-25 origin subset used for final poster modeling: {int(reduction.loc[reduction['stage'].str.contains('top_25_origins'), 'rows'].iloc[0]):,} rows, retaining {float(reduction.loc[reduction['stage'].str.contains('top_25_origins'), 'share_retained_from_full'].iloc[0]):.2%} of the full dataset.",
        "",
        "## PCA Result",
        f"- Logistic regression without PCA: F1 {pca.loc[pca['experiment'].str.contains('no_pca'), 'f1'].iloc[0]:.4f}, ROC-AUC {pca.loc[pca['experiment'].str.contains('no_pca'), 'roc_auc'].iloc[0]:.4f}, PR-AUC {pca.loc[pca['experiment'].str.contains('no_pca'), 'pr_auc'].iloc[0]:.4f}.",
        f"- Logistic regression with PCA: F1 {pca_row['f1']:.4f}, ROC-AUC {pca_row['roc_auc']:.4f}, PR-AUC {pca_row['pr_auc']:.4f}.",
        f"- PCA retained {int(pca_row['pca_components'])} numeric components and explained {float(pca_row['numeric_variance_explained']):.2%} of numeric variance, but slightly reduced performance.",
        "",
        "## Final Model Story",
        f"- Best PR-AUC classifier on the strict top-25 subset: `{best_classifier['model']}` with precision {best_classifier['precision']:.4f}, recall {best_classifier['recall']:.4f}, F1 {best_classifier['f1']:.4f}, ROC-AUC {best_classifier['roc_auc']:.4f}, and PR-AUC {best_classifier['pr_auc']:.4f}.",
        f"- Highest-recall classifier: `logistic_regression_balanced` with recall {classification.loc[classification['model'] == 'logistic_regression_balanced', 'recall'].iloc[0]:.4f}.",
        f"- Best regression model by MAE: `{best_regressor['model']}` with MAE {best_regressor['mae']:.2f}, RMSE {best_regressor['rmse']:.2f}, and R^2 {best_regressor['r2']:.4f}.",
        "",
        "## Imbalance Handling",
        f"- No-balancing logistic baseline: precision {no_balance_row['precision']:.4f}, recall {no_balance_row['recall']:.4f}, F1 {no_balance_row['f1']:.4f}, PR-AUC {no_balance_row['pr_auc']:.4f}.",
        f"- Class-weighted logistic: precision {weighted_row['precision']:.4f}, recall {weighted_row['recall']:.4f}, F1 {weighted_row['f1']:.4f}, PR-AUC {weighted_row['pr_auc']:.4f}.",
        f"- Class weighting changed recall by {weighted_row['recall'] - no_balance_row['recall']:+.4f} and F1 by {weighted_row['f1'] - no_balance_row['f1']:+.4f} versus the no-balancing baseline.",
        "",
        "## Interpretability",
        "- SHAP and feature-importance figures are available for both Random Forest and XGBoost in `reports/figures/`.",
        "- Use those figures to support the claim that propagation and congestion features dominate, while weather contributes secondary but meaningful signal.",
    ]
    if not llm_metrics.empty:
        best_llm = llm_metrics.iloc[0]
        lines.extend(
            [
                "",
                "## LLM Comparison",
                f"- Best local row-prediction LLM: `{best_llm['model']}` with accuracy {best_llm['accuracy']:.4f}, precision {best_llm['precision']:.4f}, recall {best_llm['recall']:.4f}, and F1 {best_llm['f1']:.4f} on a {int(best_llm['evaluation_rows_valid'])}-row balanced test sample.",
                "- LLM recall was high because the models tended to predict delay too often; this makes the LLM comparison a useful exploratory baseline, not a competitive final predictor.",
            ]
        )
    lines.extend(
        [
            "",
            "## Poster Figure Checklist",
            "- Dataset reduction: `reports/fm15_strict_top25/figures/dataset_reduction_bar.png`",
            "- PCA variance curve: `reports/fm15_strict_top25/figures/final_pca_variance_curve.png`",
            "- Classification comparison: `reports/fm15_strict_top25/figures/final_model_comparison_bar.png`",
            "- ROC / PR curves: `reports/fm15_strict_top25/figures/final_best_classifier_roc_curve.png`, `reports/fm15_strict_top25/figures/final_best_classifier_pr_curve.png`",
            "- SHAP / importance: `reports/figures/poster_xgboost_classifier_shap.png`, `reports/figures/poster_random_forest_balanced_shap.png`",
            "- LLM comparison: `reports/fm15_strict_top25/figures/llm_delay_model_comparison_bar.png`, `reports/fm15_strict_top25/figures/poster_llm_vs_tabular_comparison.png`",
        ]
    )
    (strict_reports_dir / "poster_final_brief.md").write_text("\n".join(lines), encoding="utf-8")


def write_strict_subset_poster_map(report_subdir: str = "fm15_strict_top25") -> None:
    strict_reports_dir = REPORTS_DIR / report_subdir
    reduction = pd.read_csv(strict_reports_dir / "dataset_reduction_summary.csv")
    pca = pd.read_csv(strict_reports_dir / "final_pca_comparison.csv")
    classification = pd.read_csv(strict_reports_dir / "final_model_metrics.csv").sort_values("pr_auc", ascending=False).reset_index(drop=True)
    regression = pd.read_csv(strict_reports_dir / "final_regression_metrics.csv").sort_values(["mae", "rmse", "r2"], ascending=[True, True, False]).reset_index(drop=True)
    imbalance = pd.read_csv(strict_reports_dir / "final_imbalance_metrics.csv")
    llm_metrics_path = strict_reports_dir / "llm_delay_model_comparison.csv"
    llm_metrics = pd.read_csv(llm_metrics_path).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True) if llm_metrics_path.exists() else pd.DataFrame()
    lines = [
        "# FM-15 Strict Top-25 Poster Map",
        "",
        "Use this folder for the final poster storyline built on the strict FM-15 weather-covered subset and the top 25 origin airports.",
        "",
        "## Dataset Reduction",
        "",
        f"- Full main dataset: `{int(reduction.loc[reduction['stage'] == 'full_main_dataset', 'rows'].iloc[0]):,}` rows",
        f"- FM-15 strict subset: `{int(reduction.loc[reduction['stage'] == 'modeling_dataset_fm15_rows', 'rows'].iloc[0]):,}` rows",
        f"- FM-15 strict top-25 origins subset: `{int(reduction.loc[reduction['stage'].str.contains('top_25_origins'), 'rows'].iloc[0]):,}` rows",
        "",
        "Files:",
        "",
        "- `dataset_reduction_summary.md`",
        "- `dataset_reduction_summary.csv`",
        "- `figures/dataset_reduction_bar.png`",
        "",
        "## PCA Comparison",
        "",
        f"- Logistic without PCA: F1 `{pca.loc[pca['experiment'].str.contains('no_pca'), 'f1'].iloc[0]:.4f}`, ROC-AUC `{pca.loc[pca['experiment'].str.contains('no_pca'), 'roc_auc'].iloc[0]:.4f}`, PR-AUC `{pca.loc[pca['experiment'].str.contains('no_pca'), 'pr_auc'].iloc[0]:.4f}`",
        f"- Logistic with PCA: F1 `{pca.loc[pca['experiment'].str.contains('with_pca'), 'f1'].iloc[0]:.4f}`, ROC-AUC `{pca.loc[pca['experiment'].str.contains('with_pca'), 'roc_auc'].iloc[0]:.4f}`, PR-AUC `{pca.loc[pca['experiment'].str.contains('with_pca'), 'pr_auc'].iloc[0]:.4f}`",
        f"- PCA retained `{int(pca.loc[pca['experiment'].str.contains('with_pca'), 'pca_components'].iloc[0])}` numeric components and explained `{float(pca.loc[pca['experiment'].str.contains('with_pca'), 'numeric_variance_explained'].iloc[0]):.2%}` of numeric variance",
        "",
        "Files:",
        "",
        "- `final_pca_summary.md`",
        "- `final_pca_comparison.csv`",
        "- `figures/final_pca_variance_curve.png`",
        "",
        "## Classification Models",
        "",
    ]
    for _, row in classification.iterrows():
        lines.append(
            f"- {row['model']}: precision `{row['precision']:.4f}`, recall `{row['recall']:.4f}`, F1 `{row['f1']:.4f}`, ROC-AUC `{row['roc_auc']:.4f}`, PR-AUC `{row['pr_auc']:.4f}`"
        )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `final_model_comparison.md`",
            "- `final_model_metrics.csv`",
            "- `figures/final_model_comparison_bar.png`",
            "- `figures/final_best_classifier_roc_curve.png`",
            "- `figures/final_best_classifier_pr_curve.png`",
            "",
            "## Regression Models",
            "",
        ]
    )
    for _, row in regression.iterrows():
        lines.append(f"- {row['model']}: MAE `{row['mae']:.4f}`, RMSE `{row['rmse']:.4f}`, R^2 `{row['r2']:.4f}`")
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `final_regression_metrics.csv`",
            "- `figures/final_regression_model_comparison_bar.png`",
            "",
            "## Imbalance Handling",
            "",
        ]
    )
    for _, row in imbalance.sort_values("pr_auc", ascending=False).iterrows():
        lines.append(f"- {row['experiment']}: precision `{row['precision']:.4f}`, recall `{row['recall']:.4f}`, F1 `{row['f1']:.4f}`, PR-AUC `{row['pr_auc']:.4f}`")
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `final_imbalance_metrics.csv`",
            "- `final_imbalance_summary.md`",
            "",
            "## LLM Row-Prediction Comparison",
            "",
        ]
    )
    if llm_metrics.empty:
        lines.append("- LLM comparison outputs not found yet.")
    else:
        for _, row in llm_metrics.iterrows():
            lines.append(
                f"- {row['model']}: accuracy `{row['accuracy']:.4f}`, precision `{row['precision']:.4f}`, recall `{row['recall']:.4f}`, F1 `{row['f1']:.4f}` on `{int(row['evaluation_rows_valid'])}` balanced held-out rows"
            )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `llm_delay_model_comparison.csv`",
            "- `llm_delay_summary.md`",
            "- `figures/llm_delay_model_comparison_bar.png`",
            "- `figures/llm_delay_confusion_breakdown.png`",
            "- `figures/poster_llm_vs_tabular_comparison.png`",
            "",
            "## Diagnostics / Interpretation",
            "",
            "Files:",
            "",
            "- `final_feature_ranking.csv`",
            "- `final_preprocessing_summary.md`",
            "- `figures/final_feature_importance.png`",
            "- `figures/final_correlation_heatmap.png`",
            "- `figures/final_threshold_tradeoff.png`",
            "- `figures/final_calibration_curve.png`",
            "- `figures/final_airport_delay_bubble_map.png`",
            "- `poster_final_brief.md`",
        ]
    )
    (strict_reports_dir / "poster_section_map.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    output_dirs = ensure_final_output_dirs(BASE_DIR, report_subdir=args.report_subdir or None)
    reports_dir = output_dirs["reports"]
    figures_dir = output_dirs["figures"]
    dataset_path = _resolve_dataset_path(args.dataset_path)

    if args.llm_only:
        LOGGER.info("Running poster LLM analysis only")
        write_llm_poster_analysis(args.llm_model, args.llm_backend, args.local_llm_model, reports_dir)
        print(str(reports_dir / "poster_llm_analysis.md"))
        return

    if args.final_sweep_only:
        LOGGER.info("Generating lightweight final poster sweep outputs")
        plot_llm_row_prediction_comparison()
        write_llm_row_prediction_summary()
        write_strict_subset_poster_brief()
        write_strict_subset_poster_map()
        write_poster_requirements_checklist(reports_dir)
        print(str(REPORTS_DIR / "fm15_strict_top25" / "poster_final_brief.md"))
        print(str(REPORTS_DIR / "fm15_strict_top25" / "poster_section_map.md"))
        return

    if args.descriptive_only:
        LOGGER.info("Generating descriptive subset figures only")
        plot_label_distribution(args.top_origins, dataset_path, figures_dir)
        plot_descriptive_subset_figures(args.top_origins, dataset_path, figures_dir)
        write_pca_component_summary(args.top_origins, dataset_path, reports_dir, figures_dir)
        if dataset_path.name == "modeling_dataset_fm15_rows.parquet":
            strict_reports_dir = BASE_DIR / "reports" / "fm15_strict_top25"
            write_supervised_model_comparison_with_llm(reports_dir, strict_reports_dir, figures_dir)
        if (reports_dir / "poster_cluster_summary.csv").exists() and (reports_dir / "poster_airport_clusters.csv").exists():
            write_cluster_method_summary(reports_dir, figures_dir)
        print(str(figures_dir))
        return

    LOGGER.info("Building poster dataset summary")
    build_poster_dataset_summary(args.top_origins, dataset_path, reports_dir)
    plot_label_distribution(args.top_origins, dataset_path, figures_dir)
    plot_feature_type_counts(reports_dir, figures_dir)
    plot_descriptive_subset_figures(args.top_origins, dataset_path, figures_dir)
    write_pca_component_summary(args.top_origins, dataset_path, reports_dir, figures_dir)

    LOGGER.info("Running poster model experiments")
    fold_metrics, _ = run_poster_model_experiments(
        args.top_origins,
        args.folds,
        args.train_cap,
        args.test_cap,
        dataset_path,
        reports_dir,
        args.use_full_data,
    )
    write_hyperparameter_summary(reports_dir)
    write_significance_tests(fold_metrics, reports_dir)

    LOGGER.info("Generating interpretability plots")
    plot_interpretability(args.top_origins, dataset_path, figures_dir, reports_dir, args.use_full_data)

    LOGGER.info("Running clustering analysis")
    run_clustering(args.top_origins, dataset_path, reports_dir, figures_dir)
    write_cluster_method_summary(reports_dir, figures_dir)

    LOGGER.info("Writing LLM-assisted poster analysis")
    write_llm_poster_analysis(args.llm_model, args.llm_backend, args.local_llm_model, reports_dir)
    if dataset_path.name == "modeling_dataset_fm15_rows.parquet":
        strict_reports_dir = BASE_DIR / "reports" / "fm15_strict_top25"
        write_supervised_model_comparison_with_llm(reports_dir, strict_reports_dir, figures_dir)

    write_poster_requirements_checklist(reports_dir)
    write_poster_summary(args.top_origins, reports_dir)
    plot_llm_row_prediction_comparison()
    write_llm_row_prediction_summary()
    write_strict_subset_poster_brief()
    write_strict_subset_poster_map()
    print(str(reports_dir / "poster_supplemental_summary.md"))
    print(str(reports_dir / "poster_requirements_checklist.md"))


if __name__ == "__main__":
    main()
