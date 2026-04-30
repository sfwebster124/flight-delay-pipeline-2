"""Final forecasting, classification, imbalance, diagnostics, and presentation workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from final_model_utils import (
    FinalFeatureBuildResult,
    apply_log_transforms,
    build_final_feature_frame,
    build_one_hot_preprocessor,
    build_ordinal_preprocessor,
    build_pca_logistic_preprocessor,
    configure_logging,
    ensure_final_output_dirs,
    fit_numeric_pca_diagnostics,
    load_final_analysis_frame,
    save_plot,
    temporal_train_test_split,
)


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_STATE = 42
TRAIN_SAMPLE_CAP = 250_000
TEST_SAMPLE_CAP = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final flight-delay modeling workflows.")
    parser.add_argument(
        "--top-origins",
        type=int,
        default=0,
        help="Optionally keep only the top N origin airports by flight count before splitting.",
    )
    parser.add_argument(
        "--pca-logistic-only",
        action="store_true",
        help="Run only the logistic PCA comparison workflow and skip the other final-analysis stages.",
    )
    parser.add_argument(
        "--full-filtered-data",
        action="store_true",
        help="Use all rows after filtering instead of the default train/test sample caps.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/modeling_dataset_fm15_strict_top25.parquet",
        help="Relative or absolute parquet path to use as the modeling dataset.",
    )
    parser.add_argument(
        "--report-subdir",
        type=str,
        default="",
        help="Optional subdirectory under reports/ for saving outputs from a specialized run.",
    )
    return parser.parse_args()


def _try_import_xgboost() -> tuple[object | None, object | None]:
    try:
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier, XGBRegressor
    except Exception:
        return None, None


def _try_import_imblearn() -> tuple[object | None, object | None, object | None]:
    try:
        from imblearn.over_sampling import SMOTE, SMOTENC
        from imblearn.pipeline import Pipeline as ImbPipeline

        return SMOTE, SMOTENC, ImbPipeline
    except Exception:
        return None, None, None


def _sample_frame(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    return frame.sample(n=cap, random_state=RANDOM_STATE)


def _classification_feature_list(result: FinalFeatureBuildResult) -> tuple[list[str], list[str], list[str]]:
    categorical = result.categorical_features.copy()
    numeric = result.numeric_features.copy()
    features = categorical + numeric
    return features, categorical, numeric


def _build_model_frames(
    dataset_path: Path,
    top_origins: int = 0,
    use_full_filtered_data: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, FinalFeatureBuildResult, pd.Timestamp, int]:
    frame = load_final_analysis_frame(BASE_DIR, dataset_path=dataset_path)
    feature_result = build_final_feature_frame(frame)
    if top_origins > 0:
        top_origin_codes = feature_result.frame["Origin"].value_counts().head(top_origins).index
        before_rows = len(feature_result.frame)
        feature_result.frame = feature_result.frame[feature_result.frame["Origin"].isin(top_origin_codes)].copy()
        LOGGER.info(
            "Filtered to top %s origin airports: kept %s of %s rows (%.1f%%)",
            top_origins,
            f"{len(feature_result.frame):,}",
            f"{before_rows:,}",
            100 * len(feature_result.frame) / max(before_rows, 1),
        )
    train, test, split_date = temporal_train_test_split(feature_result.frame)
    train, test, log_features = apply_log_transforms(train, test, feature_result.numeric_features)
    if log_features:
        feature_result.numeric_features.extend(log_features)
        feature_result.engineered_feature_notes.append(
            "log1p_* features: log-transformed versions of clearly right-skewed nonnegative numeric predictors, selected from the training split only."
        )
    retained_numeric_features: list[str] = []
    dropped_numeric_features: list[str] = []
    for feature in feature_result.numeric_features:
        if feature in train.columns and train[feature].notna().any():
            retained_numeric_features.append(feature)
        else:
            dropped_numeric_features.append(feature)
    if dropped_numeric_features:
        feature_result.numeric_features = retained_numeric_features
        feature_result.engineered_feature_notes.append(
            "Dropped all-missing numeric features for the active dataset subset: " + ", ".join(dropped_numeric_features) + "."
        )
        LOGGER.info("Dropped %s all-missing numeric features for this run", len(dropped_numeric_features))
    if use_full_filtered_data:
        LOGGER.info("Using all filtered rows for modeling: %s train rows and %s test rows", f"{len(train):,}", f"{len(test):,}")
    else:
        train = _sample_frame(train, TRAIN_SAMPLE_CAP)
        test = _sample_frame(test, TEST_SAMPLE_CAP)
        LOGGER.info(
            "Using %s sampled train rows and %s sampled test rows for final analysis",
            f"{len(train):,}",
            f"{len(test):,}",
        )
    return train, test, feature_result, split_date, len(feature_result.frame)


def write_reduction_summary(dataset_path: Path, filtered_row_count_frame: pd.DataFrame, top_origins: int) -> None:
    full_dataset_path = BASE_DIR / "data" / "modeling_dataset_fm15_strict_top25.parquet"
    if not full_dataset_path.exists():
        full_dataset_path = dataset_path
    full_main_rows = len(pd.read_parquet(full_dataset_path, columns=["Origin"]))
    source_rows = len(pd.read_parquet(dataset_path, columns=["Origin"]))
    filtered_rows = len(filtered_row_count_frame)
    reduction = pd.DataFrame(
        [
            {"stage": "full_main_dataset", "rows": full_main_rows},
            {"stage": dataset_path.stem, "rows": source_rows},
            {"stage": f"{dataset_path.stem}_top_{top_origins}_origins", "rows": filtered_rows},
        ]
    )
    reduction["rows_removed_from_previous"] = reduction["rows"].shift(1) - reduction["rows"]
    reduction["share_retained_from_previous"] = reduction["rows"] / reduction["rows"].shift(1)
    reduction["share_retained_from_full"] = reduction["rows"] / full_main_rows
    reduction.to_csv(REPORTS_DIR / "dataset_reduction_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(reduction["stage"], reduction["rows"])
    for idx, value in enumerate(reduction["rows"]):
        ax.text(idx, value, f"{int(value):,}", ha="center", va="bottom")
    ax.set_title("Dataset Reduction to Modeling Subset")
    ax.set_ylabel("Rows")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, FIGURES_DIR / "dataset_reduction_bar.png")

    markdown_lines = [
        "# Dataset Reduction Summary",
        "",
        reduction.to_markdown(index=False),
        "",
        f"- Full main dataset rows: {full_main_rows:,}",
        f"- Source dataset `{dataset_path.name}` rows: {source_rows:,}",
        f"- Top {top_origins} origins subset rows: {filtered_rows:,}",
    ]
    (REPORTS_DIR / "dataset_reduction_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    LOGGER.info("Saved %s", REPORTS_DIR / "dataset_reduction_summary.csv")


def _log_stage_completion(stage_index: int, total_stages: int, stage_label: str, stage_started_at: float, run_started_at: float) -> None:
    stage_elapsed = time.perf_counter() - stage_started_at
    total_elapsed = time.perf_counter() - run_started_at
    average_stage_seconds = total_elapsed / max(stage_index, 1)
    estimated_remaining = average_stage_seconds * max(total_stages - stage_index, 0)
    LOGGER.info(
        "Completed stage %s/%s: %s in %.1fs | elapsed %.1fs | est. remaining %.1fs",
        stage_index,
        total_stages,
        stage_label,
        stage_elapsed,
        total_elapsed,
        estimated_remaining,
    )


def _build_classifier_models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    XGBClassifier, _ = _try_import_xgboost()
    models: dict[str, Pipeline] = {
        "dummy_most_frequent": Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
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
    else:
        models["hist_gradient_boosting_classifier"] = Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=120,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    return models


def _build_regressor_models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    _, XGBRegressor = _try_import_xgboost()
    models: dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            [
                ("preprocessor", build_one_hot_preprocessor(categorical, numeric)),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest_regressor": Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=20,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    if XGBRegressor is not None:
        models["xgboost_regressor"] = Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=120,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        models["hist_gradient_boosting_regressor"] = Pipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=120,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    return models


def run_classification_models(train: pd.DataFrame, test: pd.DataFrame, feature_result: FinalFeatureBuildResult) -> tuple[pd.DataFrame, str, Pipeline, np.ndarray, pd.DataFrame]:
    features, categorical, numeric = _classification_feature_list(feature_result)
    X_train = train[features]
    y_train = train["dep_delayed_15"]
    X_test = test[features]
    y_test = test["dep_delayed_15"]

    rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    threshold_rows: list[dict[str, object]] = []

    for model_name, pipeline in _build_classifier_models(categorical, numeric).items():
        LOGGER.info("Training classification model %s", model_name)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps["model"], "predict_proba") else np.zeros(len(X_test))
        preds = (probabilities >= 0.5).astype(int) if probabilities.size else pipeline.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities) if probabilities.size else np.nan,
                "pr_auc": average_precision_score(y_test, probabilities) if probabilities.size else np.nan,
            }
        )
        if probabilities.size:
            for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
                threshold_preds = (probabilities >= threshold).astype(int)
                threshold_rows.append(
                    {
                        "model": model_name,
                        "threshold": threshold,
                        "precision": precision_score(y_test, threshold_preds, zero_division=0),
                        "recall": recall_score(y_test, threshold_preds, zero_division=0),
                        "f1": f1_score(y_test, threshold_preds, zero_division=0),
                    }
                )
        fitted_models[model_name] = pipeline

    metrics = pd.DataFrame(rows).sort_values(["pr_auc", "f1", "recall"], ascending=False).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    best_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    threshold_metrics = pd.DataFrame(threshold_rows)
    return metrics, best_model_name, best_pipeline, best_probabilities, threshold_metrics


def run_pca_logistic_comparison(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_result: FinalFeatureBuildResult,
    classification_metrics: pd.DataFrame,
) -> pd.DataFrame:
    features, categorical, numeric = _classification_feature_list(feature_result)
    X_train = train[features]
    y_train = train["dep_delayed_15"]
    X_test = test[features]
    y_test = test["dep_delayed_15"]

    baseline = classification_metrics.loc[
        classification_metrics["model"] == "logistic_regression_balanced",
        ["model", "f1", "roc_auc", "pr_auc"],
    ].copy()
    baseline["pca_components"] = np.nan
    baseline["numeric_variance_explained"] = np.nan
    baseline = baseline.rename(columns={"model": "experiment"})
    baseline["experiment"] = "logistic_regression_balanced_no_pca"

    cumulative_variance, selected_components, explained_variance = fit_numeric_pca_diagnostics(train, numeric)
    LOGGER.info(
        "PCA logistic selected %s numeric components to explain %.4f of numeric variance",
        selected_components,
        explained_variance,
    )

    pca_pipeline = Pipeline(
        [
            ("preprocessor", build_pca_logistic_preprocessor(categorical, numeric)),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
        ]
    )
    LOGGER.info("Training PCA logistic regression experiment")
    pca_pipeline.fit(X_train, y_train)
    probabilities = pca_pipeline.predict_proba(X_test)[:, 1]
    preds = (probabilities >= 0.5).astype(int)
    pca_row = pd.DataFrame(
        [
            {
                "experiment": "logistic_regression_balanced_with_pca",
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
                "pca_components": selected_components,
                "numeric_variance_explained": explained_variance,
            }
        ]
    )
    comparison = pd.concat([baseline, pca_row], ignore_index=True)
    comparison.to_csv(REPORTS_DIR / "final_pca_comparison.csv", index=False)
    LOGGER.info("Saved %s", REPORTS_DIR / "final_pca_comparison.csv")
    return comparison


def plot_pca_variance_curve(train: pd.DataFrame, feature_result: FinalFeatureBuildResult) -> tuple[int, float]:
    _, _, numeric = _classification_feature_list(feature_result)
    cumulative_variance, selected_components, explained_variance = fit_numeric_pca_diagnostics(train, numeric)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
    ax.axhline(0.95, linestyle="--")
    ax.axvline(selected_components, linestyle="--")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_pca_variance_curve.png")
    return selected_components, explained_variance


def write_pca_summary(comparison: pd.DataFrame, selected_components: int, explained_variance: float) -> None:
    baseline = comparison.loc[comparison["experiment"] == "logistic_regression_balanced_no_pca"].iloc[0]
    pca = comparison.loc[comparison["experiment"] == "logistic_regression_balanced_with_pca"].iloc[0]
    if float(pca["pr_auc"]) > float(baseline["pr_auc"]):
        outcome = "improved"
    elif float(pca["pr_auc"]) < float(baseline["pr_auc"]):
        outcome = "hurt"
    else:
        outcome = "matched"
    lines = [
        "# Final PCA Summary",
        "",
        f"- PCA selected {selected_components} numeric components to explain {explained_variance:.4f} of variance.",
        f"- Compared against the existing balanced logistic regression without PCA, the PCA variant {outcome} PR-AUC performance.",
        "- PCA reduces dimensionality in the numeric feature block but also reduces interpretability because the transformed components are no longer original variables.",
        "",
        "## Comparison Table",
        comparison.to_markdown(index=False),
        "",
        "## Figure",
        "- `reports/figures/final_pca_variance_curve.png`",
    ]
    output_path = REPORTS_DIR / "final_pca_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved %s", output_path)


def run_regression_models(train: pd.DataFrame, test: pd.DataFrame, feature_result: FinalFeatureBuildResult) -> tuple[pd.DataFrame, str, Pipeline]:
    features, categorical, numeric = _classification_feature_list(feature_result)
    X_train = train[features]
    y_train = train["dep_delay_minutes_late_only"]
    X_test = test[features]
    y_test = test["dep_delay_minutes_late_only"]

    rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, pipeline in _build_regressor_models(categorical, numeric).items():
        LOGGER.info("Training regression model %s", model_name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
            }
        )
        fitted_models[model_name] = pipeline

    metrics = pd.DataFrame(rows).sort_values(["mae", "rmse", "r2"], ascending=[True, True, False]).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    return metrics, best_model_name, fitted_models[best_model_name]


def run_imbalance_experiments(train: pd.DataFrame, test: pd.DataFrame, feature_result: FinalFeatureBuildResult) -> tuple[pd.DataFrame, str]:
    features, categorical, numeric = _classification_feature_list(feature_result)
    X_train = train[features]
    y_train = train["dep_delayed_15"]
    X_test = test[features]
    y_test = test["dep_delayed_15"]

    SMOTE, SMOTENC, ImbPipeline = _try_import_imblearn()
    rows: list[dict[str, object]] = []
    notes: list[str] = []

    experiment_pipelines: dict[str, object] = {
        "logistic_no_balancing": Pipeline(
            [
                ("preprocessor", build_one_hot_preprocessor(categorical, numeric)),
                ("model", LogisticRegression(max_iter=1200, solver="lbfgs")),
            ]
        ),
        "logistic_class_weight_balanced": Pipeline(
            [
                ("preprocessor", build_one_hot_preprocessor(categorical, numeric)),
                ("model", LogisticRegression(max_iter=1200, solver="lbfgs", class_weight="balanced")),
            ]
        ),
    }
    if SMOTENC is not None and ImbPipeline is not None:
        experiment_pipelines["logistic_smotenc"] = ImbPipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                ("sampler", SMOTENC(categorical_features=list(range(len(categorical))), random_state=RANDOM_STATE)),
                ("model", LogisticRegression(max_iter=1200, solver="lbfgs")),
            ]
        )
        notes.append("Oversampling experiment used SMOTENC after train/test splitting and inside an imblearn pipeline to avoid leakage.")
    elif SMOTE is not None and ImbPipeline is not None:
        experiment_pipelines["logistic_smote"] = ImbPipeline(
            [
                ("preprocessor", build_ordinal_preprocessor(categorical, numeric)),
                ("sampler", SMOTE(random_state=RANDOM_STATE)),
                ("model", LogisticRegression(max_iter=1200, solver="lbfgs")),
            ]
        )
        notes.append("Oversampling experiment used SMOTE after train/test splitting and inside an imblearn pipeline to avoid leakage.")
    else:
        notes.append("Oversampling experiment was skipped because imbalanced-learn is not installed in the active environment.")

    for name, pipeline in experiment_pipelines.items():
        LOGGER.info("Running imbalance experiment %s", name)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        preds = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "experiment": name,
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["pr_auc", "f1", "recall"], ascending=False).reset_index(drop=True)
    return metrics, " ".join(notes)


def build_feature_ranking(train: pd.DataFrame, feature_result: FinalFeatureBuildResult) -> pd.DataFrame:
    features, categorical, numeric = _classification_feature_list(feature_result)
    target_class = train["dep_delayed_15"]
    target_reg = train["dep_delay_minutes"]

    ranking = pd.DataFrame({"feature": features})
    ranking["pearson_with_delay_minutes"] = [pd.to_numeric(train[feature], errors="coerce").corr(target_reg) if feature in numeric else np.nan for feature in features]
    ranking["pearson_with_delay_flag"] = [pd.to_numeric(train[feature], errors="coerce").corr(target_class) if feature in numeric else np.nan for feature in features]

    ordinal = build_ordinal_preprocessor(categorical, numeric)
    encoded = ordinal.fit_transform(train[features])
    encoded_names = categorical + numeric
    mi_scores = mutual_info_classif(
        encoded,
        target_class,
        discrete_features=[name in categorical for name in encoded_names],
        random_state=RANDOM_STATE,
    )
    f_scores, _ = f_classif(encoded, target_class)
    ranking["mutual_information_class"] = mi_scores
    ranking["f_score_class"] = f_scores
    ranking["missing_rate_train"] = [float(train[feature].isna().mean()) for feature in features]
    ranking["abs_numeric_correlation_rank"] = ranking["pearson_with_delay_minutes"].abs().rank(ascending=False, method="dense")
    ranking = ranking.sort_values(["mutual_information_class", "f_score_class"], ascending=False).reset_index(drop=True)
    return ranking


def plot_delay_histogram(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(frame["dep_delay_minutes_late_only"].clip(0, 240).dropna(), bins=60)
    ax.set_title("Late-Only Departure Delay Minutes Distribution")
    ax.set_xlabel("Late Departure Minutes (clipped to [0, 240])")
    ax.set_ylabel("Flights")
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_delay_minutes_histogram.png")


def plot_correlation_heatmap(train: pd.DataFrame) -> None:
    candidate_columns = [
        "dep_delay_minutes_late_only",
        "origin_avg_dep_delay_prev_hour",
        "carrier_avg_dep_delay_prev_hour",
        "route_avg_dep_delay_prev_hour",
        "origin_delay_rate_prev_3h",
        "carrier_delay_rate_prev_3h",
        "airport_carrier_avg_dep_delay_prior",
        "origin_precip_mm",
        "origin_visibility_m",
        "origin_temp_c",
        "origin_humidity_pct",
        "Distance",
        "scheduled_departure_hour_local",
        "is_weekend",
        "is_holiday",
        "precip_peak_interaction",
    ]
    corr = train[candidate_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr.to_numpy(), aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Numeric Feature Correlation Heatmap")
    save_plot(fig, FIGURES_DIR / "final_correlation_heatmap.png")


def plot_classification_model_comparison(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(metrics))
    width = 0.16
    columns = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    for idx, column in enumerate(columns):
        ax.bar(positions + (idx - 2) * width, metrics[column], width=width, label=column)
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Final Classification Model Comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_model_comparison_bar.png")


def plot_regression_model_comparison(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(metrics))
    ax.bar(positions, metrics["mae"])
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics["model"], rotation=20, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Final Regression Model Comparison")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_regression_model_comparison_bar.png")


def plot_roc_curve(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve for Best Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_best_classifier_roc_curve.png")


def plot_pr_curve(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, label=model_name)
    ax.set_title("Precision-Recall Curve for Best Classifier")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_best_classifier_pr_curve.png")


def plot_threshold_tradeoff(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> None:
    rows: list[dict[str, float]] = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "f1": f1_score(y_true, preds, zero_division=0),
            }
        )
    work = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(work["threshold"], work["precision"], label="precision")
    ax.plot(work["threshold"], work["recall"], label="recall")
    ax.plot(work["threshold"], work["f1"], label="f1")
    ax.set_title(f"Threshold Tradeoff for {model_name}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_threshold_tradeoff.png")


def plot_calibration_curve(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Calibration Curve for Best Classifier")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_calibration_curve.png")


def plot_tree_feature_importance(best_pipeline: Pipeline, feature_result: FinalFeatureBuildResult, model_name: str) -> pd.Series:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(feature_result.categorical_features + feature_result.numeric_features, dtype=object)
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        importance = pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
    else:
        importance = pd.Series(dtype="float64")

    top = importance.head(15).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values)
    ax.set_title(f"Feature Importance for {model_name}")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_feature_importance.png")
    return importance


def plot_airport_bubble_map(test: pd.DataFrame) -> None:
    mapping_path = BASE_DIR / "data" / "airport_station_mapping.parquet"
    if not mapping_path.exists():
        LOGGER.warning("Skipping airport bubble plot because %s is not included in this package", mapping_path)
        return
    mapping = pd.read_parquet(mapping_path)[["airport", "latitude", "longitude"]]
    airport_summary = (
        test.groupby("Origin", dropna=False)
        .agg(flights=("Origin", "size"), delay_rate=("dep_delayed_15", "mean"))
        .reset_index()
        .rename(columns={"Origin": "airport"})
        .merge(mapping, on="airport", how="left")
        .dropna(subset=["latitude", "longitude"])
        .sort_values("flights", ascending=False)
        .head(75)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        airport_summary["longitude"],
        airport_summary["latitude"],
        s=airport_summary["flights"] / 300,
        c=airport_summary["delay_rate"],
    )
    fig.colorbar(scatter, ax=ax, label="Departure delay rate")
    ax.set_title("Airport Delay Bubble Plot")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    save_plot(fig, FIGURES_DIR / "final_airport_delay_bubble_map.png")


def write_preprocessing_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_result: FinalFeatureBuildResult,
    split_date: pd.Timestamp,
) -> None:
    features = feature_result.categorical_features + feature_result.numeric_features
    missingness = train[features].isna().mean().sort_values(ascending=False).head(15)
    skew_summary: list[tuple[str, float]] = []
    for feature in feature_result.numeric_features:
        series = pd.to_numeric(train[feature], errors="coerce")
        if series.dropna().empty:
            continue
        skew_summary.append((feature, float(series.skew(skipna=True))))
    skew_summary = sorted(skew_summary, key=lambda item: abs(item[1]), reverse=True)[:15]
    lines = [
        "# Final Preprocessing Summary",
        "",
        f"- Train/test split date: {split_date.date()}",
        f"- Train rows used: {len(train):,}",
        f"- Test rows used: {len(test):,}",
        "- Categorical imputation: most frequent value.",
        "- Numeric imputation: median value.",
        "- Linear and logistic models use one-hot encoding plus numeric standardization.",
        "- Tree, boosting, and oversampling pipelines use ordinal encoding for categorical features.",
        "",
        "## Engineered Features",
    ]
    lines.extend([f"- {note}" for note in feature_result.engineered_feature_notes])
    lines.extend(
        [
            "",
            "## Highest Missingness Features (Train)",
            missingness.to_markdown(),
            "",
            "## Most Skewed Numeric Features (Train)",
            pd.DataFrame(skew_summary, columns=["feature", "skew"]).to_markdown(index=False),
        ]
    )
    output_path = REPORTS_DIR / "final_preprocessing_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved %s", output_path)


def write_final_summary(
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    imbalance_metrics: pd.DataFrame,
    best_classifier_name: str,
    best_regressor_name: str,
) -> None:
    best_classifier_row = classification_metrics.iloc[0]
    best_regressor_row = regression_metrics.iloc[0]
    lines = [
        "# Final Model Comparison",
        "",
        "## Classification",
        classification_metrics.to_markdown(index=False),
        "",
        f"Best classifier: `{best_classifier_name}` with PR-AUC {float(best_classifier_row['pr_auc']):.4f}, ROC-AUC {float(best_classifier_row['roc_auc']):.4f}, and F1 {float(best_classifier_row['f1']):.4f}.",
        "",
        "## Regression",
        regression_metrics.to_markdown(index=False),
        "",
        f"Best regressor: `{best_regressor_name}` with MAE {float(best_regressor_row['mae']):.3f}, RMSE {float(best_regressor_row['rmse']):.3f}, and R^2 {float(best_regressor_row['r2']):.4f} on late-only departure minutes.",
        "",
        "## Class Imbalance Comparison",
        imbalance_metrics.to_markdown(index=False),
        "",
        "## Figures",
        "- `reports/figures/final_delay_minutes_histogram.png`",
        "- `reports/figures/final_correlation_heatmap.png`",
        "- `reports/figures/final_model_comparison_bar.png`",
        "- `reports/figures/final_regression_model_comparison_bar.png`",
        "- `reports/figures/final_best_classifier_roc_curve.png`",
        "- `reports/figures/final_best_classifier_pr_curve.png`",
        "- `reports/figures/final_feature_importance.png`",
        "- `reports/figures/final_threshold_tradeoff.png`",
        "- `reports/figures/final_calibration_curve.png`",
        "- `reports/figures/final_airport_delay_bubble_map.png`",
    ]
    output_path = REPORTS_DIR / "final_model_comparison.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    global REPORTS_DIR, FIGURES_DIR
    output_dirs = ensure_final_output_dirs(BASE_DIR, report_subdir=args.report_subdir or None)
    REPORTS_DIR = output_dirs["reports"]
    FIGURES_DIR = output_dirs["figures"]

    total_stages = 3 if args.pca_logistic_only else 10
    run_started_at = time.perf_counter()
    stage_index = 0

    stage_started_at = time.perf_counter()
    train, test, feature_result, split_date, filtered_row_count = _build_model_frames(
        dataset_path=dataset_path,
        top_origins=args.top_origins,
        use_full_filtered_data=args.full_filtered_data,
    )
    if args.top_origins > 0:
        write_reduction_summary(dataset_path, pd.DataFrame(index=range(filtered_row_count)), args.top_origins)
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "load data, engineer features, and split", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    classification_metrics, best_classifier_name, best_classifier_pipeline, best_probabilities, threshold_metrics = run_classification_models(train, test, feature_result)
    if not args.pca_logistic_only:
        classification_metrics.to_csv(REPORTS_DIR / "final_model_metrics.csv", index=False)
        threshold_metrics.to_csv(REPORTS_DIR / "final_threshold_metrics.csv", index=False)
        LOGGER.info("Saved %s", REPORTS_DIR / "final_model_metrics.csv")
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "run classification models", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    pca_comparison = run_pca_logistic_comparison(train, test, feature_result, classification_metrics)
    selected_components, explained_variance = plot_pca_variance_curve(train, feature_result)
    write_pca_summary(pca_comparison, selected_components, explained_variance)
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "run PCA logistic comparison", stage_started_at, run_started_at)

    if args.pca_logistic_only:
        print("PCA vs non-PCA logistic comparison:")
        print(pca_comparison[["experiment", "f1", "roc_auc", "pr_auc", "pca_components", "numeric_variance_explained"]].to_string(index=False))
        print(str(REPORTS_DIR / "final_pca_comparison.csv"))
        print(str(REPORTS_DIR / "final_pca_summary.md"))
        print(str(FIGURES_DIR / "final_pca_variance_curve.png"))
        return

    stage_started_at = time.perf_counter()
    regression_metrics, best_regressor_name, _ = run_regression_models(train, test, feature_result)
    regression_metrics.to_csv(REPORTS_DIR / "final_regression_metrics.csv", index=False)
    LOGGER.info("Saved %s", REPORTS_DIR / "final_regression_metrics.csv")
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "run regression models", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    imbalance_metrics, imbalance_note = run_imbalance_experiments(train, test, feature_result)
    imbalance_metrics.to_csv(REPORTS_DIR / "final_imbalance_metrics.csv", index=False)
    (REPORTS_DIR / "final_imbalance_summary.md").write_text(
        "# Final Imbalance Summary\n\n" + imbalance_metrics.to_markdown(index=False) + "\n\n" + imbalance_note + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved %s", REPORTS_DIR / "final_imbalance_metrics.csv")
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "run imbalance experiments", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    feature_ranking = build_feature_ranking(train, feature_result)
    feature_ranking.to_csv(REPORTS_DIR / "final_feature_ranking.csv", index=False)
    LOGGER.info("Saved %s", REPORTS_DIR / "final_feature_ranking.csv")
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "build feature ranking", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    plot_delay_histogram(train)
    plot_correlation_heatmap(train)
    plot_classification_model_comparison(classification_metrics)
    plot_regression_model_comparison(regression_metrics)
    plot_roc_curve(test["dep_delayed_15"], best_probabilities, best_classifier_name)
    plot_pr_curve(test["dep_delayed_15"], best_probabilities, best_classifier_name)
    plot_threshold_tradeoff(test["dep_delayed_15"], best_probabilities, best_classifier_name)
    plot_calibration_curve(test["dep_delayed_15"], best_probabilities, best_classifier_name)
    plot_tree_feature_importance(best_classifier_pipeline, feature_result, best_classifier_name)
    plot_airport_bubble_map(test)
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "save figures", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    write_preprocessing_summary(train, test, feature_result, split_date)
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "write preprocessing summary", stage_started_at, run_started_at)

    stage_started_at = time.perf_counter()
    write_final_summary(classification_metrics, regression_metrics, imbalance_metrics, best_classifier_name, best_regressor_name)
    stage_index += 1
    _log_stage_completion(stage_index, total_stages, "write final summary", stage_started_at, run_started_at)

    print("PCA vs non-PCA logistic comparison:")
    print(pca_comparison[["experiment", "f1", "roc_auc", "pr_auc", "pca_components", "numeric_variance_explained"]].to_string(index=False))
    print(str(REPORTS_DIR / "final_pca_comparison.csv"))
    print(str(REPORTS_DIR / "final_pca_summary.md"))
    print(str(FIGURES_DIR / "final_pca_variance_curve.png"))


if __name__ == "__main__":
    main()
