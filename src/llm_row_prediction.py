"""Row-level LLM delay classification experiment on the FM-15 strict top-origin subset."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from final_model_utils import configure_logging, ensure_final_output_dirs


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def _slugify_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a row-level LLM delay prediction experiment.")
    parser.add_argument("--dataset-path", default="data/modeling_dataset_fm15_strict_top25.parquet")
    parser.add_argument("--top-origins", type=int, default=25)
    parser.add_argument("--sample-size", type=int, default=80, help="Balanced evaluation sample size drawn from the temporal test split.")
    parser.add_argument("--few-shot-examples", type=int, default=8, help="Number of balanced few-shot examples drawn from training data.")
    parser.add_argument("--local-llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--report-subdir", default="fm15_strict_top25")
    return parser.parse_args()


def _load_local_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    return torch, AutoModelForCausalLM, AutoTokenizer, pipeline


def _resolve_local_model_path(model_name: str) -> str:
    direct_path = Path(model_name)
    if direct_path.exists():
        return str(direct_path)
    user_home = Path.home()
    cache_root = user_home / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    ref_path = model_dir / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = model_dir / "snapshots" / revision
        if snapshot_path.exists():
            return str(snapshot_path)
    return model_name


def _load_dataset(dataset_path: Path, top_origins: int) -> pd.DataFrame:
    columns = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "scheduled_departure_hour_local",
        "Distance",
        "origin_temp_c",
        "origin_visibility_m",
        "origin_wind_speed_mps",
        "origin_precip_mm",
        "origin_ceiling_m",
        "origin_humidity_pct",
        "origin_hourly_avg_dep_delay_prior",
        "carrier_route_avg_dep_delay_prior",
        "origin_hourly_congestion_proxy_prior",
        "dep_delayed_15",
    ]
    frame = pd.read_parquet(dataset_path, columns=columns)
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="coerce")
    frame["dep_delayed_15"] = pd.to_numeric(frame["dep_delayed_15"], errors="coerce")
    frame = frame.dropna(subset=["FlightDate", "dep_delayed_15", "Origin", "Dest", "Reporting_Airline"]).copy()
    frame["dep_delayed_15"] = frame["dep_delayed_15"].astype(int)
    if top_origins > 0:
        top_origin_codes = frame["Origin"].value_counts().head(top_origins).index
        frame = frame[frame["Origin"].isin(top_origin_codes)].copy()
    return frame


def _temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(frame["FlightDate"].dt.normalize().dropna().unique().tolist())
    split_index = max(int(len(unique_dates) * 0.8), 1)
    split_date = pd.Timestamp(unique_dates[split_index])
    train = frame[frame["FlightDate"] < split_date].copy()
    test = frame[frame["FlightDate"] >= split_date].copy()
    LOGGER.info("Temporal split at %s -> %s train rows, %s test rows", split_date.date(), f"{len(train):,}", f"{len(test):,}")
    return train, test


def _balanced_sample(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    positives = frame[frame["dep_delayed_15"] == 1]
    negatives = frame[frame["dep_delayed_15"] == 0]
    half = sample_size // 2
    positive_sample = positives.sample(n=min(len(positives), half), random_state=RANDOM_STATE)
    negative_sample = negatives.sample(n=min(len(negatives), sample_size - len(positive_sample)), random_state=RANDOM_STATE)
    sample = pd.concat([positive_sample, negative_sample], ignore_index=True)
    return sample.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def _format_row(row: pd.Series) -> str:
    return (
        f"origin={row['Origin']}; dest={row['Dest']}; carrier={row['Reporting_Airline']}; "
        f"dep_hour_local={int(row['scheduled_departure_hour_local']) if pd.notna(row['scheduled_departure_hour_local']) else 'NA'}; "
        f"distance_miles={_format_num(row['Distance'])}; "
        f"origin_temp_c={_format_num(row['origin_temp_c'])}; "
        f"origin_visibility_m={_format_num(row['origin_visibility_m'])}; "
        f"origin_wind_speed_mps={_format_num(row['origin_wind_speed_mps'])}; "
        f"origin_precip_mm={_format_num(row['origin_precip_mm'])}; "
        f"origin_ceiling_m={_format_num(row['origin_ceiling_m'])}; "
        f"origin_humidity_pct={_format_num(row['origin_humidity_pct'])}; "
        f"prior_origin_avg_delay={_format_num(row['origin_hourly_avg_dep_delay_prior'])}; "
        f"prior_carrier_route_avg_delay={_format_num(row['carrier_route_avg_dep_delay_prior'])}; "
        f"origin_congestion_proxy={_format_num(row['origin_hourly_congestion_proxy_prior'])}"
    )


def _format_num(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def _build_prompt(few_shot: pd.DataFrame, row: pd.Series) -> str:
    examples: list[str] = []
    for _, example in few_shot.iterrows():
        label = "DELAY" if int(example["dep_delayed_15"]) == 1 else "ON_TIME"
        examples.append(f"Example: {_format_row(example)} -> {label}")
    prompt = "\n".join(
        [
            "Predict whether the flight will be delayed by 15+ minutes.",
            "Use only the provided row features.",
            "Respond with exactly one label: DELAY or ON_TIME.",
            *examples,
            f"Target row: {_format_row(row)}",
            "Prediction:",
        ]
    )
    return prompt


def _parse_prediction(text: str) -> int | None:
    upper = text.upper()
    delay_idx = upper.find("DELAY")
    on_time_idx = upper.find("ON_TIME")
    if on_time_idx != -1 and (delay_idx == -1 or on_time_idx < delay_idx):
        return 0
    if delay_idx != -1:
        return 1
    if "YES" in upper:
        return 1
    if "NO" in upper:
        return 0
    return None


def run_local_llm_experiment(model_name: str, prompts: list[str]) -> list[str]:
    torch, auto_model, auto_tokenizer, text_pipeline = _load_local_llm()
    model_path = _resolve_local_model_path(model_name)
    LOGGER.info("Loading local LLM from %s", model_path)
    tokenizer = auto_tokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = auto_model.from_pretrained(model_path, local_files_only=True)
    generator = text_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    outputs: list[str] = []
    for index, prompt in enumerate(prompts, start=1):
        if index % 10 == 0 or index == 1:
            LOGGER.info("LLM row prediction %s/%s", index, len(prompts))
        response = generator(
            [
                {"role": "system", "content": "You are a flight-delay classifier. Output only DELAY or ON_TIME."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=6,
            do_sample=False,
        )
        generated = response[0]["generated_text"]
        if isinstance(generated, list):
            assistant_chunks = [item.get("content", "") for item in generated if item.get("role") == "assistant"]
            outputs.append("\n".join(chunk for chunk in assistant_chunks if chunk).strip())
        else:
            outputs.append(str(generated))
    return outputs


def main() -> None:
    args = parse_args()
    configure_logging()
    output_dirs = ensure_final_output_dirs(BASE_DIR, report_subdir=args.report_subdir)
    reports_dir = output_dirs["reports"]

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    frame = _load_dataset(dataset_path, args.top_origins)
    train, test = _temporal_split(frame)
    eval_sample = _balanced_sample(test, args.sample_size)
    few_shot = _balanced_sample(train, args.few_shot_examples)
    prompts = [_build_prompt(few_shot, row) for _, row in eval_sample.iterrows()]
    raw_outputs = run_local_llm_experiment(args.local_llm_model, prompts)

    predictions = [_parse_prediction(text) for text in raw_outputs]
    result = eval_sample.copy()
    result["llm_raw_output"] = raw_outputs
    result["llm_predicted_delay_15"] = predictions
    result["llm_prediction_valid"] = result["llm_predicted_delay_15"].notna().astype(int)

    valid = result.dropna(subset=["llm_predicted_delay_15"]).copy()
    valid["llm_predicted_delay_15"] = valid["llm_predicted_delay_15"].astype(int)
    y_true = valid["dep_delayed_15"].astype(int)
    y_pred = valid["llm_predicted_delay_15"].astype(int)

    metrics = {
        "model": args.local_llm_model,
        "dataset_path": str(dataset_path),
        "top_origins": args.top_origins,
        "evaluation_rows_requested": args.sample_size,
        "evaluation_rows_valid": len(valid),
        "few_shot_examples": args.few_shot_examples,
        "accuracy": accuracy_score(y_true, y_pred) if len(valid) else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0) if len(valid) else float("nan"),
        "recall": recall_score(y_true, y_pred, zero_division=0) if len(valid) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0) if len(valid) else float("nan"),
        "valid_prediction_rate": float(result["llm_prediction_valid"].mean()),
    }
    metrics_frame = pd.DataFrame([metrics])

    model_slug = _slugify_model_name(args.local_llm_model)
    predictions_path = reports_dir / f"llm_delay_predictions_{model_slug}.csv"
    metrics_path = reports_dir / f"llm_delay_metrics_{model_slug}.csv"
    summary_path = reports_dir / f"llm_delay_metrics_{model_slug}.md"
    result.to_csv(predictions_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)

    if len(valid):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0

    summary_lines = [
        "# LLM Row Prediction Summary",
        "",
        f"- Model: `{args.local_llm_model}`",
        f"- Dataset: `{dataset_path.name}`",
        f"- Top origins retained: {args.top_origins}",
        f"- Requested evaluation rows: {args.sample_size}",
        f"- Valid predictions parsed: {len(valid)}",
        f"- Valid prediction rate: {metrics['valid_prediction_rate']:.2%}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        "",
        "## Confusion Matrix",
        f"- TN: {tn}",
        f"- FP: {fp}",
        f"- FN: {fn}",
        f"- TP: {tp}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    comparison_path = reports_dir / "llm_delay_model_comparison.csv"
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != args.local_llm_model]
        comparison = pd.concat([existing, metrics_frame], ignore_index=True)
    else:
        comparison = metrics_frame
    comparison.to_csv(comparison_path, index=False)

    print(str(summary_path))
    print(str(comparison_path))


if __name__ == "__main__":
    main()
