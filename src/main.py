"""CLI entrypoint for the BTS + NOAA flight delay data pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bts_downloader import download_and_prepare_bts
from build_features import build_modeling_dataset, build_quality_report
from config import PipelineConfig, configure_logging, parse_csv_argument
from noaa_downloader import download_and_prepare_noaa
from station_mapper import map_airports_to_stations


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a flight delay modeling dataset from official BTS and NOAA sources.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--start-month", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--end-month", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--airlines", type=str, default="")
    parser.add_argument("--origin-airports", type=str, default="")
    parser.add_argument("--filter-start-date", type=str, default=None)
    parser.add_argument("--filter-end-date", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.start_month <= 12:
        raise ValueError("start-month must be between 1 and 12")
    if not 1 <= args.end_month <= 12:
        raise ValueError("end-month must be between 1 and 12")
    if (args.start_year, args.start_month) > (args.end_year, args.end_month):
        raise ValueError("The start year/month must not come after the end year/month")


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        output_dir=args.output_dir,
        airlines=parse_csv_argument(args.airlines),
        origin_airports=parse_csv_argument(args.origin_airports),
        filter_start_date=args.filter_start_date,
        filter_end_date=args.filter_end_date,
        overwrite=args.overwrite,
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    validate_args(args)
    config = build_config(args)
    config.ensure_directories()

    LOGGER.info("Starting pipeline for %s/%s through %s/%s", config.start_year, config.start_month, config.end_year, config.end_month)
    flights, _ = download_and_prepare_bts(config)
    mapping = map_airports_to_stations(flights, config)
    weather, mapping = download_and_prepare_noaa(flights, mapping, config)
    modeling = build_modeling_dataset(flights, weather, mapping, config)
    report_path = build_quality_report(flights, mapping, weather, modeling, config)
    LOGGER.info("Pipeline complete. Quality report written to %s", report_path)


if __name__ == "__main__":
    main()
