"""Download and normalize BTS Reporting Carrier On-Time Performance data."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import BTS_PREZIP_BASE_URL, PipelineConfig, build_retry_session, month_range, sleep_briefly


LOGGER = logging.getLogger(__name__)


def build_bts_month_url(year: int, month: int) -> str:
    filename = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    return f"{BTS_PREZIP_BASE_URL}/{filename}"


def _download_file(url: str, destination: Path, config: PipelineConfig) -> Path:
    session = build_retry_session()
    response = session.get(url, timeout=config.request_timeout_seconds, stream=True)
    if response.status_code == 404:
        raise FileNotFoundError(f"BTS file not found: {url}")
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    sleep_briefly(config.sleep_seconds_between_requests)
    return destination
def _monthly_cache_path(year: int, month: int, cache_dir: Path) -> Path:
    return cache_dir / f"bts_{year}_{month:02d}.parquet"


def _normalize_bts_frame(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    rename_map = {
        "DOT_ID_Reporting_Airline ": "DOT_ID_Reporting_Airline",
        "Flight_Number_Reporting_Airline ": "Flight_Number_Reporting_Airline",
    }
    frame = frame.rename(columns=rename_map)
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    ordered_columns = required_columns + [col for col in frame.columns if col not in required_columns]
    frame = frame[ordered_columns].copy()
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="coerce")
    for code_column in ["Reporting_Airline", "Origin", "Dest", "OriginState", "DestState"]:
        if code_column in frame.columns:
            frame[code_column] = frame[code_column].astype("string").str.strip().str.upper()
    return frame
def load_single_bts_month_from_zip(zip_path: Path, year: int, month: int, config: PipelineConfig) -> pd.DataFrame:
    cache_dir = config.raw_bts_dir / "extracted"
    cache_path = _monthly_cache_path(year, month, cache_dir)
    if cache_path.exists() and not config.overwrite:
        LOGGER.info("Loading cached BTS month parquet from %s", cache_path)
        return pd.read_parquet(cache_path)

    LOGGER.info("Loading BTS month from %s", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"Expected one CSV inside {zip_path}, found {csv_members}")
        with archive.open(csv_members[0]) as source:
            frame = pd.read_csv(source, low_memory=False)

    frame = _normalize_bts_frame(frame, config.bts_columns)
    if config.airlines:
        frame = frame[frame["Reporting_Airline"].isin(config.airlines)].copy()
    if config.origin_airports:
        frame = frame[frame["Origin"].isin(config.origin_airports)].copy()
    if config.filter_start_date:
        frame = frame[frame["FlightDate"] >= pd.Timestamp(config.filter_start_date)].copy()
    if config.filter_end_date:
        frame = frame[frame["FlightDate"] <= pd.Timestamp(config.filter_end_date)].copy()
    frame["source_csv"] = Path(csv_members[0]).name

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    LOGGER.info("Saved BTS month parquet cache to %s", cache_path)
    return frame


def download_and_prepare_bts(config: PipelineConfig) -> tuple[pd.DataFrame, Path]:
    config.ensure_directories()
    monthly_frames: list[pd.DataFrame] = []

    months = list(month_range(config.start_year, config.start_month, config.end_year, config.end_month))
    for year, month in tqdm(months, desc="BTS months"):
        zip_name = f"bts_{year}_{month:02d}.zip"
        zip_path = config.raw_bts_dir / zip_name
        url = build_bts_month_url(year, month)
        if zip_path.exists() and not config.overwrite:
            LOGGER.info("Using cached BTS archive %s", zip_path)
        else:
            LOGGER.info("Downloading BTS %s-%02d from %s", year, month, url)
            _download_file(url, zip_path, config)
        month_frame = load_single_bts_month_from_zip(zip_path, year, month, config)
        month_frame["data_year"] = year
        month_frame["data_month"] = month
        monthly_frames.append(month_frame)

    if not monthly_frames:
        raise ValueError("No BTS monthly data was loaded for the requested filters.")

    combined = pd.concat(monthly_frames, ignore_index=True)
    combined = combined.sort_values(["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime"]).reset_index(drop=True)
    parquet_path = config.processed_dir / "flights_combined.parquet"
    combined.to_parquet(parquet_path, index=False)
    LOGGER.info("Saved combined BTS flights to %s", parquet_path)
    return combined, parquet_path
