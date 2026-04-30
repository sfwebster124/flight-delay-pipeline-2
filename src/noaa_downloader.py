"""Download and clean NOAA hourly weather data for mapped airport stations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from timezonefinder import TimezoneFinder
from tqdm import tqdm

from config import NOAA_ACCESS_DATA_URL, PipelineConfig, build_retry_session, sleep_briefly


LOGGER = logging.getLogger(__name__)
TIMEZONE_FINDER = TimezoneFinder()


def _safe_numeric_from_compound(
    series: pd.Series,
    index: int = 0,
    scale: float = 1.0,
    missing_markers: tuple[str, ...] = ("9999", "99999", "999999"),
) -> pd.Series:
    values = series.fillna("").astype("string").str.split(",", expand=True)
    if index >= values.shape[1]:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    extracted = values[index].astype("string").str.strip()
    extracted = extracted.where(~extracted.isin(missing_markers), pd.NA)
    numeric = pd.to_numeric(extracted, errors="coerce")
    return numeric / scale


def _safe_text_from_compound(series: pd.Series, index: int = 0) -> pd.Series:
    values = series.fillna("").astype("string").str.split(",", expand=True)
    if index >= values.shape[1]:
        return pd.Series(pd.NA, index=series.index, dtype="string")
    return values[index].astype("string").str.strip()


def _first_present(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    active = [column for column in columns if column in df.columns]
    if not active:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    result = df[active[0]].copy()
    for column in active[1:]:
        result = result.fillna(df[column])
    return result


def add_timezones_to_mapping(mapping: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    mapping = mapping.copy()
    timezones: list[str | None] = []
    for row in mapping.itertuples(index=False):
        tz_name = None
        if pd.notna(row.latitude) and pd.notna(row.longitude):
            tz_name = TIMEZONE_FINDER.timezone_at(lng=float(row.longitude), lat=float(row.latitude))
        timezones.append(tz_name)
    mapping["timezone"] = timezones
    parquet_path = config.processed_dir / "airport_station_mapping.parquet"
    mapping.to_parquet(parquet_path, index=False)
    return mapping


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, ValueError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _download_station_year_json(station_id: str, year: int, config: PipelineConfig) -> bytes:
    params = {
        "dataset": "global-hourly",
        "stations": station_id,
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31",
        "format": "json",
        "includeStationName": "true",
        "includeStationLocation": "true",
        "units": "metric",
    }
    session = build_retry_session()
    response = session.get(NOAA_ACCESS_DATA_URL, params=params, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    if response.text.strip().startswith("{") and "errorMessage" in response.text:
        raise ValueError(f"NOAA API returned an error for {station_id} {year}: {response.text[:500]}")
    sleep_briefly(config.sleep_seconds_between_requests)
    return response.content


def _download_station_year_file(station_id: str, year: int, config: PipelineConfig) -> Path:
    destination = config.raw_noaa_dir / station_id / f"{station_id}_{year}.json"
    if destination.exists() and not config.overwrite:
        LOGGER.info("Using cached NOAA file %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _download_station_year_json(station_id, year, config)
    destination.write_bytes(content)
    LOGGER.info("Saved NOAA station-year data to %s", destination)
    return destination


def _parse_noaa_raw_json(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_json(path)
    except ValueError:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return pd.DataFrame()
        raise
    if frame.empty:
        return frame
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def clean_noaa_hourly_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    frame = raw.copy()
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    frame["temperature_c"] = _safe_numeric_from_compound(frame.get("TMP", pd.Series(dtype="string")), index=0, scale=10.0)
    frame["dew_point_c"] = _safe_numeric_from_compound(frame.get("DEW", pd.Series(dtype="string")), index=0, scale=10.0)
    frame["wind_direction_deg"] = _safe_numeric_from_compound(frame.get("WND", pd.Series(dtype="string")), index=0, missing_markers=("999", "9999"))
    frame["wind_speed_mps"] = _safe_numeric_from_compound(frame.get("WND", pd.Series(dtype="string")), index=3, scale=10.0)
    frame["visibility_m"] = _safe_numeric_from_compound(frame.get("VIS", pd.Series(dtype="string")), index=0, missing_markers=("999999", "99999"))
    frame["ceiling_m"] = _safe_numeric_from_compound(frame.get("CIG", pd.Series(dtype="string")), index=0, missing_markers=("99999", "999999"))
    frame["sea_level_pressure_hpa"] = _safe_numeric_from_compound(frame.get("SLP", pd.Series(dtype="string")), index=0, scale=10.0)

    precip_columns = []
    for column in ["AA1", "AA2", "AA3", "AA4"]:
        if column in frame.columns:
            parsed_name = f"{column.lower()}_precip_mm"
            frame[parsed_name] = _safe_numeric_from_compound(frame[column], index=1, scale=10.0)
            precip_columns.append(parsed_name)
    frame["precip_mm"] = _first_present(frame, precip_columns)

    weather_code_columns = []
    for column in ["AW1", "AW2", "AW3", "AW4", "MW1", "MW2", "MW3", "MW4"]:
        if column in frame.columns:
            parsed_name = f"{column.lower()}_code"
            frame[parsed_name] = _safe_text_from_compound(frame[column], index=0)
            weather_code_columns.append(parsed_name)
    if weather_code_columns:
        result = frame[weather_code_columns[0]].astype("string")
        for column in weather_code_columns[1:]:
            result = result.fillna(frame[column].astype("string"))
        frame["weather_code"] = result
    else:
        frame["weather_code"] = pd.Series(pd.NA, index=frame.index, dtype="string")

    temp = frame["temperature_c"]
    dew = frame["dew_point_c"]
    humidity = 100 * (np.exp((17.625 * dew) / (243.04 + dew)) / np.exp((17.625 * temp) / (243.04 + temp)))
    frame["relative_humidity_pct"] = humidity.where(temp.notna() & dew.notna())

    keep_columns = [
        "STATION", "DATE", "NAME", "LATITUDE", "LONGITUDE", "REPORT_TYPE", "SOURCE", "CALL_SIGN",
        "temperature_c", "dew_point_c", "wind_speed_mps", "wind_direction_deg", "visibility_m",
        "sea_level_pressure_hpa", "precip_mm", "weather_code", "relative_humidity_pct", "ceiling_m",
    ]
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    cleaned = frame[keep_columns].rename(columns={
        "STATION": "station_id",
        "DATE": "timestamp_utc",
        "NAME": "station_name",
        "LATITUDE": "station_latitude",
        "LONGITUDE": "station_longitude",
        "REPORT_TYPE": "report_type",
        "SOURCE": "source",
        "CALL_SIGN": "call_sign",
    })
    for column in ["station_id", "station_name", "report_type", "source", "call_sign", "weather_code"]:
        cleaned[column] = cleaned[column].astype("string")
    for column in [
        "station_latitude",
        "station_longitude",
        "temperature_c",
        "dew_point_c",
        "wind_speed_mps",
        "wind_direction_deg",
        "visibility_m",
        "sea_level_pressure_hpa",
        "precip_mm",
        "relative_humidity_pct",
        "ceiling_m",
    ]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.drop_duplicates(subset=["station_id", "timestamp_utc"]).reset_index(drop=True)
    return cleaned


def download_and_prepare_noaa(
    flights: pd.DataFrame,
    mapping: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = add_timezones_to_mapping(mapping, config)
    required_station_ids = mapping["station_id"].dropna().astype(str).unique().tolist()
    years = sorted(pd.to_datetime(flights["FlightDate"]).dt.year.dropna().astype(int).unique().tolist())

    cleaned_frames: list[pd.DataFrame] = []
    for station_id in tqdm(required_station_ids, desc="NOAA stations"):
        for year in years:
            raw_path = _download_station_year_file(station_id, year, config)
            raw_frame = _parse_noaa_raw_json(raw_path)
            if raw_frame.empty:
                LOGGER.warning("No NOAA hourly rows returned for station %s year %s", station_id, year)
                continue
            cleaned = clean_noaa_hourly_frame(raw_frame)
            cleaned["station_id"] = cleaned["station_id"].astype("string").fillna(station_id)
            cleaned_frames.append(cleaned)

    if not cleaned_frames:
        raise ValueError("NOAA download completed but no hourly weather records were parsed.")

    combined = pd.concat(cleaned_frames, ignore_index=True)
    combined = combined.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    parquet_path = config.processed_dir / "weather_hourly.parquet"
    combined.to_parquet(parquet_path, index=False)
    LOGGER.info("Saved cleaned NOAA weather to %s", parquet_path)
    return combined, mapping
