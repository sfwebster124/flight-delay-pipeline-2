"""Shared configuration and utility helpers for the flight delay pipeline."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BTS_PREZIP_BASE_URL = "https://transtats.bts.gov/PREZIP"
NOAA_ACCESS_DATA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NOAA_ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


DEFAULT_BTS_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "DOT_ID_Reporting_Airline", "IATA_CODE_Reporting_Airline",
    "Tail_Number", "Flight_Number_Reporting_Airline", "OriginAirportID",
    "OriginAirportSeqID", "OriginCityMarketID", "Origin", "OriginCityName",
    "OriginState", "OriginStateName", "DestAirportID", "DestAirportSeqID",
    "DestCityMarketID", "Dest", "DestCityName", "DestState", "DestStateName",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "CRSArrTime", "ArrTime",
    "ArrDelay", "ArrDelayMinutes", "ArrDel15", "Cancelled", "CancellationCode",
    "Diverted", "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the data pipeline."""

    start_year: int
    start_month: int
    end_year: int
    end_month: int
    output_dir: Path
    airlines: list[str] = field(default_factory=list)
    origin_airports: list[str] = field(default_factory=list)
    filter_start_date: str | None = None
    filter_end_date: str | None = None
    overwrite: bool = False
    request_timeout_seconds: int = 120
    bts_columns: list[str] = field(default_factory=lambda: DEFAULT_BTS_COLUMNS.copy())
    noaa_token: str | None = None
    sleep_seconds_between_requests: float = 0.35

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.noaa_token = self.noaa_token or os.getenv("NOAA_TOKEN")
        self.airlines = [item.strip().upper() for item in self.airlines if item.strip()]
        self.origin_airports = [item.strip().upper() for item in self.origin_airports if item.strip()]

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def raw_bts_dir(self) -> Path:
        return self.raw_dir / "bts"

    @property
    def raw_noaa_dir(self) -> Path:
        return self.raw_dir / "noaa"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    def ensure_directories(self) -> None:
        for path in [
            self.output_dir, self.raw_dir, self.raw_bts_dir, self.raw_noaa_dir,
            self.cache_dir, self.processed_dir, self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_retry_session(
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "flight-delay-pipeline/1.0 (official BTS/NOAA downloader)",
        "Accept": "*/*",
    })
    return session


def month_range(start_year: int, start_month: int, end_year: int, end_month: int) -> Iterator[tuple[int, int]]:
    year = start_year
    month = start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            month = 1
            year += 1


def sleep_briefly(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def parse_csv_argument(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def iter_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
