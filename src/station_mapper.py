"""Map BTS airport codes to NOAA hourly weather stations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from config import NOAA_ACCESS_DATA_URL, NOAA_ISD_HISTORY_URL, PipelineConfig, build_retry_session, sleep_briefly


LOGGER = logging.getLogger(__name__)


def _target_years_from_flights(flights: pd.DataFrame) -> list[int]:
    years = pd.to_datetime(flights["FlightDate"], errors="coerce").dt.year.dropna().astype(int).unique().tolist()
    return sorted(years)


def download_station_history(config: PipelineConfig) -> Path:
    config.ensure_directories()
    destination = config.cache_dir / "noaa_isd_history.csv"
    if destination.exists() and not config.overwrite:
        LOGGER.info("Using cached NOAA station metadata %s", destination)
        return destination

    session = build_retry_session()
    response = session.get(NOAA_ISD_HISTORY_URL, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    destination.write_bytes(response.content)
    sleep_briefly(config.sleep_seconds_between_requests)
    LOGGER.info("Saved NOAA station metadata to %s", destination)
    return destination


def load_station_history(path: Path) -> pd.DataFrame:
    stations = pd.read_csv(path, dtype=str)
    stations.columns = [column.strip().replace(" ", "_").replace("(", "").replace(")", "") for column in stations.columns]
    rename_map = {
        "STATION_NAME": "station_name",
        "CTRY": "country",
        "STATE": "state",
        "ICAO": "icao",
        "CALL": "call",
        "USAF": "usaf",
        "WBAN": "wban",
        "LAT": "latitude",
        "LON": "longitude",
        "ELEV_M": "elevation_m",
        "BEGIN": "begin_date",
        "END": "end_date",
    }
    stations = stations.rename(columns=rename_map)
    for required in rename_map.values():
        if required not in stations.columns:
            stations[required] = pd.NA
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations["begin_date"] = pd.to_datetime(stations["begin_date"], format="%Y%m%d", errors="coerce")
    stations["end_date"] = pd.to_datetime(stations["end_date"], format="%Y%m%d", errors="coerce")
    stations["station_id"] = stations["usaf"].fillna("").str.zfill(6) + stations["wban"].fillna("").str.zfill(5)
    stations["icao"] = stations["icao"].astype("string").str.strip().str.upper()
    stations["call"] = stations["call"].astype("string").str.strip().str.upper()
    stations["state"] = stations["state"].astype("string").str.strip().str.upper()
    stations["country"] = stations["country"].astype("string").str.strip().str.upper()
    stations["station_name"] = stations["station_name"].astype("string").str.strip()
    stations = stations[stations["station_id"].str.len() == 11].copy()
    return stations


def _station_name_has_code(station_name: pd.Series, airport: str) -> pd.Series:
    pattern = rf"(?:^|[^A-Z]){airport}(?:[^A-Z]|$)"
    return station_name.fillna("").str.upper().str.contains(pattern, regex=True)


def _read_cached_noaa_payload(raw_path: Path) -> list[dict] | None:
    if not raw_path.exists():
        return None
    text = raw_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _station_has_year_data(station_id: str, year: int, config: PipelineConfig) -> bool:
    raw_path = config.raw_noaa_dir / station_id / f"{station_id}_{year}.json"
    payload = _read_cached_noaa_payload(raw_path)
    if payload is not None:
        return len(payload) > 0

    fallback_raw = Path.cwd() / "raw" / "noaa" / station_id / f"{station_id}_{year}.json"
    if fallback_raw.resolve() != raw_path.resolve():
        fallback_payload = _read_cached_noaa_payload(fallback_raw)
        if fallback_payload is not None:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(fallback_raw.read_bytes())
            return len(fallback_payload) > 0

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
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(response.content)
    sleep_briefly(config.sleep_seconds_between_requests)
    payload = _read_cached_noaa_payload(raw_path) or []
    return len(payload) > 0


def _score_station_candidate(airport: str, airport_state: str | None, station_row: pd.Series) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    icao_value = station_row.get("icao")
    call_value = station_row.get("call")
    name_value = station_row.get("station_name")
    state_value = station_row.get("state")
    country_value = station_row.get("country")

    icao = "" if pd.isna(icao_value) else str(icao_value)
    call = "" if pd.isna(call_value) else str(call_value)
    station_name = "" if pd.isna(name_value) else str(name_value).upper()
    station_state = "" if pd.isna(state_value) else str(state_value)

    if icao == airport:
        score += 120
        reasons.append("icao_exact")
    elif len(icao) == 4 and icao.endswith(airport):
        score += 100
        reasons.append("icao_suffix")
    if call == airport:
        score += 90
        reasons.append("call_exact")
    if airport in station_name:
        score += 25
        reasons.append("name_code_match")
    if "AIRPORT" in station_name or "INTL" in station_name or "FIELD" in station_name:
        score += 10
        reasons.append("aviation_name")
    if airport_state and airport_state == station_state:
        score += 20
        reasons.append("state_match")
    if ("" if pd.isna(country_value) else str(country_value)) == "US":
        score += 5
        reasons.append("us_station")
    if pd.notna(station_row.get("end_date")):
        score += 5
        reasons.append("has_period_of_record")

    return score, ",".join(reasons)


def _score_station_candidate_for_years(
    airport: str,
    airport_state: str | None,
    station_row: pd.Series,
    years: list[int],
    config: PipelineConfig,
) -> tuple[float, str]:
    score, details = _score_station_candidate(airport, airport_state, station_row)
    begin_date = station_row.get("begin_date")
    end_date = station_row.get("end_date")
    begin_year = int(begin_date.year) if pd.notna(begin_date) else None
    end_year = int(end_date.year) if pd.notna(end_date) else None

    if years:
        min_year = min(years)
        max_year = max(years)
        if begin_year is not None and begin_year > max_year:
            score -= 500
            details = f"{details},begins_after_target".strip(",")
        if end_year is not None and end_year < min_year:
            score -= 500
            details = f"{details},ends_before_target".strip(",")
        if (begin_year is None or begin_year <= max_year) and (end_year is None or end_year >= min_year):
            score += 40
            details = f"{details},active_in_target_period".strip(",")

        availability_hits = 0
        for year in years:
            if _station_has_year_data(str(station_row["station_id"]), year, config):
                availability_hits += 1
        if availability_hits:
            score += 80 + 20 * availability_hits
            details = f"{details},has_noaa_data".strip(",")
        else:
            score -= 400
            details = f"{details},no_noaa_data".strip(",")

    return score, details


def map_airports_to_stations(flights: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    history_path = download_station_history(config)
    stations = load_station_history(history_path)
    years = _target_years_from_flights(flights)

    airport_rows = pd.concat(
        [
            flights[["Origin", "OriginState"]].rename(columns={"Origin": "airport", "OriginState": "state"}),
            flights[["Dest", "DestState"]].rename(columns={"Dest": "airport", "DestState": "state"}),
        ],
        ignore_index=True,
    ).dropna()
    airport_rows["airport"] = airport_rows["airport"].astype("string").str.upper()
    airport_rows["state"] = airport_rows["state"].astype("string").str.upper()
    airport_rows = airport_rows.drop_duplicates().sort_values(["airport", "state"]).reset_index(drop=True)

    results: list[dict[str, object]] = []
    for row in airport_rows.itertuples(index=False):
        airport = row.airport
        airport_state = row.state
        exact_candidates = stations[
            (
                (stations["icao"] == airport)
                | (stations["call"] == airport)
                | (
                    (stations["icao"].fillna("").str.len() == 4)
                    & (stations["icao"].fillna("").str.endswith(airport))
                )
            )
        ].copy()
        if airport_state:
            exact_same_state = exact_candidates[exact_candidates["state"] == airport_state].copy()
            if not exact_same_state.empty:
                exact_candidates = exact_same_state

        candidates = exact_candidates
        if candidates.empty:
            name_candidates = stations[_station_name_has_code(stations["station_name"], airport)].copy()
            if airport_state:
                same_state = name_candidates[name_candidates["state"] == airport_state].copy()
                if not same_state.empty:
                    name_candidates = same_state
            candidates = name_candidates

        us_candidates = candidates[candidates["country"] == "US"].copy()
        if not us_candidates.empty:
            candidates = us_candidates

        if candidates.empty:
            results.append({
                "airport": airport,
                "state": airport_state,
                "station_id": pd.NA,
                "station_name": pd.NA,
                "distance_km": pd.NA,
                "mapping_confidence": "unmatched",
                "confidence_score": 0.0,
                "latitude": pd.NA,
                "longitude": pd.NA,
                "timezone": pd.NA,
                "match_details": "no_candidate_found",
            })
            LOGGER.warning("No NOAA station candidate found for airport %s", airport)
            continue

        scored = []
        for candidate in candidates.itertuples(index=False):
            candidate_series = pd.Series(candidate._asdict())
            score, details = _score_station_candidate_for_years(airport, airport_state, candidate_series, years, config)
            scored.append((score, details, candidate_series))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, details, best = scored[0]
        confidence = "high" if best_score >= 110 else "medium" if best_score >= 70 else "low"
        results.append({
            "airport": airport,
            "state": airport_state,
            "station_id": best["station_id"],
            "station_name": best["station_name"],
            "distance_km": 0.0 if confidence in {"high", "medium"} else pd.NA,
            "mapping_confidence": confidence,
            "confidence_score": best_score,
            "latitude": best["latitude"],
            "longitude": best["longitude"],
            "timezone": pd.NA,
            "match_details": details,
        })

    mapping = pd.DataFrame(results).sort_values("airport").reset_index(drop=True)
    parquet_path = config.processed_dir / "airport_station_mapping.parquet"
    mapping.to_parquet(parquet_path, index=False)
    LOGGER.info("Saved airport-to-station mapping to %s", parquet_path)
    return mapping
