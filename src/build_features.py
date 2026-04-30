"""Feature engineering, weather joins, and quality reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def _parse_hhmm_to_minutes(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    if "." in text:
        text = text.split(".", maxsplit=1)[0]
    text = text.zfill(4)
    if not text.isdigit():
        return np.nan
    hour = int(text[:2])
    minute = int(text[2:])
    if hour == 24 and minute == 0:
        return 0.0
    if hour > 23 or minute > 59:
        return np.nan
    return float(hour * 60 + minute)


def _local_timestamp_from_date_and_minutes(date_series: pd.Series, minutes_series: pd.Series) -> pd.Series:
    base = pd.to_datetime(date_series, errors="coerce")
    minute_offsets = pd.to_timedelta(minutes_series.fillna(0), unit="m")
    result = base + minute_offsets
    return result.where(minutes_series.notna())


def prepare_flight_timestamps(flights: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    frame = flights.copy()
    mapping_subset = mapping[["airport", "station_id", "timezone"]].copy()
    origin_map = mapping_subset.rename(columns={"airport": "Origin", "station_id": "origin_station_id", "timezone": "origin_timezone"})
    dest_map = mapping_subset.rename(columns={"airport": "Dest", "station_id": "dest_station_id", "timezone": "dest_timezone"})
    frame = frame.merge(origin_map, on="Origin", how="left")
    frame = frame.merge(dest_map, on="Dest", how="left")

    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="coerce")
    frame["crs_dep_minutes"] = frame["CRSDepTime"].map(_parse_hhmm_to_minutes)
    frame["crs_arr_minutes"] = frame["CRSArrTime"].map(_parse_hhmm_to_minutes)
    frame["sched_dep_local_naive"] = _local_timestamp_from_date_and_minutes(frame["FlightDate"], frame["crs_dep_minutes"])
    frame["sched_arr_local_naive"] = _local_timestamp_from_date_and_minutes(frame["FlightDate"], frame["crs_arr_minutes"])

    frame["sched_dep_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame["sched_arr_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for timezone_name in frame["origin_timezone"].dropna().unique().tolist():
        mask = frame["origin_timezone"] == timezone_name
        localized = frame.loc[mask, "sched_dep_local_naive"].dt.tz_localize(timezone_name, nonexistent="shift_forward", ambiguous="NaT")
        frame.loc[mask, "sched_dep_utc"] = localized.dt.tz_convert("UTC")
    for timezone_name in frame["dest_timezone"].dropna().unique().tolist():
        mask = frame["dest_timezone"] == timezone_name
        localized = frame.loc[mask, "sched_arr_local_naive"].dt.tz_localize(timezone_name, nonexistent="shift_forward", ambiguous="NaT")
        frame.loc[mask, "sched_arr_utc"] = localized.dt.tz_convert("UTC")

    frame["sched_dep_utc"] = pd.to_datetime(frame["sched_dep_utc"], utc=True, errors="coerce")
    frame["sched_arr_utc"] = pd.to_datetime(frame["sched_arr_utc"], utc=True, errors="coerce")

    needs_roll = frame["sched_arr_utc"].notna() & frame["sched_dep_utc"].notna() & (frame["sched_arr_utc"] < frame["sched_dep_utc"] - pd.Timedelta(hours=2))
    if needs_roll.any():
        frame.loc[needs_roll, "sched_arr_local_naive"] = frame.loc[needs_roll, "sched_arr_local_naive"] + pd.Timedelta(days=1)
        for timezone_name in frame.loc[needs_roll, "dest_timezone"].dropna().unique().tolist():
            mask = needs_roll & (frame["dest_timezone"] == timezone_name)
            localized = frame.loc[mask, "sched_arr_local_naive"].dt.tz_localize(timezone_name, nonexistent="shift_forward", ambiguous="NaT")
            frame.loc[mask, "sched_arr_utc"] = localized.dt.tz_convert("UTC")
    frame["sched_arr_utc"] = pd.to_datetime(frame["sched_arr_utc"], utc=True, errors="coerce")

    frame["scheduled_departure_hour_local"] = (frame["crs_dep_minutes"] // 60).astype("Int64")
    frame["scheduled_arrival_hour_local"] = (frame["crs_arr_minutes"] // 60).astype("Int64")
    frame["month"] = frame["FlightDate"].dt.month.astype("Int64")
    frame["day_of_week"] = frame["FlightDate"].dt.dayofweek.add(1).astype("Int64")
    frame["route"] = frame["Origin"].astype("string") + "_" + frame["Dest"].astype("string")
    return frame


def _merge_weather_asof(flights: pd.DataFrame, weather: pd.DataFrame, station_column: str, prefix: str, ts_column: str) -> pd.DataFrame:
    flight_subset = flights.reset_index().copy()
    flight_subset[station_column] = flight_subset[station_column].astype("string")
    flight_subset[ts_column] = pd.to_datetime(flight_subset[ts_column], utc=True, errors="coerce")
    weather = weather.copy()
    weather["station_id"] = weather["station_id"].astype("string")
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True, errors="coerce")
    valid_mask = flight_subset[station_column].notna() & flight_subset[ts_column].notna()
    valid_flights = flight_subset.loc[valid_mask].sort_values([ts_column, station_column]).copy()
    weather_subset = weather.sort_values(["timestamp_utc", "station_id"]).copy()
    merged = pd.merge_asof(
        valid_flights,
        weather_subset,
        left_on=ts_column,
        right_on="timestamp_utc",
        left_by=station_column,
        right_by="station_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=3),
    )
    rename_map = {
        "timestamp_utc": f"{prefix}_weather_timestamp_utc",
        "temperature_c": f"{prefix}_temp_c",
        "dew_point_c": f"{prefix}_dew_point_c",
        "wind_speed_mps": f"{prefix}_wind_speed_mps",
        "wind_direction_deg": f"{prefix}_wind_direction_deg",
        "visibility_m": f"{prefix}_visibility_m",
        "sea_level_pressure_hpa": f"{prefix}_sea_level_pressure_hpa",
        "precip_mm": f"{prefix}_precip_mm",
        "weather_code": f"{prefix}_weather_code",
        "relative_humidity_pct": f"{prefix}_humidity_pct",
        "ceiling_m": f"{prefix}_ceiling_m",
    }
    merged = merged.rename(columns=rename_map)
    keep = ["index"] + list(rename_map.values())
    result = flight_subset[["index"]].merge(merged[keep], on="index", how="left")
    return result


def attach_weather_features(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    origin_weather = _merge_weather_asof(flights, weather, "origin_station_id", "origin", "sched_dep_utc")
    dest_weather = _merge_weather_asof(flights, weather, "dest_station_id", "dest", "sched_arr_utc")
    enriched = flights.reset_index().merge(origin_weather, on="index", how="left").merge(dest_weather, on="index", how="left")
    return enriched.drop(columns=["index"])


def add_leakage_safe_operational_features(flights: pd.DataFrame) -> pd.DataFrame:
    frame = flights.copy().sort_values("sched_dep_utc").reset_index(drop=True)
    frame["DepDelay"] = pd.to_numeric(frame["DepDelay"], errors="coerce").fillna(0.0)
    frame["ArrDelay"] = pd.to_numeric(frame["ArrDelay"], errors="coerce")

    airport_keys = ["Origin", "day_of_week", "scheduled_departure_hour_local"]
    airport_count_prior = frame.groupby(airport_keys, dropna=False).cumcount()
    airport_cumsum_prior = frame.groupby(airport_keys, dropna=False)["DepDelay"].cumsum() - frame["DepDelay"]
    frame["origin_hourly_avg_dep_delay_prior"] = airport_cumsum_prior / airport_count_prior.replace(0, np.nan)

    route_keys = ["Reporting_Airline", "route"]
    route_count_prior = frame.groupby(route_keys, dropna=False).cumcount()
    route_cumsum_prior = frame.groupby(route_keys, dropna=False)["DepDelay"].cumsum() - frame["DepDelay"]
    frame["carrier_route_avg_dep_delay_prior"] = route_cumsum_prior / route_count_prior.replace(0, np.nan)

    hourly_group = frame.groupby(["Origin", frame["sched_dep_utc"].dt.floor("h")], dropna=False)
    frame["origin_hourly_congestion_proxy_prior"] = hourly_group.cumcount()

    if "Tail_Number" in frame.columns:
        tail_group = frame.groupby("Tail_Number", dropna=False)
        frame["previous_leg_arr_delay_minutes"] = tail_group["ArrDelay"].shift(1)
        frame["previous_leg_delay_available"] = frame["previous_leg_arr_delay_minutes"].notna().astype("int8")
    else:
        frame["previous_leg_arr_delay_minutes"] = np.nan
        frame["previous_leg_delay_available"] = 0
    return frame


def build_modeling_dataset(flights: pd.DataFrame, weather: pd.DataFrame, mapping: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    prepared = prepare_flight_timestamps(flights, mapping)
    prepared["DepDelay"] = pd.to_numeric(prepared["DepDelay"], errors="coerce")
    prepared["ArrDelay"] = pd.to_numeric(prepared["ArrDelay"], errors="coerce")
    prepared["dep_delay_minutes"] = prepared["DepDelay"]
    prepared["arr_delay_minutes"] = prepared["ArrDelay"]
    prepared["dep_delayed_15"] = (prepared["DepDelay"] >= 15).astype("Int64")
    prepared["arr_delayed_15"] = (prepared["ArrDelay"] >= 15).astype("Int64")

    with_weather = attach_weather_features(prepared, weather)
    featured = add_leakage_safe_operational_features(with_weather)

    final_columns = [
        "FlightDate", "Reporting_Airline", "DOT_ID_Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
        "Origin", "Dest", "route", "Distance", "CRSDepTime", "CRSArrTime", "scheduled_departure_hour_local",
        "scheduled_arrival_hour_local", "day_of_week", "month", "Cancelled", "CancellationCode", "Diverted",
        "TaxiOut", "TaxiIn", "AirTime", "dep_delay_minutes", "arr_delay_minutes", "dep_delayed_15", "arr_delayed_15",
        "previous_leg_delay_available", "previous_leg_arr_delay_minutes", "origin_hourly_avg_dep_delay_prior",
        "carrier_route_avg_dep_delay_prior", "origin_hourly_congestion_proxy_prior", "origin_temp_c",
        "origin_wind_speed_mps", "origin_visibility_m", "origin_precip_mm", "origin_weather_code", "origin_dew_point_c",
        "origin_humidity_pct", "origin_ceiling_m", "dest_temp_c", "dest_wind_speed_mps", "dest_visibility_m",
        "dest_precip_mm", "dest_weather_code", "dest_dew_point_c", "dest_humidity_pct", "dest_ceiling_m",
        "origin_station_id", "dest_station_id", "sched_dep_utc", "sched_arr_utc", "origin_weather_timestamp_utc",
        "dest_weather_timestamp_utc",
    ]
    for column in final_columns:
        if column not in featured.columns:
            featured[column] = pd.NA
    final = featured[final_columns].copy()
    final_parquet = config.processed_dir / "modeling_dataset.parquet"
    final.to_parquet(final_parquet, index=False)
    LOGGER.info("Saved final modeling dataset to %s", final_parquet)
    return final


def build_quality_report(
    flights: pd.DataFrame,
    mapping: pd.DataFrame,
    weather: pd.DataFrame,
    modeling: pd.DataFrame,
    config: PipelineConfig,
) -> Path:
    duplicate_subset = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"]
    duplicate_count = int(flights.duplicated(subset=duplicate_subset).sum())
    malformed_dep_times = int(flights["CRSDepTime"].map(_parse_hhmm_to_minutes).isna().sum())
    malformed_arr_times = int(flights["CRSArrTime"].map(_parse_hhmm_to_minutes).isna().sum())
    missing_weather_mapping = int(mapping["station_id"].isna().sum())
    origin_weather_join_rate = float(modeling["origin_temp_c"].notna().mean()) if len(modeling) else 0.0
    dest_weather_join_rate = float(modeling["dest_temp_c"].notna().mean()) if len(modeling) else 0.0

    carrier_summary = modeling.groupby("Reporting_Airline", dropna=False)[["dep_delayed_15", "arr_delayed_15"]].mean().sort_index().round(4)
    airport_summary = modeling.groupby("Origin", dropna=False)[["dep_delay_minutes"]].agg(["count", "mean", "median"]).round(2)

    report_path = config.reports_dir / "data_quality_report.md"
    lines = [
        "# Data Quality Report",
        "",
        "## Overview",
        f"- Flights rows: {len(flights):,}",
        f"- Weather rows: {len(weather):,}",
        f"- Modeling rows: {len(modeling):,}",
        f"- Duplicate flights: {duplicate_count:,}",
        f"- Malformed CRS departure times: {malformed_dep_times:,}",
        f"- Malformed CRS arrival times: {malformed_arr_times:,}",
        f"- Missing airport-to-station mappings: {missing_weather_mapping:,}",
        f"- Origin weather join rate: {origin_weather_join_rate:.2%}",
        f"- Destination weather join rate: {dest_weather_join_rate:.2%}",
        f"- Departure delay target positive rate: {modeling['dep_delayed_15'].mean():.2%}",
        f"- Arrival delay target positive rate: {modeling['arr_delayed_15'].mean():.2%}",
        "",
        "## Missingness",
        f"- Missing DepDelay: {flights['DepDelay'].isna().mean():.2%}",
        f"- Missing ArrDelay: {flights['ArrDelay'].isna().mean():.2%}",
        f"- Missing CarrierDelay: {flights['CarrierDelay'].isna().mean():.2%}",
        f"- Missing WeatherDelay: {flights['WeatherDelay'].isna().mean():.2%}",
        "",
        "## Carrier Summary",
        carrier_summary.to_markdown(),
        "",
        "## Origin Airport Summary",
        airport_summary.to_markdown(),
        "",
        "## Mapping Confidence Counts",
        mapping["mapping_confidence"].value_counts(dropna=False).to_markdown(),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved quality report to %s", report_path)
    return report_path
