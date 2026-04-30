# Included Dataset Note

This folder contains the final included modeling file:

- `modeling_dataset_fm15_strict_top25.parquet`

## Pipeline Summary

**BTS raw flights + NOAA raw weather**
-> download scripts
-> airport-to-station mapping
-> flight/weather joins
-> FM-15 validity filtering and missing handling
-> strict weather-covered subset selection
-> top-25 origin filtering
-> final modeling parquet

## Step-by-Step

1. BTS monthly domestic flight files were downloaded.
2. NOAA FM-15 / METAR-style hourly weather observations were downloaded.
3. Airports were mapped to NOAA weather stations.
4. Weather observations were aligned to flight schedule times using a three-hour tolerance window.
5. Flights with valid strict FM-15 weather coverage were retained for the strict subset.
6. The strict subset was filtered to the top 25 origin airports used in the final modeling workflow.
7. The final filtered dataset was written as `modeling_dataset_fm15_strict_top25.parquet`.

## What This File Represents

- source family: BTS flights + NOAA FM-15 weather
- modeling role: final included dataset for the GitHub package
- row count: `1,796,653`

## Dataset Size Decision

- file size: about `56.1 MB`
- because the file is below 100 MB, it was kept directly in the package
- Git LFS is not required for this exported version

## What Is Not Included

This GitHub package does not include:

- raw BTS ZIP files
- raw NOAA JSON station downloads
- intermediate processed parquets
- cache files

Those were intentionally excluded to keep the repository smaller and cleaner.
