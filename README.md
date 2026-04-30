# Flight Delay Prediction Project

This folder is a clean GitHub-ready export of the final project from `DSCI Project 2`. It contains:

- the source code used to build and analyze the project
- the final filtered modeling dataset used in the final workflow
- small non-visual summary outputs for the paper/poster results

Raw downloads, large intermediates, caches, and visual assets are intentionally excluded.

## Project Overview

The project predicts U.S. domestic flight delays using:

- BTS On-Time Performance flight records
- NOAA FM-15 / METAR-style hourly weather observations

The main tasks are:

- binary classification for departure delays of 15+ minutes
- late-only regression for departure delay severity
- imbalance handling comparison
- PCA comparison
- supplemental poster analysis including clustering, neural network, significance testing, and LLM comparison

## Repository Layout

- `src/`: project source code
- `data/`: final included modeling parquet and data note
- `reports/`: small non-visual result summaries only

## Data Pipeline

The pipeline is:

**BTS raw flights + NOAA raw weather**
-> `bts_downloader.py` and `noaa_downloader.py`
-> airport-to-station mapping with `station_mapper.py`
-> weather/flight joins and feature building in `build_features.py`
-> filtering to valid FM-15 weather-covered rows and handling missing values
-> optional subset selection such as strict FM-15 rows or top-origin subsets
-> final modeling parquet used by the analysis scripts

For this GitHub package, the included final dataset is already the strict FM-15 top-25 subset:

- `data/modeling_dataset_fm15_strict_top25.parquet`
- `data/airport_station_mapping.parquet`
- rows: `1,796,653`

See `data/README.md` for the dataset-specific note.

## Methodology & Justification

- **Missing handling:** categorical variables use most-frequent imputation and numeric variables use median imputation so models can train consistently without dropping large numbers of rows.
- **Encoding:** logistic and MLP models use one-hot encoding because they need explicit categorical expansion; tree-based models use ordinal encoding because split-based learners can work efficiently with integer-coded categories.
- **Scaling:** scaling is applied for linear models, neural networks, and PCA because their optimization and variance structure depend on feature scale.
- **Feature choice:** schedule, calendar, weather, and propagation features were included because delay formation depends on time-of-day effects, local weather, and prior congestion/delay spillover.
- **Model choice:** logistic regression provides an interpretable baseline, random forest captures nonlinear interactions with strong recall, and XGBoost provides a stronger boosting-based ranking model.
- **Metrics:** F1 and PR-AUC are emphasized because the delayed class is imbalanced; ROC-AUC is included as a standard ranking metric, but PR-AUC is more informative for the minority delayed class.

## Statistical Testing

The poster workflow uses:

- one-way **ANOVA** across temporal-fold model metrics
- **Tukey HSD** for pairwise follow-up comparisons

Why:

- multiple supervised models were compared across repeated temporal folds, so ANOVA is a reasonable overall significance test before pairwise post-hoc comparisons.

Included statistical summary:

- `reports/statistical_tests_summary.md`

## Install Requirements

```powershell
pip install -r requirements.txt
```

Recommended Windows setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Final Modeling

Exact command:

```powershell
python src/final_model_analysis.py
```

That command runs the included strict top-25 parquet because the packaged script defaults were adjusted for this repo layout.

## Data Limits, Splits, and Runtime Modes

- **Included dataset:** `data/modeling_dataset_fm15_strict_top25.parquet`
- **Included row count:** `1,796,653`
- **Default temporal split:** the packaged workflow splits by date, with training on earlier flights and testing on later flights.
- **Current split point in the included dataset:** `2025-07-11`
- **Current split sizes:** `1,425,317` train rows and `371,336` test rows

### Default `final_model_analysis.py` run

- uses the full included parquet for loading and filtering
- then caps model-training runtime with sampled subsets
- **sampled train size:** `250,000`
- **sampled test size:** `100,000`
- applies to:
  - classification models
  - regression models
  - imbalance comparison
  - PCA logistic comparison

### Full-data `final_model_analysis.py` run

- command:

```powershell
python src/final_model_analysis.py --full-filtered-data
```

- uses the same temporal split
- trains and evaluates on **all rows after filtering**
- no `250k / 100k` sample cap
- intended for slower but more complete final runs

### Poster workflow

- command:

```powershell
python src/poster_analysis.py --use-full-data
```

- uses the same included strict top-25 dataset
- uses temporal fold-based model comparison on the full filtered dataset
- intended for poster-level comparison outputs such as:
  - model comparison summaries
  - significance testing
  - clustering
  - interpretability outputs

### LLM workflow

- command:

```powershell
python src/llm_row_prediction.py
```

- does **not** score the full dataset row-by-row by default
- uses a temporal split first, then evaluates a smaller held-out sample of test rows
- intended as an exploratory prompt-based comparison rather than a full tabular training workflow

To reproduce the full final-model run on all included rows:

```powershell
python src/final_model_analysis.py --full-filtered-data
```

## Reproduce Poster Supplemental Results

```powershell
python src/poster_analysis.py --use-full-data
```

## Rebuild the Dataset from Raw Data

Raw data are not included in this GitHub package, but the dataset can be regenerated by:

1. downloading BTS monthly flight files with `src/bts_downloader.py`
2. downloading NOAA hourly weather with `src/noaa_downloader.py`
3. mapping airports to stations with `src/station_mapper.py`
4. joining flights and weather with `src/build_features.py`
5. building the strict FM-15 weather-covered dataset
6. filtering to the top 25 origin airports used in the final project

The original project folder used these steps to produce the final included parquet.

## Final Dataset Size Check

Included parquet:

- `data/modeling_dataset_fm15_strict_top25.parquet`
- size: about `56.1 MB`

Decision:

- this is under 100 MB, so standard Git storage is acceptable and Git LFS is not required for this packaged version

## Included Summary Outputs

The `reports/` folder contains small non-visual summaries only:

- classification metrics
- regression metrics
- imbalance comparison
- PCA comparison
- preprocessing summary
- poster model summary
- poster significance tests
- poster hyperparameter summary
- poster LLM analysis
- statistical tests summary

## Excluded Contents

This GitHub package excludes:

- raw downloads
- intermediate processed parquets
- caches
- notebook outputs
- Python cache folders
- visual figures

One exception is the final airport bubble-map figure, which can now be regenerated from the included airport-to-station mapping file.

That keeps the repository smaller and focused on code, the final included dataset, and compact summary outputs.
