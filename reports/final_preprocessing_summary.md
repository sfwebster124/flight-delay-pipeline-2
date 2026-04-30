# Final Preprocessing Summary

- Train/test split date: 2025-07-11
- Train rows used: 1,425,317
- Test rows used: 371,336
- Categorical imputation: most frequent value.
- Numeric imputation: median value.
- Linear and logistic models use one-hot encoding plus numeric standardization.
- Tree, boosting, and oversampling pipelines use ordinal encoding for categorical features.

## Engineered Features
- origin_avg_dep_delay_prev_hour: average departure delay at the same origin during the prior clock hour, shifted so only past hours are used.
- carrier_avg_dep_delay_prev_hour: average departure delay for the same carrier during the prior clock hour.
- route_avg_dep_delay_prev_hour: average departure delay for the same route during the prior clock hour.
- origin_delay_rate_prev_3h: rolling three-hour mean of departure-delay rate for the origin, computed from prior hours only.
- carrier_delay_rate_prev_3h: rolling three-hour mean of departure-delay rate for the carrier, computed from prior hours only.
- airport_carrier_avg_dep_delay_prior: cumulative average departure delay for the origin-carrier pair using only prior flights in time order.
- precip_peak_interaction: origin precipitation multiplied by a peak-hour flag to capture weather effects during heavy traffic windows.
- is_weekend, is_holiday, and season: calendar features derived from FlightDate.
- log1p_* features: log-transformed versions of clearly right-skewed nonnegative numeric predictors, selected from the training split only.
- Dropped all-missing numeric features for the active dataset subset: previous_leg_delay_available, previous_leg_arr_delay_minutes, origin_dew_point_c, dest_temp_c, dest_wind_speed_mps, dest_visibility_m, dest_precip_mm, dest_dew_point_c, dest_humidity_pct, dest_ceiling_m.

## Highest Missingness Features (Train)
|                                     |           0 |
|:------------------------------------|------------:|
| origin_ceiling_m                    | 0.00293408  |
| carrier_route_avg_dep_delay_prior   | 0.00266257  |
| route_avg_dep_delay_prev_hour       | 0.00194694  |
| origin_hourly_avg_dep_delay_prior   | 0.00155053  |
| origin_wind_speed_mps               | 0.000338872 |
| airport_carrier_avg_dep_delay_prior | 0.000185924 |
| origin_avg_dep_delay_prev_hour      | 9.19094e-05 |
| origin_delay_rate_prev_3h           | 9.19094e-05 |
| origin_visibility_m                 | 7.36678e-05 |
| carrier_avg_dep_delay_prev_hour     | 1.4032e-05  |
| carrier_delay_rate_prev_3h          | 1.4032e-05  |
| Reporting_Airline                   | 0           |
| scheduled_departure_hour_local      | 0           |
| scheduled_arrival_hour_local        | 0           |
| Dest                                | 0           |

## Most Skewed Numeric Features (Train)
| feature                             |      skew |
|:------------------------------------|----------:|
| origin_humidity_pct                 | 111.952   |
| precip_peak_interaction             |  37.9656  |
| origin_precip_mm                    |  32.5359  |
| log1p_precip_peak_interaction       |  16.0705  |
| log1p_origin_precip_mm              |   9.34564 |
| route_avg_dep_delay_prev_hour       |   8.85435 |
| origin_temp_c                       |   8.6995  |
| carrier_avg_dep_delay_prev_hour     |   7.96769 |
| airport_carrier_avg_dep_delay_prior |   7.71835 |
| origin_avg_dep_delay_prev_hour      |   7.29958 |
| carrier_route_avg_dep_delay_prior   |   5.91363 |
| is_holiday                          |   5.47691 |
| log1p_is_holiday                    |   5.47691 |
| origin_visibility_m                 |  -4.32843 |
| origin_hourly_avg_dep_delay_prior   |   3.94913 |