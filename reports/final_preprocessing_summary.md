# Final Preprocessing Summary

- Train/test split date: 2025-07-11
- Train rows used: 250,000
- Test rows used: 100,000
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
|                                     |        0 |
|:------------------------------------|---------:|
| origin_ceiling_m                    | 0.002828 |
| carrier_route_avg_dep_delay_prior   | 0.00258  |
| route_avg_dep_delay_prev_hour       | 0.001808 |
| origin_hourly_avg_dep_delay_prior   | 0.001564 |
| origin_wind_speed_mps               | 0.000364 |
| airport_carrier_avg_dep_delay_prior | 0.000164 |
| origin_visibility_m                 | 6e-05    |
| origin_avg_dep_delay_prev_hour      | 6e-05    |
| origin_delay_rate_prev_3h           | 6e-05    |
| carrier_avg_dep_delay_prev_hour     | 2.8e-05  |
| carrier_delay_rate_prev_3h          | 2.8e-05  |
| Reporting_Airline                   | 0        |
| scheduled_departure_hour_local      | 0        |
| scheduled_arrival_hour_local        | 0        |
| Dest                                | 0        |

## Most Skewed Numeric Features (Train)
| feature                             |      skew |
|:------------------------------------|----------:|
| origin_humidity_pct                 | 122.37    |
| precip_peak_interaction             |  35.671   |
| origin_precip_mm                    |  32.9058  |
| log1p_precip_peak_interaction       |  15.9333  |
| log1p_origin_precip_mm              |   9.29854 |
| carrier_avg_dep_delay_prev_hour     |   8.61182 |
| route_avg_dep_delay_prev_hour       |   8.51286 |
| origin_temp_c                       |   7.94807 |
| origin_avg_dep_delay_prev_hour      |   6.74233 |
| carrier_route_avg_dep_delay_prior   |   5.61436 |
| is_holiday                          |   5.50317 |
| log1p_is_holiday                    |   5.50317 |
| origin_hourly_avg_dep_delay_prior   |   5.44569 |
| airport_carrier_avg_dep_delay_prior |   4.90571 |
| origin_visibility_m                 |  -4.35432 |