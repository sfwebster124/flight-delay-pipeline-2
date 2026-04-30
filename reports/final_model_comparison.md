# Final Model Comparison

## Classification
| model                        |   precision |   recall |       f1 |   roc_auc |   pr_auc |
|:-----------------------------|------------:|---------:|---------:|----------:|---------:|
| xgboost_classifier           |    0.617984 | 0.318412 | 0.420278 |  0.742484 | 0.547778 |
| random_forest_balanced       |    0.410197 | 0.740193 | 0.527865 |  0.738926 | 0.540533 |
| logistic_regression_balanced |    0.374968 | 0.787944 | 0.508127 |  0.724906 | 0.51741  |
| dummy_most_frequent          |    0        | 0        | 0        |  0.5      | 0.27455  |

Best classifier: `xgboost_classifier` with PR-AUC 0.5478, ROC-AUC 0.7425, and F1 0.4203.

## Regression
| model                   |     mae |    rmse |        r2 |
|:------------------------|--------:|--------:|----------:|
| random_forest_regressor | 26.7744 | 54.9842 | 0.0958783 |
| xgboost_regressor       | 26.9883 | 55.1741 | 0.0896251 |
| linear_regression       | 27.3094 | 55.7367 | 0.070963  |

Best regressor: `random_forest_regressor` with MAE 26.774, RMSE 54.984, and R^2 0.0959 on late-only departure minutes.

## Class Imbalance Comparison
| experiment                     |   precision |   recall |       f1 |   roc_auc |   pr_auc |
|:-------------------------------|------------:|---------:|---------:|----------:|---------:|
| logistic_no_balancing          |    0.587791 | 0.32617  | 0.419536 |  0.724695 | 0.519468 |
| logistic_class_weight_balanced |    0.374968 | 0.787944 | 0.508127 |  0.724906 | 0.51741  |
| logistic_smotenc               |    0.369899 | 0.777563 | 0.501315 |  0.707568 | 0.478603 |

## Figures
- `reports/figures/final_delay_minutes_histogram.png`
- `reports/figures/final_correlation_heatmap.png`
- `reports/figures/final_model_comparison_bar.png`
- `reports/figures/final_regression_model_comparison_bar.png`
- `reports/figures/final_best_classifier_roc_curve.png`
- `reports/figures/final_best_classifier_pr_curve.png`
- `reports/figures/final_feature_importance.png`
- `reports/figures/final_threshold_tradeoff.png`
- `reports/figures/final_calibration_curve.png`
- `reports/figures/final_airport_delay_bubble_map.png`