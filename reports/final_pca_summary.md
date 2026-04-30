# Final PCA Summary

- PCA selected 20 numeric components to explain 0.9520 of variance.
- Compared against the existing balanced logistic regression without PCA, the PCA variant hurt PR-AUC performance.
- PCA reduces dimensionality in the numeric feature block but also reduces interpretability because the transformed components are no longer original variables.

## Comparison Table
| experiment                            |       f1 |   roc_auc |   pr_auc |   pca_components |   numeric_variance_explained |
|:--------------------------------------|---------:|----------:|---------:|-----------------:|-----------------------------:|
| logistic_regression_balanced_no_pca   | 0.508127 |  0.724906 | 0.51741  |              nan |                   nan        |
| logistic_regression_balanced_with_pca | 0.508601 |  0.722685 | 0.514759 |               20 |                     0.952025 |

## Figure
- `reports/figures/final_pca_variance_curve.png`