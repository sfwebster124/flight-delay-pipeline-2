# Statistical Tests Summary

- Test design: one-way **ANOVA** across temporal-fold model metrics, followed by **Tukey HSD** pairwise comparisons.
- Why this test: multiple supervised models were compared across repeated temporal folds, so ANOVA was used to test overall differences before pairwise follow-up testing.

## F1

- ANOVA F-statistic: `1.9015`
- ANOVA p-value: `0.183219`
- Significant at alpha=0.05: `no`
- Interpretation: the fold-level F1 differences were not statistically significant across the compared supervised models.

## PR-AUC

- ANOVA F-statistic: `5.4121`
- ANOVA p-value: `0.013765`
- Significant at alpha=0.05: `yes`
- Tukey HSD significant pairs:
  - logistic_regression_balanced vs mlp_classifier
  - logistic_regression_balanced vs random_forest_balanced
  - logistic_regression_balanced vs xgboost_classifier
- Interpretation: PR-AUC differences were statistically significant, and logistic regression underperformed the other supervised models on this metric in the poster fold study.

## ROC-AUC

- ANOVA F-statistic: `31.9859`
- ANOVA p-value: `0.000005`
- Significant at alpha=0.05: `yes`
- Tukey HSD significant pairs:
  - logistic_regression_balanced vs mlp_classifier
  - logistic_regression_balanced vs random_forest_balanced
  - logistic_regression_balanced vs xgboost_classifier
- Interpretation: ROC-AUC differences were statistically significant, with logistic regression again performing worse than the other supervised models in the temporal-fold comparison.

## Source

This summary is derived from the included poster significance output:

- `reports/significance_tests.md`
