# Poster Significance Tests

## F1
- ANOVA F-statistic: 3.8037
- ANOVA p-value: 0.039765

| group1                       | group2                 |   meandiff |   p-adj |   lower |   upper | reject   |
|:-----------------------------|:-----------------------|-----------:|--------:|--------:|--------:|:---------|
| logistic_regression_balanced | mlp_classifier         |    -0.1012 |  0.2672 | -0.2568 |  0.0545 | False    |
| logistic_regression_balanced | random_forest_balanced |     0.0167 |  0.9883 | -0.139  |  0.1724 | False    |
| logistic_regression_balanced | xgboost_classifier     |    -0.1285 |  0.1196 | -0.2842 |  0.0271 | False    |
| mlp_classifier               | random_forest_balanced |     0.1178 |  0.1658 | -0.0379 |  0.2735 | False    |
| mlp_classifier               | xgboost_classifier     |    -0.0274 |  0.9521 | -0.1831 |  0.1283 | False    |
| random_forest_balanced       | xgboost_classifier     |    -0.1452 |  0.0704 | -0.3009 |  0.0105 | False    |

## PR_AUC
- ANOVA F-statistic: 0.1447
- ANOVA p-value: 0.931133

| group1                       | group2                 |   meandiff |   p-adj |   lower |   upper | reject   |
|:-----------------------------|:-----------------------|-----------:|--------:|--------:|--------:|:---------|
| logistic_regression_balanced | mlp_classifier         |     0.0082 |  0.9978 | -0.1279 |  0.1444 | False    |
| logistic_regression_balanced | random_forest_balanced |     0.0212 |  0.9661 | -0.115  |  0.1573 | False    |
| logistic_regression_balanced | xgboost_classifier     |     0.0273 |  0.9318 | -0.1089 |  0.1635 | False    |
| mlp_classifier               | random_forest_balanced |     0.0129 |  0.9918 | -0.1233 |  0.1491 | False    |
| mlp_classifier               | xgboost_classifier     |     0.019  |  0.9749 | -0.1171 |  0.1552 | False    |
| random_forest_balanced       | xgboost_classifier     |     0.0061 |  0.9991 | -0.1301 |  0.1423 | False    |

## ROC_AUC
- ANOVA F-statistic: 1.0554
- ANOVA p-value: 0.403991

| group1                       | group2                 |   meandiff |   p-adj |   lower |   upper | reject   |
|:-----------------------------|:-----------------------|-----------:|--------:|--------:|--------:|:---------|
| logistic_regression_balanced | mlp_classifier         |     0.0053 |  0.9463 | -0.0237 |  0.0344 | False    |
| logistic_regression_balanced | random_forest_balanced |     0.0126 |  0.5898 | -0.0165 |  0.0416 | False    |
| logistic_regression_balanced | xgboost_classifier     |     0.0158 |  0.4086 | -0.0133 |  0.0449 | False    |
| mlp_classifier               | random_forest_balanced |     0.0072 |  0.8797 | -0.0218 |  0.0363 | False    |
| mlp_classifier               | xgboost_classifier     |     0.0105 |  0.7149 | -0.0186 |  0.0395 | False    |
| random_forest_balanced       | xgboost_classifier     |     0.0032 |  0.9872 | -0.0259 |  0.0323 | False    |
