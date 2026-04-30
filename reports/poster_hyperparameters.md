# Poster Hyperparameter Summary

- The current poster experiments use fixed, manually specified hyperparameters rather than a full grid or random search.
- This should be presented as targeted tuning based on runtime constraints and prior baseline results, not as exhaustive optimization.

| model                        | search_method         | key_hyperparameters                                                                                  |
|:-----------------------------|:----------------------|:-----------------------------------------------------------------------------------------------------|
| logistic_regression_balanced | manual fixed settings | solver=lbfgs; max_iter=1200; class_weight=balanced                                                   |
| random_forest_balanced       | manual fixed settings | n_estimators=120; max_depth=14; min_samples_leaf=20; class_weight=balanced_subsample                 |
| xgboost_classifier           | manual fixed settings | n_estimators=120; max_depth=6; learning_rate=0.08; subsample=0.8; colsample_bytree=0.8               |
| mlp_classifier               | manual fixed settings | hidden_layer_sizes=(64,32); alpha=0.0005; learning_rate_init=0.001; max_iter=80; early_stopping=True |