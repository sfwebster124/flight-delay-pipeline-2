# Final Imbalance Summary

| experiment                     |   precision |   recall |       f1 |   roc_auc |   pr_auc |
|:-------------------------------|------------:|---------:|---------:|----------:|---------:|
| logistic_no_balancing          |    0.587791 | 0.32617  | 0.419536 |  0.724695 | 0.519468 |
| logistic_class_weight_balanced |    0.374968 | 0.787944 | 0.508127 |  0.724906 | 0.51741  |
| logistic_smotenc               |    0.369899 | 0.777563 | 0.501315 |  0.707568 | 0.478603 |

Oversampling experiment used SMOTENC after train/test splitting and inside an imblearn pipeline to avoid leakage.
