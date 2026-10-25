# Paired statistics (final runs, real predictions)

Bootstrap: 2000 resamples, seed 42, percentile 95% CI. McNemar: exact binomial two-sided on paired correctness.

| Run | Test set | n | Acc | 95% bootstrap CI | Wilson CI | Macro F1 |
|---|---|---|---|---|---|---|
| custom_final-s42 | Dataset_leakfree | 392 | 0.9923 | [0.9821, 1.0000] | [0.9777, 0.9974] | 0.9932 |
| custom_final-s43 | Dataset_leakfree | 392 | 0.9847 | [0.9719, 0.9949] | [0.9670, 0.9930] | 0.9830 |
| custom_final-s44 | Dataset_leakfree | 392 | 0.9413 | [0.9184, 0.9643] | [0.9135, 0.9606] | 0.9392 |
| transfer_EfficientNetB0_final-s42 | Dataset_leakfree | 392 | 0.9949 | [0.9872, 1.0000] | [0.9816, 0.9986] | 0.9950 |
| transfer_EfficientNetB0_final-s43 | Dataset_leakfree | 392 | 0.9898 | [0.9796, 0.9974] | [0.9741, 0.9960] | 0.9884 |
| transfer_EfficientNetB0_final-s44 | Dataset_leakfree | 392 | 1.0000 | [1.0000, 1.0000] | [0.9903, 1.0000] | 1.0000 |

## Paired comparisons

| A | B | n | Acc A | Acc B | Delta (B-A) | b/c | McNemar p | Verdict |
|---|---|---|---|---|---|---|---|---|
| custom_final-s42 | transfer_EfficientNetB0_final-s42 | 392 | 0.9923 | 0.9949 | +0.0026 | 2/1 | 1.0000 | within noise at alpha=0.05 |
| custom_final-s43 | transfer_EfficientNetB0_final-s43 | 392 | 0.9847 | 0.9898 | +0.0051 | 6/4 | 0.7539 | within noise at alpha=0.05 |
| custom_final-s44 | transfer_EfficientNetB0_final-s44 | 392 | 0.9413 | 1.0000 | +0.0587 | 23/0 | 0.0000 | significant |

## Multi-seed final configurations

| Group | Seeds | Acc mean +/- std | Macro F1 mean +/- std |
|---|---|---|---|
| custom_final | 3 | 0.9728 +/- 0.0275 | 0.9718 +/- 0.0287 |
| transfer_EfficientNetB0_final | 3 | 0.9949 +/- 0.0051 | 0.9945 +/- 0.0058 |
