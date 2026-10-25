# Paired statistics (final runs, real predictions)

Bootstrap: 2000 resamples, seed 42, percentile 95% CI. McNemar: exact binomial two-sided on paired correctness.

| Run | Test set | n | Acc | 95% bootstrap CI | Wilson CI | Macro F1 |
|---|---|---|---|---|---|---|
| custom_leakfree | Dataset_leakfree | 392 | 0.9923 | [0.9821, 1.0000] | [0.9777, 0.9974] | 0.9932 |
| transfer_MobileNetV2_leakfree | Dataset_leakfree | 392 | 0.9872 | [0.9745, 0.9974] | [0.9705, 0.9945] | 0.9872 |
| transfer_MobileNetV2_frozen | Dataset | 382 | 0.9948 | [0.9869, 1.0000] | [0.9811, 0.9986] | 0.9942 |
| transfer_MobileNetV2_noaug | Dataset | 382 | 0.9948 | [0.9869, 1.0000] | [0.9811, 0.9986] | 0.9946 |

## Paired comparisons

| A | B | n | Acc A | Acc B | Delta (B-A) | b/c | McNemar p | Verdict |
|---|---|---|---|---|---|---|---|---|
| custom_leakfree | transfer_MobileNetV2_leakfree | 392 | 0.9923 | 0.9872 | -0.0051 | 3/5 | 0.7266 | within noise at alpha=0.05 |
| transfer_MobileNetV2_frozen | transfer_MobileNetV2_noaug | 382 | 0.9948 | 0.9948 | +0.0000 | 1/1 | 1.0000 | within noise at alpha=0.05 |
