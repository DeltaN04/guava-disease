# Error analysis (generated from metrics JSON)

Source: `outputs/results/*_metrics.json`, `outputs/results/experiments.csv`, `outputs/results/stats.json`. Every number below is read from an executed run; no value is typed by hand.

## Final configurations (E6, seeds 42-44)

| Run | Split | Best val acc | Test acc | Macro F1 | Minutes |
|---|---|---|---|---|---|
| custom_final-s42 | leak-free | 0.9737 | 0.9923 | 0.9932 | 144.7 |
| custom_final-s43 | leak-free | 0.9697 | 0.9847 | 0.9830 | 59.5 |
| custom_final-s44 | leak-free | 0.9684 | 0.9413 | 0.9392 | 77.9 |
| transfer_EfficientNetB0_final-s42 | leak-free | 0.9908 | 0.9949 | 0.9950 | 18.8 |
| transfer_EfficientNetB0_final-s43 | leak-free | 0.9921 | 0.9898 | 0.9884 | 28.5 |
| transfer_EfficientNetB0_final-s44 | leak-free | 0.9895 | 1.0000 | 1.0000 | 28.6 |

## Multi-seed summary

| Group | Seeds | Accuracy mean +/- std | Macro F1 mean +/- std |
|---|---|---|---|
| custom_final | 3 | 0.9728 +/- 0.0275 | 0.9718 +/- 0.0287 |
| transfer_EfficientNetB0_final | 3 | 0.9949 +/- 0.0051 | 0.9945 +/- 0.0058 |

## Per-class metrics: custom_final-s42

Test set: `Dataset_leakfree` (n=392)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anthracnose | 1.0000 | 0.9812 | 0.9905 | 160 |
| Fruit Flies | 0.9784 | 1.0000 | 0.9891 | 136 |
| Healthy | 1.0000 | 1.0000 | 1.0000 | 96 |

Largest off-diagonal confusions:

- Anthracnose -> Fruit Flies: 3 images

## Per-class metrics: custom_final-s43

Test set: `Dataset_leakfree` (n=392)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anthracnose | 1.0000 | 0.9938 | 0.9969 | 160 |
| Fruit Flies | 0.9851 | 0.9706 | 0.9778 | 136 |
| Healthy | 0.9596 | 0.9896 | 0.9744 | 96 |

Largest off-diagonal confusions:

- Fruit Flies -> Healthy: 4 images
- Healthy -> Fruit Flies: 1 images
- Anthracnose -> Fruit Flies: 1 images

## Per-class metrics: custom_final-s44

Test set: `Dataset_leakfree` (n=392)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anthracnose | 1.0000 | 0.9375 | 0.9677 | 160 |
| Fruit Flies | 0.9318 | 0.9044 | 0.9179 | 136 |
| Healthy | 0.8727 | 1.0000 | 0.9320 | 96 |

Largest off-diagonal confusions:

- Fruit Flies -> Healthy: 13 images
- Anthracnose -> Fruit Flies: 9 images
- Anthracnose -> Healthy: 1 images

## Per-class metrics: transfer_EfficientNetB0_final-s42

Test set: `Dataset_leakfree` (n=392)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anthracnose | 1.0000 | 0.9875 | 0.9937 | 160 |
| Fruit Flies | 0.9927 | 1.0000 | 0.9963 | 136 |
| Healthy | 0.9897 | 1.0000 | 0.9948 | 96 |

Largest off-diagonal confusions:

- Anthracnose -> Healthy: 1 images
- Anthracnose -> Fruit Flies: 1 images

## Per-class metrics: transfer_EfficientNetB0_final-s43

Test set: `Dataset_leakfree` (n=392)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anthracnose | 1.0000 | 0.9938 | 0.9969 | 160 |
| Fruit Flies | 1.0000 | 0.9779 | 0.9888 | 136 |
| Healthy | 0.9600 | 1.0000 | 0.9796 | 96 |

Largest off-diagonal confusions:

- Fruit Flies -> Healthy: 3 images
- Anthracnose -> Healthy: 1 images

## Per-class metrics: transfer_EfficientNetB0_final-s44

Test set: `Dataset_leakfree` (n=392)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anthracnose | 1.0000 | 1.0000 | 1.0000 | 160 |
| Fruit Flies | 1.0000 | 1.0000 | 1.0000 | 136 |
| Healthy | 1.0000 | 1.0000 | 1.0000 | 96 |

Largest off-diagonal confusions:

- none

## Paired statistical tests

Bootstrap: 2,000 resamples, seed 42. McNemar: exact two-sided binomial on paired correctness. A difference is called noise when p >= 0.05; a significant difference spanning at most two images is flagged separately.

| A | B | n | Acc A | Acc B | Delta | b/c | p | Verdict |
|---|---|---|---|---|---|---|---|---|
| custom_final-s42 | transfer_EfficientNetB0_final-s42 | 392 | 0.9923 | 0.9949 | +0.0026 | 2/1 | 1.0000 | within noise at alpha=0.05 |
| custom_final-s43 | transfer_EfficientNetB0_final-s43 | 392 | 0.9847 | 0.9898 | +0.0051 | 6/4 | 0.7539 | within noise at alpha=0.05 |
| custom_final-s44 | transfer_EfficientNetB0_final-s44 | 392 | 0.9413 | 1.0000 | +0.0587 | 23/0 | 0.0000 | significant |

## What the numbers do and do not support

- The released GFDD24 split is 100% leaked (454/473 originals span splits); released test accuracy is a near-duplicate recognition score, not a field-generalisation estimate.
- The leak-free split removes same-fruit duplicates (0/473 clusters span splits). It is the primary comparison.
- Final A/B differences are interpreted only through the bootstrap intervals and exact McNemar tests above; a non-significant difference is reported as within noise.
- The old `custom_base` and `transfer_MobileNetV2_base` checkpoints/logs were clobbered by a smoke test (DECISIONS D14). Their metrics JSONs remain provenance only and are not used for paired tests.
- Model A is CPU-limited at 224px; E5 uses 8 epochs (cap 12) and AdamW/RMSprop were cut (DECISIONS D13). These are limitations, not hidden results.

