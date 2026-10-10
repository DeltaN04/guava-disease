# Error analysis (Phase 4) — real test numbers, seed 42, 224px CPU build

Source: `outputs/results/{custom_base,transfer_MobileNetV2_base,...}_metrics.json`,
`outputs/plots/*_cm*.png`, `outputs/logs/*.csv`. All numbers from executed runs.

## Headline (released split, test 382)

| Run | Test acc | Macro F1 | Macro AUC | Best val acc | Train time |
|---|---|---|---|---|---|
| A custom CNN (1.24M params) | 0.9110 | 0.9173 | 0.9949 | 0.9364 | 31.6 min (9 epochs, ES) |
| B MobileNetV2 TL (2.59M) | 0.9921 | 0.9911 | 0.9999 | 0.9960 | 14.5 min (15+20) |
| B frozen head-only | 0.9948 | 0.9942 | 0.9999 | 0.9907 | 6.2 min (15) |
| B no-augment (+FT) | 0.9948 | 0.9946 | 0.9999 | 0.9974 | 13.1 min (15+18) |

B beats A by ~8pp test accuracy. All transfer variants saturate ≥0.99.

## Confusion details

- A (`[[130,26,0],[1,130,1],[0,6,88]]`, classes Anthracnose/fruit_fly/healthy):
  dominant error is **26 Anthracnose → Fruit Flies**; 6 Healthy → Fruit Flies;
  only 1 Fruit-Flies miss. Per-class F1: Anthracnose ~0.90, Fruit Flies ~0.93,
  Healthy ~0.93 (see metrics JSON). Anthracnose lesions vs fruit-fly puncture
  marks are the visually confusable pair (dark spots/rot), consistent with
  Grad-CAM checks on B.
- B (`[[155,0,1],[0,131,1],[0,1,93]]`): 3 errors total, no systematic confusion.

## Curves / overfitting

- A: train acc 0.83→0.97 over 9 epochs; val 0.928 (ep1) → dip 0.71/0.58
  (ep3-4) → recover 0.936 (ep8). Val loss best 0.176 (ep1); ES + restore-best
  kept ep8 weights. Classic small-data wobble; Dropout 0.25/0.5 + BN momentum
  0.9 + augmentation contained it, but the gap says A is at capacity.
- B: head warm-up already 0.992 val by ep11; fine-tune lifts best to 0.996
  (ep19) with train/val tracking (loss 0.006-0.02). No overfitting signal;
  ReduceLROnPlateau fired twice, ES never triggered (ran full 15+20).

## Ablations say

- (a) Frozen-only B (0.9948) slightly *beats* full-FT B (0.9921) on the leaked
  test: with near-duplicate train/test views, the ImageNet head alone already
  separates; fine-tuning adds nothing here. Do not over-claim — gap is 1-2
  images out of 382.
- (b) No-augment B (0.9948, val 0.9974) matches augmented B: same reason —
  the x8 release augmentation already covers the variation; extra in-graph
  augmentation is redundant on the leaked split. It may matter more on
  leak-free/field data (untested here beyond the control below).

## Leakage control (honest, partial)

- Released split is leaked: 454/473 originals span splits; 100% of val/test
  share an original with train (`outputs/audit/`). Released scores are
  near-duplicate recognition (supports Tampu et al. 2022 inflation thesis).
- Leak-free re-split (`Dataset_leakfree/`, cluster-grouped, 0/473 span):
  custom-leakfree ckpt (interrupted 17/40, best val 0.9737) scores test
  **0.9923 / F1 0.9932** (392 imgs); transfer-leakfree (15+17, ES, val
  0.9855) scores **0.9872 / F1 0.9872** — both *match or beat* released
  scores. No visible inflation because the task saturates (~0.99) and the
  leak-free runs trained fully/longer. The control still matters
  methodologically (proves the result is not pure memorization), and field
  photos from new orchards would be the real test.

## A vs B verdict

Transfer wins on accuracy (0.992 vs 0.911), training time (14.5 vs 31.6 min),
and stability. From-scratch CNN is a valid baseline but at capacity on 2.6k
images. Single seed 42 throughout — no multi-seed stats (documented scope cut).
