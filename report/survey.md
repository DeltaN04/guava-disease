# Research Survey — Guava Fruit Disease Classification (GFDD24 context)

> All factual claims below were verified live on 9 Oct 2026 against the
> publisher/landing page (Mendeley / arXiv / Nature / Frontiers).
> Verification method and URL are recorded per source in
> `report/references.bib`. No paper was cited from memory alone.

## 1. The dataset and the problem

The **Guava Fruit Disease Dataset (GFDD24)** [gfdd24] was published
4 Nov 2024 (v1, DOI `10.17632/bkdkc4n835.1`) by Al Amin et al. (Hajee
Mohammad Danesh Science and Technology Univ., Bangladesh, CC BY 4.0).
Field photos from orchards in Rajshahi/Pabna, taken in July (ripening
season), pathologist-verified, into three fruit classes: **Anthracnose /
Fruit Flies / Healthy**. 473 originals, each preprocessed (unsharp mask +
CLAHE) to 512x512 RGB PNG and augmented x8 → **3,784 images**; released
splits train 2647 / val 775 / test 382. Our disk audit finds train 2647 /
val **755** / test 382 — the 20-image val shortfall is reported, not
hidden (see `outputs/audit/`). Crucially, augmentation was applied
*before* splitting: 454/473 originals span splits, so 100% of val/test
images share a source original with train — the central methodological
fact of this project (see §7).

## 2. Plant-disease CNNs: what is known

Mohanty et al. [mohanty2016plant] (Frontiers in Plant Science 2016)
established the baseline result the whole subfield builds on: AlexNet /
GoogLeNet trained on **PlantVillage (54,306 leaf images, 38 classes)**
reach **up to 99.35% accuracy** on held-out images, but accuracy drops
sharply on field photos from other sources — i.e. lab-curated accuracy
overstates field performance. That is exactly the dynamic we reproduce:
GFDD24's released test set is near-duplicate-adjacent to train, while
our leak-free re-split measures something closer to field
generalization. The survey conclusion stands: CNNs easily separate
plant-disease classes *within* a curated collection; honest
evaluation is the hard part.

## 3. Architectures

**MobileNetV2** [sandler2018mobilenetv2] (CVPR 2018, arXiv:1801.04381)
is our Model B backbone: inverted residuals (thin bottleneck in/out,
expanded depthwise filtering in the middle) plus *linear* bottlenecks
(no ReLU where channels are few, preserving representational power).
~3.4M params / ~300M MAdds at 224px — the accuracy-per-FLOP frontier
for mobile vision, hence the standard transfer-learning backbone for
small agricultural datasets. Our codebase also supports EfficientNetB0
and ResNet50 as drop-in alternatives.

Our **Model A** (custom 4-block CNN, 32/64/128/256 filters, Conv-ReLU-BN
x2 + MaxPool + Dropout per block, GAP + 256-dense head, ~1.24M params)
follows the standard small-data recipe: batch normalization
[ioffe2015bn] (ICML 2015, arXiv:1502.03167 — normalizing layer inputs
against internal covariate shift, allowing higher LR and acting as a
regularizer) and dropout [srivastava2014dropout] (JMLR 2014,
arXiv:1207.0580 — randomly omitting units, p=0.5 in dense layers,
preventing co-adaptation; we use 0.25 in conv blocks / 0.5 in the head).
Residual learning [he2016resnet] (CVPR 2016, arXiv:1512.03385 — skip
connections enabling 152-layer optimization) and compound scaling
[tan2019efficientnet] (ICML 2019, arXiv:1905.11946) are the backbone
family Model B belongs to. Both models optimize with **Adam**
[kingma2015adam] (ICLR 2015, arXiv:1412.6980 — adaptive first/second
moment estimates, default lr 0.001).

## 4. Transfer learning: why Model B should win

Yosinski et al. [yosinski2014transfer] (NeurIPS 2014, arXiv:1411.1792)
quantified it: early-layer filters (Gabor/color-blob) are general across
natural-image tasks while late layers specialize; initializing from
almost any depth and fine-tuning boosts generalization even after full
adaptation. ImageNet [russakovsky2015imagenet] (IJCV 2015,
arXiv:1409.0575 — 14M images / 21k synsets, 1k ILSVRC) is the standard
source task. Our Model B therefore warms up a frozen MobileNetV2 +
256-dense head (lr 1e-3, 15 epochs), then fine-tunes the last 30
backbone layers at lr 1e-5 with BatchNorm kept frozen — the textbook
two-phase protocol. Frozen-head-only and no-augmentation ablations
isolate the contribution of each phase.

## 5. Explainability

**Grad-CAM** [selvaraju2020gradcam] (IJCV 2020, ICCV 2017 version,
arXiv:1610.02391) uses gradients of the predicted class flowing into the
last conv layer to weight its feature maps into a coarse localization
map — no architecture change or retraining, applicable to any CNN with
fully-connected heads. We apply it to Model B's deepest backbone conv
layer on correct + misclassified test images; trustworthy models should
highlight lesions/rot rather than background.

## 6. Data leakage: the cautionary literature

Tampu et al. [tampu2022leakage] (Scientific Data 2022, DOI
10.1038/s41597-022-01618-6) is the direct precedent: splitting 2D slices
from the same OCT volume across train/test inflated accuracy by **5-30
pp** (MCC 0.07-0.43) on three public datasets. GFDD24's x8 augmentation
is structurally identical (8 views of one fruit = 8 slices of one
volume), so our 100%-overlap finding predicts the same inflation — which
the leak-free re-split (cluster-grouped 70/20/10, seed 42) is designed
to measure.

## 7. Gap this project fills

No prior work we found trains *both* a from-scratch CNN and two-phase
MobileNetV2 transfer learning on GFDD24 **with the leakage quantified
and a leak-free control reported side by side**. Papers report
curated-split accuracy; we report curated-split *and* leak-free accuracy
plus Grad-CAM evidence — the honest comparison a deployment decision
needs.

## 7. Gap this project fills

No prior work we found trains *both* a from-scratch CNN and two-phase
MobileNetV2 transfer learning on GFDD24 **with the leakage quantified
and a leak-free control reported side by side**. Papers report
curated-split accuracy; we report curated-split *and* leak-free accuracy
plus Grad-CAM evidence — the honest comparison a deployment decision
needs.

## 8. Comparative overview table (2.1 — the teacher's section)

12 disease-classification studies + GFDD24 itself. Every row verified
live 9 Oct 2026 (URLs in Verification log; BibTeX in
`report/references.bib`). Numbers are the authors' own reported results
on their own splits — included so the teacher can check them, not as
claims we reproduce. "Limitation" is what the paper itself omits or what
a careful reader should notice.

| # | Authors + year | Method family | Dataset | Reported result | Limitation |
|---|---|---|---|---|---|
| [1] | Al Amin et al. 2024 (GFDD24 data record) | Dataset release: field photos, unsharp+CLAHE 512px, x8 aug | 473 originals -> 3,784; 3 fruit classes; train 2647/val 775/test 382 | — (dataset, not a model) | Augmentation applied *before* splitting — our audit finds 100% val/test share an original with train (see §1); val on disk 755 vs 775 stated |
| [2] | Mostafa et al. 2021, Appl. Sci. 12(1):239 | 5 CNNs from scratch/transfer (AlexNet, SqueezeNet, GoogLeNet, ResNet-50/101) + augmentation | 321 guava images (5 disease classes) augmented to 2,889 | ResNet-50 best, 99.54% acc | Different 5-class leaf/fruit set, not GFDD24; heavy augmentation, no leak-free control, no XAI; black-box accuracy only |
| [3] | Doutoum et al. 2023, WSEAS Trans. Inf. Sci. Appl. 20:356-363 | Transfer learning comparison (incl. EfficientNet-B3) | 1,834 guava leaf images, 5 classes | EfficientNet-B3 94.93% test acc | No XAI; single curated split; leaf-only (no fruit) |
| [4] | Nandi et al. 2022, MIET / arXiv:2209.12557 | 5 pretrained CNNs (VGG-16, GoogLeNet, ResNet-18, MobileNet-v2, EfficientNet) + float16/dynamic quantization | ~681 originals + 8,525 augmented guava fruit+leaf, 5 classes | EfficientNet ~99% acc (4.2 MB); quantized GoogLeNet 97% (0.143 MB) | Original+augmented combined for experiments (leakage risk); no XAI; device-size focus, not generalization audit |
| [5] | Kilci & Koklu 2024, Proc. 4th Int. Conf. Frontiers Acad. Res. | Hybrid: SqueezeNet deep features + AdaBoost / Gradient Boosting | GFDD24 (3,784 fruit images, 3 classes) | Gradient Boosting 95.6%; AdaBoost 82.7% | Curated split only (no leak-free control); no XAI; specificity/NPV/FPR not reported |
| [6] | Tewari et al. 2024 (via Zhou 2026 Table 4) | DenseNet169 transfer | 527 guava images, 5 classes | 99.62% acc | Small curated set; no XAI (black box); no leakage analysis |
| [7] | Zhou 2026, Front. Plant Sci. 17:1738585 | 6 hybrid transfer combos (VGG16/MobileNetV2/InceptionV3/ResNet50/custom CNN); ImageNet pretrain + fine-tune top layers; Grad-CAM | 527 guava fruit+leaf images, 5 classes (Disease Free/Phytophthora/Red Rust/Scab/Styler+Root) | VGG16+MobileNetV2 best: 96% acc/P/R/F1 | Small set (527); 50 epochs Adam; different 5-class taxonomy from GFDD24; lab-curated |
| [8] | ResNet-101 + Grad-CAM study 2025 (PMC12719609) | Transfer: ResNet-101 + Grad-CAM | GFDD24 (Anthracnose / Fruit Fly / Healthy) | 98.48% acc, spec 99.24%, F1 0.9848 | Curated split only; 98.48% is exactly the inflation our leakage analysis predicts — no leak-free control to rule it out |
| [9] | CNN–Gradient Boosting ensemble 2025, arXiv:2512.19989 | Hybrid: EfficientNet-B0 features + RF/AdaBoost/LightGBM/XGBoost cascade, undersampling | GFDD24 | Claimed ~99.99% (preprint) | Preprint (Dec 2025); near-perfect score on a leaked split is evidence *for* the leakage hypothesis, not against it; heavy ensemble overhead |
| [10] | Mohanty et al. 2016, Front. Plant Sci. 7:1419 | AlexNet/GoogLeNet from scratch + transfer | PlantVillage: 54,306 leaves, 38 classes, 14 crops | Up to 99.35% held-out acc | Lab-curated leaves; accuracy collapses on field photos from other sources (authors' own warning) — the field-gap precedent for our work |
| [11] | Maitlo et al. 2023, Mendeley w3fg8jjmzr | Dataset release: variety/maturity grading (Canon EOS 5D, 850x1300) | 2,309 guava fruit images; 3 varieties x 3 maturity stages | — (dataset) | Grading, not disease; different task/taxonomy; included to show dataset diversity, not as a baseline |
| [12] | Ghosh et al. 2026, arXiv:2602.10967 | InceptionV3 comparison on GFDD24 | GFDD24 (473 originals, Anthracnose/Fruit Flies/Healthy) | Comparative InceptionV3 baseline (see preprint for per-split numbers) | Preprint; curated split; included as the closest architectural comparator to our Model B |
| [13] | Pratama et al. 2025 (JESSI, MobileNetV2 on guava) | — | — | — | **DROPPED 9 Oct 2026: could not verify** via web search / publisher page. Per project rules (no unverifiable citations) it is excluded; listed here so the omission is explicit, not silent. |

## 9. Categorization by methodological family (2.2)

- **Transfer-learning architectures** [2,3,4,6,7,8,10,12]: ImageNet-pretrained
  CNNs (ResNet, EfficientNet, DenseNet, MobileNetV2, InceptionV3, VGG16)
  fine-tuned on guava/plant data. Our Model B belongs here (MobileNetV2,
  frozen warm-up + last-30-layer fine-tune, BN frozen). Consistent finding:
  transfer beats from-scratch on <5k images — which is exactly what our
  A-vs-B comparison tests on GFDD24.
- **Hybrid deep-features + classical ML** [5,9]: frozen CNN as feature
  extractor (SqueezeNet, EfficientNet-B0) + boosting (AdaBoost, Gradient
  Boosting, XGBoost/LightGBM). Strong curated-split numbers (95.6%,
  ~99.99%) but no leak-free control and no localization evidence; our
  Grad-CAM panels are the missing check.
- **Data-centric techniques**: augmentation-before-split [1,2,4]
  (inflates scores — our §1 finding), quantization for edge deployment
  [4] (0.143 MB GoogLeNet at 97%), undersampling for imbalance [9],
  CLAHE/unsharp preprocessing [1] (Zuiderveld 1994). Our contribution is
  methodological hygiene: quantify the leakage [1], add the leak-free
  re-split control (Tampu et al. 2022 precedent), and report both.

## 10. Research gap and positioning (2.3 — defensible wording)

We do **not** claim "no one has used MobileNetV2 on guava" — [4] and [7]
did. The defensible gap, supported by the table above, is narrower:
*no prior GFDD24 study we could verify reports (i) a from-scratch CNN vs
two-phase transfer head-to-head, (ii) with cross-split duplicate
quantification, (iii) a leak-free re-split control, and (iv) Grad-CAM
localization on the same split.* [5,8,9] report GFDD24 accuracy (95.6–
99.99%) but all on the released (leaked) split with no control and
([5,9]) no XAI; [8] has XAI but no control; [2–4,6,7] use non-GFDD24
sets. Our work fills exactly that 4-part gap. If a verifier finds an
earlier 4-part study, the claim softens to "among the first" — the table
above gives them everything needed to check.

## Verification log (9 Oct 2026, all pages fetched live)

| Claim | Verified at |
|---|---|
| GFDD24: 473 originals, x8 = 3784, 512px RGB PNG, train 2647/val 775/test 382, DOI 10.17632/bkdkc4n835.1, CC BY 4.0 | https://data.mendeley.com/datasets/bkdkc4n835 |
| MobileNetV2 inverted residuals + linear bottlenecks, CVPR 2018, arXiv:1801.04381 | https://arxiv.org/abs/1801.04381 |
| Grad-CAM gradient-weighted last-conv maps, IJCV 2020 (ICCV 2017), arXiv:1610.02391 | https://arxiv.org/abs/1610.02391 |
| Leakage inflates accuracy 5-30pp, Sci Data 9:580 (2022) | https://www.nature.com/articles/s41597-022-01618-6 |
| PlantVillage 54,306 imgs / 38 classes / 99.35%, Front. Plant Sci. 2016 | https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full |
| Adam (ICLR 2015, arXiv:1412.6980); BN (ICML 2015, arXiv:1502.03167); transferability (NeurIPS 2014, arXiv:1411.1792) | arXiv abs pages above |
| Dropout arXiv:1207.0580; ResNet arXiv:1512.03385; EfficientNet arXiv:1905.11946; ImageNet arXiv:1409.0575 | arXiv abs pages above |
| CLAHE: Zuiderveld 1994, Graphics Gems IV, pp 474-485 | https://dl.acm.org/doi/10.5555/180895.180940 + https://github.com/erich666/GraphicsGems/blob/master/gemsiv/clahe.c |
| Mostafa et al. 2021, Appl. Sci. 12(1):239 (guava, ResNet-50 99.54%, 321->2889) | https://arxiv.org/pdf/2209.12557 (cites [14]) + https://openagriculturejournal.com/VOLUME/18/ELOCATOR/e18743315305194/FULLTEXT (cites ref 12) + Frontiers 2026 Table 4 |
| Doutoum et al. 2023, WSEAS 20:356-363, EfficientNet-B3 94.93%, 1834 leaf imgs | https://www.semanticscholar.org/paper/Guava-Leaf-Disease-Identification-Using-the-with-Verma-Raj/894b15c458ffbae727f649315b40330ca8361c63 + https://avesis.ankara.edu.tr/reryigit/yayinlar + Front. Plant Sci. 2026 Table 4 |
| Nandi et al. 2022, MIET / arXiv:2209.12557 (5 CNNs, EfficientNet 99%, GoogLeNet quant 0.143MB 97%) | https://arxiv.org/abs/2209.12557 |
| Kilci & Koklu 2024, SqueezeNet+GB 95.6% / AdaBoost 82.7% on GFDD24 | https://arxiv.org/html/2512.19989v1 (cites [3]) + https://pmc.ncbi.nlm.nih.gov/articles/PMC12719609 (Table 5) |
| Tewari et al. 2024, DenseNet169 99.62% (527 imgs) | https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1738585/full (Table 4) |
| Zhou 2026, Front. Plant Sci. 17:1738585, VGG16+MobileNetV2 96% + Grad-CAM, 527 imgs | https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1738585/full + https://pmc.ncbi.nlm.nih.gov/articles/PMC13160900 |
| ResNet-101 + Grad-CAM 98.48% on GFDD24 (2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12719609 |
| CNN-GB ensemble ~99.99% on GFDD24 (arXiv Dec 2025 preprint) | https://arxiv.org/abs/2512.19989 |
| Maitlo et al. 2023 guava grading dataset (Mendeley w3fg8jjmzr, 2309 imgs) | https://data.mendeley.com/datasets/w3fg8jjmzr/1 + https://pmc.ncbi.nlm.nih.gov/articles/PMC10412757 |
| Ghosh et al. 2026, arXiv:2602.10967, InceptionV3 on GFDD24 | https://arxiv.org/abs/2602.10967 |
| Pratama et al. 2025 (JESSI) — SEARCHED 9 Oct 2026, no publisher/DOI page found | web search "Pratama MobileNetV2 guava fruit JESSI 2025" returned only unrelated JAIC mushroom paper — DROPPED per no-unverifiable-citations rule |

