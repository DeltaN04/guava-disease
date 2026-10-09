# Guava Fruit Disease Classification (GFDD24)

Course project by **Nihar Ranjan Mishra, Aaviraj Nikumbh, Neel Lakhani**:
classify guava fruit photos into *Anthracnose / Fruit Flies / Healthy* and
compare a custom CNN trained from scratch (**Model A**) against two-phase
transfer learning with ImageNet-pretrained MobileNetV2 (**Model B**).

Seed: 42 everywhere. Machine used: CPU-only Mac (Apple Silicon), so the
default image size is 224x224 (see `DECISIONS.md` D1).

## Setup

```bash
cd guava-disease
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Dataset: copy the released `GuavaDiseaseDataset/{train,val,test}/` folders
(class subfolders `Anthracnose/`, `fruit_fly/`, `healthy_guava/`) to
`./Dataset/`. (Already staged in this repo's working copy.)

## Run each stage

```bash
./venv/bin/python scripts/audit_dataset.py        # Phase 1 -> outputs/audit/
./venv/bin/python train.py --model both           # Phase 3 -> outputs/models/, outputs/logs/
./venv/bin/python train.py --model transfer --head-only --tag frozen   # ablation (a)
./venv/bin/python train.py --model transfer --no-augment --tag noaug   # ablation (b)
./venv/bin/python train.py --model both --data-dir Dataset_leakfree --tag leakfree  # ablation (c)
./venv/bin/python evaluate.py                     # Phase 4 -> outputs/results/, outputs/plots/
./venv/bin/python scripts/gradcam.py              # Phase 4 Grad-CAM panels
./venv/bin/python scripts/make_figures.py         # slide-ready figures
```

Smoke test (1 epoch, tiny subset, checks save/reload parity):

```bash
./venv/bin/python train.py --model both --subset 64 --epochs-custom 1 --head-epochs 1 --ft-epochs 1
```

## Reproduce the results table and slides

```bash
./venv/bin/python evaluate.py          # rebuilds outputs/results/results_table.{csv,tex}
cd report && ../bin/tectonic slides.tex -o slides.pdf   # or: tectonic slides.tex
```

## Layout

```
config.py data_pipeline.py models.py train.py evaluate.py
scripts/{audit_dataset,gradcam,make_figures}.py
Dataset/  Dataset_leakfree/  outputs/{models,logs,plots,audit,results}/
report/{survey.md,references.bib,slides.tex,slides.pdf}
PROGRESS.md  DECISIONS.md  README.md
```
