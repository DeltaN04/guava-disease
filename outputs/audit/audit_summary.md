# Dataset audit summary (generated -- do not hand-edit)

Class folders on disk: Anthracnose, fruit_fly, healthy_guava

## Per-split, per-class counts

|  | Anthracnose | fruit_fly | healthy_guava |
|---|---|---|---|
| test | 156 | 132 | 94 |
| train | 1080 | 918 | 649 |
| val | 308 | 262 | 185 |

Totals: {'test': 382, 'train': 2647, 'val': 755} -- expected {'train': 2647, 'val': 775, 'test': 382}.

Corrupt/unreadable files: 0

Image sizes: {(512, 512): 3784}
Modes: {'RGB': 3784}
Formats: {'PNG': 3784}

## Cross-split near-duplicate (leakage) analysis

phash hamming threshold: 6

|  | exact | near |
|---|---|---|
| train-val | 0 | 8 |
| train-test | 0 | 5 |
| val-test | 0 | 3 |

Filename prefixes spanning splits: 454
Val files sharing an original with train: 755
Test files sharing an original with train: 382
