# Reproducing ESP-Fi HAR

Reproduction of Wen et al., "ESP-Fi HAR: A low-power WiFi CSI dataset for Ad-Hoc IoT human activity recognition", Ad Hoc Networks 186 (2026) 104192. The upstream code in `Model Code/` is left unchanged, and the models are imported from `Model Code/ESP_Fi_model.py`.

## Data

1. `git lfs pull`
2. `unrar x ESP-Fi_dataset.rar full_data/` (inside the repo; `full_data/` is gitignored)
   Gives `EnvironmentNo.<k>(<room>)/{mat,csv}/X-Y-Z-M.*`

## Naming `X-Y-Z-M`

- **X**: Scene (1 = corridor, 2 = office, 3 = boardroom, 4 = laboratory)
- **Y**: Person (1–8)
- **Z**: Activity
- **M**: Trial (1–10)

The paper and the original README assign activity IDs differently:

| ID | Paper | README |
| --- | --- | --- |
| 1 | Run | Run |
| 2 | Walk | Fall |
| 3 | Jump | Walk |
| 4 | Squat | Turn |
| 5 | Arm wave | Jump |
| 6 | Turn | Squat |
| 7 | Fall | Arm wave |

This reproduction follows the paper's activity IDs.

## Protocols

| Paper result | Command |
| --- | --- |
| Table 4, within-scene 8-fold leave-one-person-out | `python -u scripts/run_folds.py --protocol loso --models ResNet18` |
| Summary vs paper | `python scripts/aggregate.py` |

Results go to `results/<protocol>/<model>/bs<batch>/seed<seed>/<fold>.json` (gitignored).

## Training settings (same as upstream run.py)

AdamW lr 1e-3, weight decay 1e-4; cosine LR over the run; cross-entropy;
per-sample z-score; 50 epochs (CNN, ResNet18, MobileNetV3, EfficientNetLite)
or 100 (GRU, LSTM, Transformer); checkpoint = first epoch with the best
training accuracy; seed 666; `cudnn.benchmark=True`.

Open choices, not stated in the paper:
- Std across folds: sample vs population unknown; we report sample std.

## Reference runs

| Date | What | Result |
| --- | --- | --- |
| 2026-11-02 | Upstream `run.py --model ResNet18` on shipped fold (Meeting Room, person 1 out), RTX 2000 Ada laptop | acc 81.43%, macro-F1 81.78%, best epoch 48 |

## Results

Within-scene LOSO accuracy %  (ours mean±std [folds x seeds]  |  paper)

ResNet18  (batch 4):
| Reproduced | Paper |
| --- | --- |
| Corridor   57.86 ± 11.76  [8 x 1]   |  63.59 ± 14.91 |
| Office     73.57 ±  8.05  [8 x 1]   |  70.11 ± 19.60 |
| Meeting    73.93 ±  5.60  [8 x 1]   |  65.53 ± 18.22 |
| Lab        53.57 ±  9.60  [8 x 1]   |  58.33 ± 22.73 |

ResNet18  (batch 32):
| Reproduced | Paper |
| --- | --- |
| Corridor   56.96 ±  9.33  [8 x 1]   |  63.59 ± 14.91 |
| Office     69.64 ± 13.80  [8 x 1]   |  70.11 ± 19.60 |
| Meeting    73.21 ±  9.47  [8 x 1]   |  65.53 ± 18.22 |
| Lab        49.29 ± 13.14  [8 x 1]   |  58.33 ± 22.73 |

