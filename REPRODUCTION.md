# Reproducing ESP-Fi HAR

Reproduction of Wen et al., "ESP-Fi HAR: A low-power WiFi CSI dataset for Ad-Hoc IoT human activity recognition", Ad Hoc Networks 186 (2026) 104192. The upstream code in `Model Code/` is left unchanged, and the models are imported from `Model Code/ESP_Fi_model.py`.

## Data

1. `git lfs pull`
2. `unrar x ESP-Fi_dataset.rar full_data/` (inside the repo; `full_data/` is gitignored)
   Gives `EnvironmentNo.<k>(<room>)/{mat,csv}/X-Y-Z-M.*`

Naming (X-Y-X-M):
- X: Scene (1 corridor, 2 office, 3 boardrooms, 4 labratory)
- Y: Person (1 - 8)
- Z: Activity
- M: Trial (1 - 10)
There is a discrepancy between the README and paper. In the paper, activity IDs are given by
- 1: Run
- 2: Walk
- 3: Jump
- 4: Squat
- 5: Arm Wave
- 6: Turn
- 7: Fall
In the README, activity IDs are given by
- 1: Run
- 2: Fall
- 3: Walk
- 4: Turn
- 5: Jump
- 6: Squat
- 7: Arm wave
In this reproduction, we follow the paper's listing of activity IDs.

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
- Batch size: upstream code uses 4, upstream README recommends 32. Default 4.
- Std across folds: sample vs population unknown; we report sample std.

## Reference runs

| Date | What | Result |
| --- | --- | --- |
| 2026-11-02 | Upstream `run.py --model ResNet18` on shipped fold (Meeting Room, person 1 out), RTX 2000 Ada laptop | acc 81.43%, macro-F1 81.78%, best epoch 48 |

## Results (so far)

Within-scene LOSO accuracy %  (ours mean±std [folds x seeds]  |  paper)

ResNet18  (batch 4)
  Corridor   57.86 ± 11.76  [8 x 1]   |  63.59 ± 14.91
  Office     73.57 ±  8.05  [8 x 1]   |  70.11 ± 19.60
  Meeting    73.93 ±  5.60  [8 x 1]   |  65.53 ± 18.22
  Lab        53.57 ±  9.60  [8 x 1]   |  58.33 ± 22.73

ResNet18  (batch 32)
  Office     69.64 ± 13.80  [8 x 1]   |  70.11 ± 19.60

