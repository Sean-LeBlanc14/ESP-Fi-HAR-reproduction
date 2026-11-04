#!/usr/bin/env python3
"""Summarize results/ into paper-style tables next to the paper's numbers.

Within-scene (Table 4): for each model, batch size, seed and scene, take mean
and std of accuracy over the held-out persons, then average those over seeds.
The paper does not say whether its std is sample or population; we use
sample std (ddof=1) and print how many folds each cell has.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

SCENE_NAMES = {1: "Corridor", 2: "Office", 3: "Meeting", 4: "Lab"}
# Table 4: accuracy mean, std (%) per scene 1-4
PAPER_T4 = {
    "LSTM": [(32.15, 13.40), (38.02, 14.21), (36.20, 12.04), (32.73, 13.25)],
    "GRU": [(30.93, 3.78), (38.02, 8.63), (34.55, 4.21), (28.38, 8.98)],
    "ResNet18": [(63.59, 14.91), (70.11, 19.60), (65.53, 18.22), (58.33, 22.73)],
    "EfficientNetLite": [(61.60, 10.60), (63.74, 8.74), (61.98, 5.14), (50.35, 6.96)],
    "CNN": [(57.55, 11.43), (48.31, 8.02), (48.26, 11.14), (52.61, 12.19)],
    "Transformer": [(54.43, 12.30), (41.93, 9.47), (47.51, 10.71), (50.33, 15.11)],
    "MobileNetV3": [(53.64, 13.24), (51.82, 11.85), (53.28, 15.18), (43.31, 16.25)],
}
# Fig. 12: cross-scene accuracy (%) per test scene 1-4 (Office ResNet18 given as "around 70")
PAPER_F12 = {
    "ResNet18": [61.05, 70.0, 72.97, 75.52],
    "EfficientNetLite": [72.57, 76.22, 75.17, 69.79],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    rows = [json.load(open(p)) for p in glob.glob(os.path.join(args.results, "**", "*.json"), recursive=True)]
    if not rows:
        print("no results found"); return

    loso = defaultdict(list)
    cross = defaultdict(list)
    for r in rows:
        key = (r["model"], r["batch_size"], r["scene"])
        if r["protocol"] == "loso":
            loso[key + (r["seed"],)].append(100 * r["acc"])
        else:
            cross[key].append(100 * r["acc"])

    if loso:
        print("\nWithin-scene LOSO accuracy %  (ours mean±std [folds x seeds]  |  paper)")
        cells = defaultdict(list)
        for (m, bs, sc, seed), accs in loso.items():
            sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else float("nan")
            cells[(m, bs, sc)].append((np.mean(accs), sd, len(accs)))
        for m, bs in sorted({(k[0], k[1]) for k in cells}):
            print(f"\n{m}  (batch {bs})")
            for sc in range(1, 5):
                if (m, bs, sc) not in cells:
                    continue
                v = cells[(m, bs, sc)]
                mean = np.mean([a for a, _, _ in v]); sd = np.nanmean([s for _, s, _ in v])
                folds = "/".join(str(n) for _, _, n in v)
                ref = PAPER_T4.get(m, [None] * 4)[sc - 1]
                ref_s = f"{ref[0]:.2f} ± {ref[1]:.2f}" if ref else "-"
                print(f"  {SCENE_NAMES[sc]:<9} {mean:6.2f} ± {sd:5.2f}  [{folds} x {len(v)}]   |  {ref_s}")

    if cross:
        print("\nCross-scene accuracy %  (ours mean over seeds [n]  |  paper)")
        for m, bs in sorted({(k[0], k[1]) for k in cross}):
            print(f"\n{m}  (batch {bs})")
            for sc in range(1, 5):
                if (m, bs, sc) in cross:
                    v = cross[(m, bs, sc)]
                    ref = PAPER_F12.get(m, [None] * 4)[sc - 1]
                    print(f"  test {SCENE_NAMES[sc]:<9} {np.mean(v):6.2f}  [{len(v)}]   |  {ref if ref else '-'}")


if __name__ == "__main__":
    main()
