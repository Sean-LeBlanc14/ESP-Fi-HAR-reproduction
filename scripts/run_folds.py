#!/usr/bin/env python3
"""Run the paper's evaluation protocols and save one JSON per fold.

Each finished fold is written immediately and skipped on rerun, so a crash,
sleep, or Ctrl-C only loses the fold in progress.

Examples
--------
# Check against the upstream smoke test (Meeting Room, person 1 held out; got 81.4%)
python -u scripts/run_folds.py --protocol loso --models ResNet18 --scenes 3 --persons 1

# Table 4, ResNet18 + EfficientNet-Lite, all rooms, 3 seeds
python -u scripts/run_folds.py --protocol loso --models ResNet18 EfficientNetLite --seeds 666 1 2

# Fig. 12 cross-scene
python -u scripts/run_folds.py --protocol cross_scene --models ResNet18 EfficientNetLite
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esp_fi_repro.data import load_mats  # noqa: E402
from esp_fi_repro.splits import cross_scene, loso  # noqa: E402
from esp_fi_repro.train import MODELS, run  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["loso", "cross_scene"], required=True)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), required=True)
    ap.add_argument("--scenes", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--persons", nargs="+", type=int, default=list(range(1, 9)),
                    help="held-out persons (loso only)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[666])
    ap.add_argument("--batch-size", type=int, default=4, help="upstream code uses 4; README suggests 32")
    ap.add_argument("--epochs", type=int, default=None, help="default: upstream per-model value")
    ap.add_argument("--data-root", default="full_data",
                    help="folder the rar was extracted into (default: full_data/ in the repo)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    X, y, meta = load_mats(args.data_root)
    print(f"loaded {len(X)} samples from {args.data_root}", flush=True)
    commit = git_commit()

    jobs = []
    for model in args.models:
        for seed in args.seeds:
            for scene in args.scenes:
                if args.protocol == "loso":
                    for person in args.persons:
                        jobs.append((model, seed, scene, person))
                else:
                    jobs.append((model, seed, scene, None))

    for i, (model, seed, scene, person) in enumerate(jobs, 1):
        fold = f"scene{scene}" + (f"_person{person}" if person else "")
        path = os.path.join(args.out, args.protocol, model, f"bs{args.batch_size}",
                            f"seed{seed}", fold + ".json")
        if os.path.exists(path):
            print(f"[{i}/{len(jobs)}] skip {path} (done)", flush=True)
            continue
        tr, te = loso(meta, scene, person) if person else cross_scene(meta, scene)
        print(f"[{i}/{len(jobs)}] {model} seed {seed} {fold}: train {len(tr)} test {len(te)}", flush=True)
        t0 = time.time()
        res = run(X, y, tr, te, model, batch_size=args.batch_size, seed=seed,
                  epochs=args.epochs, log=lambda s: print(s, flush=True))
        res.update(protocol=args.protocol, model=model, seed=seed, scene=scene,
                   test_person=person, batch_size=args.batch_size, git_commit=commit,
                   seconds=round(time.time() - t0, 1))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(path + ".tmp", path)
        print(f"    -> acc {res['acc']:.4f} f1 {res['macro_f1']:.4f} "
              f"(best epoch {res['best_epoch']}, {res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
