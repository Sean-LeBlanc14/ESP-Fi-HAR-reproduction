"""Load the released ESP-Fi HAR amplitude .mat files.

Files live at <data_root>/EnvironmentNo.<k>(<room>)/mat/X-Y-Z-M.mat, where
X = scene (1 corridor, 2 office, 3 meeting room, 4 laboratory),
Y = person 1-8, Z = activity 1-7, M = trial 1-10.

Activity IDs follow the paper's Table 3 (the repo README's list is wrong):
1 run, 2 walk, 3 jump, 4 squat, 5 arm wave, 6 turn, 7 fall.

Preprocessing matches the authors' dataset.py: per-sample z-score over the
whole 950 x 52 matrix, output shape (1, 950, 52) float32.
"""

import glob
import os

import numpy as np
import scipy.io as sio

SCENES = {1: "corridor", 2: "office", 3: "meeting_room", 4: "laboratory"}
ACTIVITIES = {1: "run", 2: "walk", 3: "jump", 4: "squat", 5: "arm_wave", 6: "turn", 7: "fall"}
SHAPE = (950, 52)


def zscore(x):
    """Authors' normalization: one mean and std per sample."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def load_mats(data_root):
    """Return X (N, 1, 950, 52) float32, y (N,) int64 in 0..6, meta (N, 4) int16.

    meta columns: scene, person, activity, trial. y = activity - 1.
    """
    root = os.path.expanduser(data_root)
    paths = sorted(glob.glob(os.path.join(root, "*", "mat", "*.mat")))
    if not paths:
        raise FileNotFoundError(f"no .mat files under {root}/*/mat/")
    X, meta = [], []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        ids = tuple(int(v) for v in name.split("-"))
        if len(ids) != 4:
            raise ValueError(f"unexpected file name {p}")
        x = sio.loadmat(p)["CSIamp"]
        if x.shape == SHAPE[::-1]:
            x = x.T
        if x.shape != SHAPE:
            raise ValueError(f"shape {x.shape} in {p}")
        X.append(zscore(x).reshape(1, *SHAPE).astype(np.float32))
        meta.append(ids)
    meta = np.array(meta, dtype=np.int16)
    y = (meta[:, 2] - 1).astype(np.int64)
    return np.stack(X), y, meta
