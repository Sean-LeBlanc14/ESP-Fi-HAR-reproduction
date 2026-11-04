"""Train/test splits used in the paper.

within-scene LOSO (Table 4): for one scene, test on one person, train on the
other seven. 8 folds per scene; the paper reports mean and std over folds.

cross-scene (Fig. 12): train on three scenes, test on the fourth. All eight
people appear on both sides, so this measures room shift only.
"""

import numpy as np


def loso(meta, scene, test_person):
    in_scene = meta[:, 0] == scene
    test = in_scene & (meta[:, 1] == test_person)
    train = in_scene & ~test
    return np.flatnonzero(train), np.flatnonzero(test)


def cross_scene(meta, test_scene):
    test = meta[:, 0] == test_scene
    return np.flatnonzero(~test), np.flatnonzero(test)
