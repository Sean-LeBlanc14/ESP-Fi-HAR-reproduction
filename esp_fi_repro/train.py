"""One training run, matching the authors' run.py.

Kept identical to upstream unless a flag says otherwise:
AdamW (lr 1e-3, weight decay 1e-4), CosineAnnealingLR(T_max=epochs),
cross-entropy, checkpoint = first epoch reaching the best TRAINING accuracy
(strictly greater), evaluated once on the test set at the end.
Epochs per model as in upstream util.py. Batch size defaults to 4 (upstream
code), though the upstream README recommends 32.
"""

import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Use the authors' model definitions unchanged.
_UPSTREAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model Code")
sys.path.insert(0, _UPSTREAM)
import ESP_Fi_model as M

MODELS = {
    "CNN": (M.CNN, 50),
    "ResNet18": (M.ESP_Fi_ResNet18, 50),
    "Transformer": (M.ESP_Fi_Transformer, 100),
    "GRU": (M.ESP_Fi_GRU, 100),
    "LSTM": (M.ESP_Fi_LSTM, 100),
    "MobileNetV3": (M.MobileNetV3, 50),
    "EfficientNetLite": (M.EfficientNetLite, 50),
}


def set_seed(seed):
    """Same as upstream (cudnn.benchmark on, so runs are not bit-exact)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _loader(X, y, idx, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    preds, labels, loss = [], [], 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss += crit(out, yb).item()
        preds.append(out.argmax(1).cpu()); labels.append(yb.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"acc": float((p == t).mean()),
            "macro_f1": float(f1_score(t, p, average="macro")),
            "loss": loss / len(t), "pred": p.tolist(), "true": t.tolist()}


def run(X, y, train_idx, test_idx, model_name, batch_size=4, seed=666,
        epochs=None, lr=1e-3, device=None, log=print):
    """Train one model on train_idx, return final test metrics and per-epoch history."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cls, default_epochs = MODELS[model_name]
    epochs = epochs or default_epochs
    model = cls(7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    tr = _loader(X, y, train_idx, batch_size, True)
    te = _loader(X, y, test_idx, batch_size, False)

    best_acc, best_state, best_epoch, history = 0.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        run_loss = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item(); total += len(yb)
        train_acc = correct / total
        ev = evaluate(model, te, device)  # logged only, never used for selection
        history.append({"epoch": ep, "train_acc": train_acc, "train_loss": run_loss / total,
                        "test_acc": ev["acc"], "test_f1": ev["macro_f1"]})
        if train_acc > best_acc:
            best_acc, best_epoch = train_acc, ep
            best_state = copy.deepcopy(model.state_dict())
        sched.step()
        log(f"  ep {ep:3d}/{epochs} train {train_acc:.3f} test {ev['acc']:.3f}")

    model.load_state_dict(best_state)
    final = evaluate(model, te, device)
    final.update(best_epoch=best_epoch, best_train_acc=best_acc, history=history,
                 n_train=len(train_idx), n_test=len(test_idx))
    return final
