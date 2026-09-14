"""Bootstrap confidence intervals for every reported test-set number.

Why: the downstream splits used here are small (`--probe-test-limit` 2000, GUE/NT test splits
of a few hundred, `--lrb-test-limit` 128), so a 1-2 point difference between two arms is
routinely inside the sampling noise of the split itself. Every classification metric is
therefore reported with a percentile bootstrap over the *test predictions* — the predictions
are computed once and only the resampling is repeated, so this costs seconds of CPU and never
re-runs the model.

What it does NOT cover: variance from training (seed, data order, C selection). This is
split-sampling uncertainty only — an arm whose CI overlaps another's is not distinguishable
on this split; non-overlap is necessary, not sufficient, for a real architecture difference.

Convention for the emitted keys: `<metric>` stays the point estimate on the full test split
(unchanged, so old rows stay comparable), and each metric gains
`<metric>_mean` / `<metric>_lo` / `<metric>_hi` (2.5/97.5 percentiles of `n_boot` resamples).
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

DEFAULT_N_BOOT = 1000
ALPHA = 0.05

# Set once per suite from --bootstrap (scripts/benchmark.py). Callers that do not thread an
# explicit n_boot through (the shared linear probe, the fine-tune scorer) read this.
N_BOOT = DEFAULT_N_BOOT
SEED = 0


def configure(n_boot: int, seed: int = 0) -> None:
    global N_BOOT, SEED
    N_BOOT, SEED = int(n_boot), int(seed)


def _f1_of(y, pred):
    average = "binary" if len(np.unique(np.concatenate([y, pred]))) <= 2 else "macro"
    return float(f1_score(y, pred, average=average, zero_division=0))


def _auroc_of(y, score):
    try:
        if score.ndim == 1:
            return float(roc_auc_score(y, score))
        if len(np.unique(y)) == score.shape[1]:
            return float(roc_auc_score(y, score, multi_class="ovr"))
    except (ValueError, IndexError):
        pass
    return None


def _pack(name, values, out):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return
    lo, hi = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    out[f"{name}_mean"] = round(float(values.mean()), 4)
    out[f"{name}_lo"] = round(float(lo), 4)
    out[f"{name}_hi"] = round(float(hi), 4)


def bootstrap_classification(y_true, pred, score=None, n_boot: int | None = None,
                             seed: int | None = None) -> dict:
    """Percentile bootstrap over test predictions for acc / mcc / f1 (+ auroc if `score`,
    the decision function or probabilities, is given). Returns {} when disabled (n_boot<=0)
    or the split is degenerate."""
    n_boot = N_BOOT if n_boot is None else n_boot
    seed = SEED if seed is None else seed
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    n = len(y_true)
    if n_boot <= 0 or n < 2:
        return {}
    score = None if score is None else np.asarray(score)
    rng = np.random.default_rng(seed)
    acc, mcc, f1s, auc = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_true[idx], pred[idx]
        acc.append(accuracy_score(yb, pb))
        mcc.append(matthews_corrcoef(yb, pb))
        f1s.append(_f1_of(yb, pb))
        if score is not None:
            auc.append(_auroc_of(yb, score[idx]))
    out = {"n_boot": int(n_boot)}
    _pack("acc", acc, out)
    _pack("mcc", mcc, out)
    _pack("f1", f1s, out)
    if score is not None:
        _pack("auroc", auc, out)
    return out


def bootstrap_mean(values, name: str, n_boot: int | None = None, seed: int | None = None,
                   transform=None) -> dict:
    """Percentile bootstrap of a mean over per-item values (per-window cross-entropy, per-trial
    recall accuracy, ...). `transform` maps a resampled mean to the reported quantity, e.g.
    `math.exp` for CE -> perplexity."""
    n_boot = N_BOOT if n_boot is None else n_boot
    seed = SEED if seed is None else seed
    values = np.asarray(values, dtype=float)
    if n_boot <= 0 or values.size < 2:
        return {}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    if transform is not None:
        means = np.asarray([transform(m) for m in means], dtype=float)
    out = {"n_boot": int(n_boot)}
    _pack(name, means, out)
    return out
