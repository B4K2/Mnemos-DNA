"""Frozen-backbone linear probe, shared by every downstream classification benchmark.

Why a probe and not full fine-tuning: HyenaDNA/NT report full fine-tuning numbers, but full
fine-tuning of 7 arms x 8-18 tasks x several context lengths does not fit a single 8GB card in
any reasonable time. A logistic-regression head on frozen pooled features (mean/max/std +
positional bins over the last three layers, C chosen on a train hold-out — see
benchmarks/model_io.py) is the standard cheap stand-in, is identical across arms (so the architecture comparison stays controlled), and
is a strict lower bound on fine-tuned accuracy. Every table produced here is labelled `probe`
for exactly that reason — do not present these numbers next to published fine-tuned numbers
without saying so.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmarks.bootstrap import bootstrap_classification
from benchmarks.model_io import DEFAULT_BINS, embed_sequences


C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _fit(train_x, train_y, C, max_iter, seed):
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    return clf.fit(train_x, train_y)


def _select_C(train_x, train_y, max_iter, seed, grid=C_GRID, val_frac=0.2, min_train=60):
    """Pick C on a stratified 20% hold-out of the *training* split (never the test split).
    With ~25k features and a few thousand samples the regulariser is no longer a detail.
    Returns (C, dev_acc) — dev_acc is None when the split is too small to sweep."""
    n = len(train_y)
    if n < min_train or len(np.unique(train_y)) < 2:
        return 1.0, None
    try:
        tr, dev = train_test_split(np.arange(n), test_size=val_frac,
                                   random_state=seed, stratify=train_y)
    except ValueError:  # a class too rare to stratify
        tr, dev = train_test_split(np.arange(n), test_size=val_frac, random_state=seed)
    if len(np.unique(train_y[tr])) < 2:
        return 1.0, None
    best = None
    for C in grid:
        clf = _fit(train_x[tr], train_y[tr], C, max_iter, seed)
        acc = accuracy_score(train_y[dev], clf.predict(train_x[dev]))
        if best is None or acc > best[1] + 1e-9:  # ties -> smaller C (stronger regulariser)
            best = (C, acc)
    return best


def _f1(y, pred) -> float:
    """NT reports F1 (not MCC) for the promoter and splice-site tasks. Binary -> positive-class
    F1; multi-class -> macro F1, matching the NT paper's convention."""
    average = "binary" if len(np.unique(np.concatenate([y, pred]))) <= 2 else "macro"
    return round(float(f1_score(y, pred, average=average, zero_division=0)), 4)


def linear_probe(train_x, train_y, test_x, test_y, max_iter: int = 2000, seed: int = 0,
                 C=None) -> dict:
    """Standardise -> pick C on a train hold-out (unless `C` is given) -> refit on the full
    training split -> score on test."""
    train_y, test_y = np.asarray(train_y), np.asarray(test_y)
    scaler = StandardScaler().fit(train_x)
    train_z, test_z = scaler.transform(train_x), scaler.transform(test_x)

    dev_acc = None
    if C is None:
        C, dev_acc = _select_C(train_z, train_y, max_iter, seed)
    clf = _fit(train_z, train_y, C, max_iter, seed)
    logits = clf.decision_function(test_z)
    pred = clf.predict(test_z)

    metrics = {
        "acc": round(float(accuracy_score(test_y, pred)), 4),
        "mcc": round(float(matthews_corrcoef(test_y, pred)), 4),
        "f1": _f1(test_y, pred),
    }
    try:
        if logits.ndim == 1:
            metrics["auroc"] = round(float(roc_auc_score(test_y, logits)), 4)
        elif len(np.unique(test_y)) == logits.shape[1]:
            metrics["auroc"] = round(
                float(roc_auc_score(test_y, logits, multi_class="ovr")), 4)
    except ValueError:
        pass  # degenerate label set in this split; acc/mcc still stand
    # 1000x resample of the test predictions -> mean + 95% CI per metric (seconds of CPU,
    # the model is not re-run). See benchmarks/bootstrap.py for what this does and does not cover.
    metrics.update(bootstrap_classification(test_y, pred, score=logits))
    metrics["probe_C"] = C
    if dev_acc is not None:
        metrics["probe_dev_acc"] = round(float(dev_acc), 4)
    metrics["feat_dim"] = int(np.shape(train_x)[1])
    return metrics


def _parse_layers(spec, n_layers: int) -> list[int]:
    """`--probe-layers` -> ascending, de-duplicated, non-negative block indices.
    Accepts `all`, or a comma list where negatives count from the end (`-1` = last block)."""
    if spec in (None, "", "default"):
        return []
    if str(spec).strip() == "all":
        return list(range(n_layers))
    out = set()
    for tok in str(spec).split(","):
        li = int(tok)
        li = li + n_layers if li < 0 else li
        if not 0 <= li < n_layers:
            raise ValueError(f"--probe-layers: block {tok} out of range for {n_layers} blocks")
        out.add(li)
    return sorted(out)


def probe_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx: int,
               batch_size: int = 8) -> dict:
    """Embed both splits with the frozen model, then fit the probe.

    With `--probe-layers` set, the requested blocks are captured in ONE forward pass over each
    split and a separate probe is fitted per block off contiguous slices of that feature
    matrix. The row then carries `acc_layer{i}` (and mcc/f1) per block plus `best_layer`,
    which is what the mechanistic claim needs: a jump in linear separability at the block
    that does the work is only interpretable against the blocks either side of it, measured on
    identical splits and an identical embedding pass.
    """
    spec = getattr(bctx.args, "probe_layers", None)
    # Block count from the config, not `model.blocks` — the external wrappers in
    # models/external/ have no `.blocks` attribute.
    layers = _parse_layers(spec, int(bctx.config["num_layers"]))
    kwargs = dict(ctx=ctx, device=bctx.device, batch_size=batch_size)
    if layers:
        kwargs["layers"] = layers
    train_x = embed_sequences(bctx.model, train_seqs, **kwargs)
    test_x = embed_sequences(bctx.model, test_seqs, **kwargs)

    if not layers:
        metrics = linear_probe(train_x, train_y, test_x, test_y)
    else:
        width = train_x.shape[1] // len(layers)
        per_layer, best = {}, (-1.0, None)
        for j, li in enumerate(layers):
            sl = slice(j * width, (j + 1) * width)
            m = linear_probe(train_x[:, sl], train_y, test_x[:, sl], test_y)
            for key in ("acc", "mcc", "f1", "auroc"):
                if key in m:
                    # `_layer{i}`, not `_l{i}`: bootstrap_classification already emits
                    # `acc_lo`/`acc_hi`, and `acc_l0` next to `acc_lo` is one glyph apart.
                    per_layer[f"{key}_layer{li}"] = m[key]
            if m["acc"] > best[0]:
                best = (m["acc"], li, m)
        metrics = dict(best[2])          # headline metrics come from the best block
        metrics.update(per_layer)
        metrics["best_layer"] = best[1]
        metrics["probe_layers"] = ",".join(str(li) for li in layers)
    metrics["n_train"] = len(train_y)
    metrics["n_test"] = len(test_y)
    metrics["seq_len"] = ctx
    return metrics


def subsample(seqs, labels, limit: int, seed: int = 0):
    """Cap a split's size so an 8GB card can embed it in minutes rather than hours.
    Stratification is not attempted — a plain uniform draw keeps the label prior intact."""
    if limit is None or len(seqs) <= limit:
        return list(seqs), np.asarray(labels)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=limit, replace=False)
    return [seqs[i] for i in idx], np.asarray(labels)[idx]
