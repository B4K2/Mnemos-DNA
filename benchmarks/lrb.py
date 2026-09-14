"""Genomics Long-Range Benchmark (Kao et al. 2024, InstaDeepAI/genomics-long-range-benchmark).

This is the only benchmark family in the suite whose sequence length is a free parameter rather
than a property of the data: every task is a set of hg38 coordinates, and the input is a window
of the requested length centred on that coordinate. That makes it the right place to ask
"does more context actually help this architecture on real biology", which the fixed-length
Genomic Benchmarks / NT / GUE tasks cannot answer.

Tasks implemented (the four capability rows CLAUDE.md §6.2 is missing from the fixed-length
suites):
  vep_eqtl        variant effect, causal eQTL          (variant_effect_causal_eqtl/All_Tissues.csv)
  vep_clinvar     variant effect, pathogenic non-coding (variant_effect_pathogenic/...)
  chromatin       20 histone marks + 20 DNase tracks, multi-label   (chromatin_features/...)
  enhancer        regulatory element: enhancer                       (regulatory_elements/...)
  promoter        regulatory element: promoter                       (regulatory_elements/...)

The upstream repo ships these as CSV coordinate tables plus a dataset script; the script no
longer runs under datasets>=3, so the CSVs are downloaded directly with huggingface_hub and the
windows are cut from the local hg38 FASTA — the same FASTA used for pretraining, so the
train/test coordinates and the pretraining data are on identical assembly coordinates.

Variant tasks embed the reference and the alternate window separately and probe on
[ref, alt-ref]; this is the standard frozen-embedding VEP readout.
"""
import ast

import numpy as np

from benchmarks.fasta import Fasta, n_fraction
from benchmarks.harness import SkipBenchmark, SKIP_MISSING
from benchmarks.model_io import embed_sequences
from benchmarks.probe import linear_probe

REPO = "InstaDeepAI/genomics-long-range-benchmark"
MAX_N_FRACTION = 0.5

FILES = {
    "vep_eqtl": "variant_effect_causal_eqtl/All_Tissues.csv",
    "vep_clinvar": "variant_effect_pathogenic/vep_pathogenic_non_coding_subset.csv",
    "chromatin": "chromatin_features/histones_and_dnase_subset.csv",
    "enhancer": "regulatory_elements/enhancer_dataset_subset.csv",
    "promoter": "regulatory_elements/promoter_dataset_subset.csv",
}
DEFAULT_TASKS = list(FILES)


def _download(task):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, FILES[task], repo_type="dataset")


def _read(task):
    import pandas as pd
    return pd.read_csv(_download(task))


def _split_frames(df, train_limit, test_limit, seed):
    """LRB tables carry their own `split` column (a chromosome holdout) — never re-split."""
    if "split" not in df.columns:
        raise SkipBenchmark(f"no split column; got {list(df.columns)[:8]}")
    rng = np.random.default_rng(seed)
    out = {}
    for split, limit in (("train", train_limit), ("test", test_limit)):
        sub = df[df["split"] == split]
        if len(sub) == 0:
            raise SkipBenchmark(f"empty {split} split")
        if limit and len(sub) > limit:
            sub = sub.iloc[rng.choice(len(sub), size=limit, replace=False)]
        out[split] = sub.reset_index(drop=True)
    return out["train"], out["test"]


def _windows(fasta, chroms, centres, ctx):
    """Returns (sequences, keep_mask) — rows whose window is missing or mostly N are dropped."""
    seqs, keep = [], []
    for chrom, centre in zip(chroms, centres):
        seq = fasta.window(str(chrom), int(centre), ctx)
        ok = bool(seq) and n_fraction(seq) <= MAX_N_FRACTION
        keep.append(ok)
        if ok:
            seqs.append(seq)
    return seqs, np.array(keep)


def _substitute(seq, alt):
    """Replace the centre base (the variant position, by construction of the window)."""
    mid = len(seq) // 2
    return seq[:mid] + alt.upper()[0] + seq[mid + 1:]


def _embed(bctx, seqs, ctx):
    return embed_sequences(bctx.model, seqs, ctx=ctx, device=bctx.device,
                           batch_size=bctx.args.lrb_batch_size)


def _variant_features(bctx, fasta, df, ctx):
    seqs, keep = _windows(fasta, df["CHROM"], df["POS"], ctx)
    alts = df["ALT"][keep].tolist()
    alt_seqs = [_substitute(s, a) for s, a in zip(seqs, alts)]
    ref_emb = _embed(bctx, seqs, ctx)
    alt_emb = _embed(bctx, alt_seqs, ctx)
    return np.concatenate([ref_emb, alt_emb - ref_emb], axis=1), keep


def _run_vep(bctx, fasta, task, ctx):
    df = _read(task)
    label_col = "label" if "label" in df.columns else "INT_LABEL"
    if label_col not in df.columns:
        raise SkipBenchmark(f"no label column; got {list(df.columns)[:8]}")
    train_df, test_df = _split_frames(df, bctx.args.lrb_train_limit,
                                      bctx.args.lrb_test_limit, bctx.args.seed)
    train_x, k_tr = _variant_features(bctx, fasta, train_df, ctx)
    test_x, k_te = _variant_features(bctx, fasta, test_df, ctx)
    train_y = train_df[label_col].to_numpy()[k_tr].astype(int)
    test_y = test_df[label_col].to_numpy()[k_te].astype(int)
    metrics = linear_probe(train_x, train_y, test_x, test_y)
    metrics.update(n_train=len(train_y), n_test=len(test_y))
    return metrics


def _run_regulatory(bctx, fasta, task, ctx):
    df = _read(task)
    centres = ((df["START"] + df["STOP"]) // 2)
    df = df.assign(_centre=centres)
    train_df, test_df = _split_frames(df, bctx.args.lrb_train_limit,
                                      bctx.args.lrb_test_limit, bctx.args.seed)
    tr_seqs, k_tr = _windows(fasta, train_df["CHROM"], train_df["_centre"], ctx)
    te_seqs, k_te = _windows(fasta, test_df["CHROM"], test_df["_centre"], ctx)
    metrics = linear_probe(_embed(bctx, tr_seqs, ctx), train_df["label"].to_numpy()[k_tr],
                           _embed(bctx, te_seqs, ctx), test_df["label"].to_numpy()[k_te])
    metrics.update(n_train=int(k_tr.sum()), n_test=int(k_te.sum()))
    return metrics


def _run_chromatin(bctx, fasta, ctx):
    """Multi-label: 20 histone marks + 20 DNase tracks. Reported as the mean over tracks of a
    per-track one-vs-rest probe, skipping tracks that are single-class in this subsample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df = _read("chromatin")
    train_df, test_df = _split_frames(df, bctx.args.lrb_train_limit,
                                      bctx.args.lrb_test_limit, bctx.args.seed)

    def targets(frame):
        rows = [ast.literal_eval(h) + ast.literal_eval(d)
                for h, d in zip(frame["HISTONES"], frame["DNASE"])]
        return np.array(rows, dtype=int)

    tr_seqs, k_tr = _windows(fasta, train_df["CHROM"], train_df["POS"], ctx)
    te_seqs, k_te = _windows(fasta, test_df["CHROM"], test_df["POS"], ctx)
    train_y, test_y = targets(train_df)[k_tr], targets(test_df)[k_te]

    train_raw = _embed(bctx, tr_seqs, ctx)
    scaler = StandardScaler().fit(train_raw)
    train_x = scaler.transform(train_raw)
    test_x = scaler.transform(_embed(bctx, te_seqs, ctx))

    aurocs, accs = [], []
    for col in range(train_y.shape[1]):
        yt, ye = train_y[:, col], test_y[:, col]
        if len(np.unique(yt)) < 2 or len(np.unique(ye)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(train_x, yt)
        aurocs.append(roc_auc_score(ye, clf.decision_function(test_x)))
        accs.append((clf.predict(test_x) == ye).mean())
    if not aurocs:
        raise SkipBenchmark("every track was single-class at this subsample size")
    return {"mean_auroc": round(float(np.mean(aurocs)), 4),
            "mean_acc": round(float(np.mean(accs)), 4),
            "n_tracks_scored": len(aurocs),
            "n_train": int(k_tr.sum()), "n_test": int(k_te.sum())}


def run(bctx) -> list[dict]:
    ctx = bctx.ctx
    if ctx > bctx.args.lrb_max_ctx:
        raise SkipBenchmark(
            f"ctx {ctx} > --lrb-max-ctx {bctx.args.lrb_max_ctx}: embedding "
            f"{bctx.args.lrb_train_limit + bctx.args.lrb_test_limit} windows of this length "
            "would dominate the whole suite's runtime; raise the flag to include it")

    fasta = Fasta(bctx.args.hg38_fasta)
    tasks = bctx.args.lrb_tasks or DEFAULT_TASKS
    results = []
    for task in tasks:
        try:
            if task.startswith("vep"):
                metrics = _run_vep(bctx, fasta, task, ctx)
            elif task == "chromatin":
                metrics = _run_chromatin(bctx, fasta, ctx)
            else:
                metrics = _run_regulatory(bctx, fasta, task, ctx)
        except SkipBenchmark as exc:
            results.append({"_task": task, "_status": SKIP_MISSING, "_note": str(exc)[:140]})
            continue
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise
            results.append({"_task": task, "_status": SKIP_MISSING,
                            "_note": f"{type(exc).__name__}: {str(exc)[:110]}"})
            continue
        metrics["_task"] = task
        results.append(metrics)
    fasta.close()
    return results
