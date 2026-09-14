"""Genomic Benchmarks (Grešová et al. 2023) — the 8 tasks HyenaDNA reports in its Table 4.1.

Loaded from the authors' own HuggingFace mirror, unmodified, using each task's published
train/test split — no re-splitting, so rows line up with HyenaDNA's table (with the standing
caveat that these are frozen-feature probe numbers at a much smaller pretraining budget; see
benchmarks/probe.py).

Sequence lengths here are 200-4776bp and are a property of the task, so this benchmark is not
context-length swept: each task runs at its own native length (capped by --probe-max-ctx).
Padding a 200bp promoter out to 1M tokens would measure padding, not genomics.
"""
from benchmarks.harness import SkipBenchmark, SKIP_MISSING
from benchmarks.downstream import classify_task
from benchmarks.probe import subsample

REPO = "katarinagresova"

# (short name, HF dataset suffix, approximate max sequence length in bp)
TASKS = [
    ("mouse_enhancers", "Genomic_Benchmarks_dummy_mouse_enhancers_ensembl", 4776),
    ("coding_vs_intergenomic", "Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs", 200),
    ("human_vs_worm", "Genomic_Benchmarks_demo_human_or_worm", 200),
    ("human_enhancers_cohn", "Genomic_Benchmarks_human_enhancers_cohn", 500),
    ("human_enhancers_ensembl", "Genomic_Benchmarks_human_enhancers_ensembl", 573),
    ("human_regulatory", "Genomic_Benchmarks_human_ensembl_regulatory", 802),
    ("human_nontata_promoters", "Genomic_Benchmarks_human_nontata_promoters", 251),
    ("human_ocr_ensembl", "Genomic_Benchmarks_human_ocr_ensembl", 593),
]


def _load(name):
    from datasets import load_dataset
    return load_dataset(f"{REPO}/{name}")


def run(bctx) -> list[dict]:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SkipBenchmark("`datasets` not installed") from exc

    wanted = bctx.args.gb_tasks
    tasks = [t for t in TASKS if t[0] in wanted] if wanted else TASKS
    if wanted and not tasks:
        raise SkipBenchmark(f"--gb-tasks {wanted} matched none of {[t[0] for t in TASKS]}")

    results = []
    for short, hf_name, native_len in tasks:
        cached = bctx.task_cached(short)
        if cached is not None:
            print(f"  -> [{short}] reused: {cached.status}  {cached.metrics}", flush=True)
            entry = dict(cached.metrics, _task=short, _ctx=cached.ctx,
                        _status=cached.status, _note=cached.note)
            results.append(entry)
            continue
        try:
            ds = _load(hf_name)
        except Exception as exc:  # network / gated / renamed dataset
            entry = {"_task": short, "_status": SKIP_MISSING,
                     "_note": f"load failed: {type(exc).__name__}: {str(exc)[:90]}"}
            results.append(entry)
            bctx.checkpoint_task(short, entry)
            continue

        ctx = min(native_len, bctx.args.probe_max_ctx)
        train_seqs, train_y = subsample(ds["train"]["seq"], ds["train"]["label"],
                                        bctx.args.probe_train_limit, bctx.args.seed)
        test_seqs, test_y = subsample(ds["test"]["seq"], ds["test"]["label"],
                                      bctx.args.probe_test_limit, bctx.args.seed)
        metrics = classify_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx,
                                label=short)
        metrics["_task"] = short
        metrics["_ctx"] = ctx
        results.append(metrics)
        bctx.checkpoint_task(short, metrics)
    return results
