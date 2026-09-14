"""Nucleotide Transformer downstream benchmark (Dalla-Torre et al. 2023).

Loaded from InstaDeepAI's `nucleotide_transformer_downstream_tasks_revised` repo, which stores
one parquet pair (train/test) per task and therefore needs no dataset-script execution. The
"revised" repo is the one InstaDeep points at after the original script-based repo stopped
loading under datasets>=3.

CLAUDE.md §6.2 asks for a 4-6 task subset chosen for position sensitivity, so the default set is
the promoter and splice-site tasks plus two histone marks; --nt-tasks overrides it, and
--nt-tasks all runs all 18.

NT reports MCC for most tasks and F1 for the promoter/splice-site ones; probe.py and
finetune.py report acc, mcc and f1 for every task, so either convention can be quoted.
Sequence lengths are 200-600bp (task-native), so like Genomic Benchmarks this is not
context-length swept.
"""
from benchmarks.harness import SkipBenchmark, SKIP_MISSING
from benchmarks.downstream import classify_task
from benchmarks.probe import subsample

REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"

ALL_TASKS = [
    "promoter_all", "promoter_tata", "promoter_no_tata",
    "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
    "enhancers", "enhancers_types",
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
    "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
]

# position-sensitive subset (promoters + splice sites are the ones a recall/positional deficit
# should show up in first), plus two chromatin marks for contrast
DEFAULT_TASKS = ["promoter_all", "promoter_no_tata", "splice_sites_all",
                 "splice_sites_donors", "enhancers", "H3K4me3"]


def _load(task):
    from datasets import load_dataset
    return load_dataset(
        "parquet",
        data_files={"train": f"hf://datasets/{REPO}/{task}/train.parquet",
                    "test": f"hf://datasets/{REPO}/{task}/test.parquet"},
    )


def _columns(ds):
    cols = ds["train"].column_names
    seq_col = next((c for c in ("sequence", "seq", "dna") if c in cols), None)
    label_col = next((c for c in ("label", "labels", "target") if c in cols), None)
    if seq_col is None or label_col is None:
        raise SkipBenchmark(f"unexpected columns {cols}")
    return seq_col, label_col


def run(bctx) -> list[dict]:
    requested = bctx.args.nt_tasks
    tasks = ALL_TASKS if requested == ["all"] else (requested or DEFAULT_TASKS)

    results = []
    for task in tasks:
        cached = bctx.task_cached(task)
        if cached is not None:
            print(f"  -> [{task}] reused: {cached.status}  {cached.metrics}", flush=True)
            entry = dict(cached.metrics, _task=task, _ctx=cached.ctx,
                        _status=cached.status, _note=cached.note)
            results.append(entry)
            continue
        try:
            ds = _load(task)
            seq_col, label_col = _columns(ds)
        except Exception as exc:
            entry = {"_task": task, "_status": SKIP_MISSING,
                     "_note": f"load failed: {type(exc).__name__}: {str(exc)[:90]}"}
            results.append(entry)
            bctx.checkpoint_task(task, entry)
            continue

        train_seqs, train_y = subsample(ds["train"][seq_col], ds["train"][label_col],
                                        bctx.args.probe_train_limit, bctx.args.seed)
        test_seqs, test_y = subsample(ds["test"][seq_col], ds["test"][label_col],
                                      bctx.args.probe_test_limit, bctx.args.seed)
        ctx = min(max(len(s) for s in train_seqs), bctx.args.probe_max_ctx)
        metrics = classify_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx,
                                label=task)
        metrics["_task"] = task
        metrics["_ctx"] = ctx
        results.append(metrics)
        bctx.checkpoint_task(task, metrics)
    return results
