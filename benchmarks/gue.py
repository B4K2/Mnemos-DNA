"""GUE — Genome Understanding Evaluation (DNABERT-2, Zhou et al. 2023).

GUE has no non-gated HuggingFace mirror that can be trusted for a paper (the official
distribution is a zip linked from the DNABERT-2 README), so this benchmark reads a local
extracted copy and skips cleanly when it is absent:

    # download GUE.zip from https://github.com/MAGICS-LAB/DNABERT_2 (README -> GUE)
    unzip GUE.zip -d data/
    uv run python scripts/benchmark.py ... --gue-dir data/GUE

Expected layout (as shipped): data/GUE/<group>/<task>/{train,dev,test}.csv with a
`sequence,label` header. Every task directory found is run; --gue-tasks restricts it.
Sequence lengths are 70-1000bp and task-native, so this is not context-length swept.
"""
import csv
from pathlib import Path

from benchmarks.harness import SkipBenchmark, SKIP_MISSING
from benchmarks.downstream import classify_task
from benchmarks.probe import subsample


def _read_csv(path: Path):
    seqs, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "sequence" not in reader.fieldnames:
            raise SkipBenchmark(f"{path} is not a GUE sequence,label csv")
        for row in reader:
            seqs.append(row["sequence"])
            labels.append(int(row["label"]))
    return seqs, labels


def _find_tasks(root: Path):
    """A task dir is any directory containing train.csv and test.csv."""
    return sorted(p.parent for p in root.rglob("train.csv") if (p.parent / "test.csv").exists())


def run(bctx) -> list[dict]:
    gue_dir = bctx.args.gue_dir
    if not gue_dir:
        raise SkipBenchmark("no --gue-dir given (GUE has no auto-downloadable mirror; "
                            "see benchmarks/gue.py for the download command)")
    root = Path(gue_dir)
    if not root.exists():
        raise SkipBenchmark(f"--gue-dir {root} does not exist")

    task_dirs = _find_tasks(root)
    if not task_dirs:
        raise SkipBenchmark(f"no <task>/train.csv found under {root}")
    wanted = bctx.args.gue_tasks
    if wanted:
        task_dirs = [d for d in task_dirs if d.name in wanted]

    results = []
    for task_dir in task_dirs:
        name = f"{task_dir.parent.name}/{task_dir.name}"
        cached = bctx.task_cached(name)
        if cached is not None:
            print(f"  -> [{name}] reused: {cached.status}  {cached.metrics}", flush=True)
            entry = dict(cached.metrics, _task=name, _ctx=cached.ctx,
                        _status=cached.status, _note=cached.note)
            results.append(entry)
            continue
        try:
            train_seqs, train_labels = _read_csv(task_dir / "train.csv")
            test_seqs, test_labels = _read_csv(task_dir / "test.csv")
        except Exception as exc:
            entry = {"_task": name, "_status": SKIP_MISSING,
                     "_note": f"{type(exc).__name__}: {str(exc)[:90]}"}
            results.append(entry)
            bctx.checkpoint_task(name, entry)
            continue

        train_seqs, train_y = subsample(train_seqs, train_labels,
                                        bctx.args.probe_train_limit, bctx.args.seed)
        test_seqs, test_y = subsample(test_seqs, test_labels,
                                      bctx.args.probe_test_limit, bctx.args.seed)
        ctx = min(max(len(s) for s in train_seqs), bctx.args.probe_max_ctx)
        metrics = classify_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx,
                                label=name)
        metrics["_task"] = name
        metrics["_ctx"] = ctx
        results.append(metrics)
        bctx.checkpoint_task(name, metrics)
    return results
