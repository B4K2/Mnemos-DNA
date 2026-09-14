"""Aggregate multi-seed downstream runs into mean +/- std per task.

Replaces the single-seed bootstrap interval in Table 4 with the quantity a reviewer actually
wants: run-to-run reliability. The bootstrap in benchmarks/bootstrap.py resamples one fixed set
of test predictions, so it measures split-sampling noise only — it cannot see variance from the
fine-tune seed (init of the head, train/dev split, batch order), which on datasets this small is
usually the larger of the two. A gain smaller than the across-seed std is not a result.

    uv run python scripts/aggregate_seeds.py results/d06/ft_seed*.json \
        --label d06 --out results/d06/ft_seeds_summary.csv

Multiple --label groups can be aggregated in one call by repeating the invocation and passing
--append, or by globbing both arms and letting model_id separate them (the default).
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

# Reported per task. Bootstrap-derived keys are deliberately excluded: mixing a within-split
# interval into an across-seed one would double-count the same noise.
METRICS = ("acc", "mcc", "f1", "auroc")

# Only the fixed-length classification suites have a seed-varying fine-tune to aggregate;
# lm/needle/scaling rows are deterministic given the seed and would just add noise-free rows.
DOWNSTREAM = ("genomic_benchmarks", "nt", "gue", "lrb", "species")


def _rows(path):
    """Yield (model_id, seed, benchmark, task, metrics) from one benchmark JSON.
    `model_id`/`seed` live in `meta`, the scores in each row's nested `metrics` dict."""
    with open(path) as fh:
        doc = json.load(fh)
    meta = doc.get("meta", {})
    for r in doc.get("rows", []):
        if r.get("status") != "ok":              # SKIP/OOM rows carry no metrics
            continue
        yield (meta.get("model_id", "?"), meta.get("seed", path),
               r.get("benchmark", "?"), r.get("task", "?"), r.get("metrics", {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="per-seed benchmark JSONs")
    ap.add_argument("--label", default=None, help="override the model id in the output")
    ap.add_argument("--min-seeds", type=int, default=2,
                    help="tasks with fewer seeds than this are reported but flagged")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()

    # (model, benchmark, task) -> metric -> [values across seeds]
    acc = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for path in args.files:
        if not os.path.exists(path):
            print(f"  missing, skipped: {path}")
            continue
        for model_id, seed, bench, task, metrics in _rows(path):
            if bench not in DOWNSTREAM:
                continue
            key = (args.label or model_id, bench, task)
            seeds[key].add(seed)
            for m in METRICS:
                if metrics.get(m) is not None:
                    acc[key][m].append(float(metrics[m]))

    out_rows = []
    for key in sorted(acc):
        model, bench, task = key
        row = {"model_id": model, "benchmark": bench, "task": task,
               "n_seeds": len(seeds[key])}
        for m, vals in acc[key].items():
            v = np.asarray(vals, dtype=float)
            row[f"{m}_mean"] = round(float(v.mean()), 4)
            # Sample std (ddof=1): the seeds are a sample of the run-to-run distribution, not
            # the population. Undefined for a single seed, reported as nan rather than 0 so a
            # one-seed row can never be mistaken for a perfectly reproducible one.
            row[f"{m}_std"] = round(float(v.std(ddof=1)), 4) if v.size > 1 else float("nan")
            row[f"{m}_min"] = round(float(v.min()), 4)
            row[f"{m}_max"] = round(float(v.max()), 4)
        row["underpowered"] = len(seeds[key]) < args.min_seeds
        out_rows.append(row)

    if not out_rows:
        print("no scored rows found")
        return 1

    width = max(len(f"{r['benchmark']}/{r['task']}") for r in out_rows)
    print(f"{'model':<8} {'task':<{width}} {'seeds':>5}  acc mean +/- std      range")
    for r in out_rows:
        std = r.get("acc_std", float("nan"))
        flag = "  (< min-seeds)" if r["underpowered"] else ""
        print(f"{r['model_id']:<8} {r['benchmark']}/{r['task']:<{width - len(r['benchmark']) - 1}}"
              f" {r['n_seeds']:>5}  {r.get('acc_mean', float('nan')):.4f} +/- "
              f"{std:.4f}   [{r.get('acc_min', float('nan')):.4f}, "
              f"{r.get('acc_max', float('nan')):.4f}]{flag}")

    # Per-arm macro average, the number Table 4's headline gain is quoted from. The std of the
    # per-task means understates the arm-level uncertainty; the mean of the per-task stds is
    # printed alongside so the comparison in the text can be made against the right one.
    print()
    for model in sorted({r["model_id"] for r in out_rows}):
        sel = [r for r in out_rows if r["model_id"] == model and "acc_mean" in r]
        means = np.asarray([r["acc_mean"] for r in sel])
        stds = np.asarray([r["acc_std"] for r in sel if r["acc_std"] == r["acc_std"]])
        print(f"{model}: macro-avg acc {means.mean():.4f} over {len(sel)} tasks; "
              f"mean per-task across-seed std {stds.mean() if stds.size else float('nan'):.4f}")

    if args.out:
        cols = sorted({c for r in out_rows for c in r},
                      key=lambda c: (c not in ("model_id", "benchmark", "task", "n_seeds"), c))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
