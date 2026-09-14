#!/usr/bin/env python3
"""
Scans results/<arm>/benchmark_cuda*.json (produced by scripts/benchmark.py) and writes
template/src/results_data.json — a normalized, per-arm view of the raw benchmark rows that
template/dna_model_comparison.jsx imports directly. Run this whenever a new arm finishes
benchmarking; the dashboard picks up the new arm automatically, no jsx edits required.

Usage: uv run python scripts/gen_dashboard_data.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Superseded-checkpoint result directories (results/d03_old, results/d03_old2,
# results/d05_old, results/d06_old, ...) are old runs kept on disk for reference —
# each has its own benchmark_cuda.json, so without this filter they show up as
# spurious extra "arms" on the dashboard alongside the current d03/d05/d06.
OLD_ARM_DIR_RE = re.compile(r"_old\d*$")
RESULTS_DIR = ROOT / "results"
OUT_PATH = ROOT / "template" / "src" / "results_data.json"


def is_superseded(path: Path) -> bool:
    """benchmark_cuda_old*.json holds a superseded run kept for comparison — it is surfaced
    as its own `<arm>_old...` entry rather than competing to be the arm's current result."""
    return path.stem.startswith("benchmark_cuda_old")


# Superseded runs that are still worth showing as their own arm on the dashboard, e.g.
# an earlier checkpoint of the same run that scored better on some tasks. Explicit,
# because most benchmark_cuda_old*.json files are just stale sweeps nobody wants plotted.
SHOW_SUPERSEDED = {
    # d07's step-24490 sweep: it beats the current step-33900 checkpoint on a handful of
    # downstream tasks, so it is surfaced beside it rather than silently replaced.
    "d07": ["benchmark_cuda_old_step24490.json"],
}


def superseded_files(arm_dir: Path) -> list[Path]:
    """The superseded runs explicitly opted into SHOW_SUPERSEDED, in name order."""
    names = SHOW_SUPERSEDED.get(arm_dir.name, [])
    return [arm_dir / n for n in names if (arm_dir / n).exists()]


def _coverage(path: Path) -> tuple:
    """(distinct benchmarks, checkpoint_step, is_full, mtime) — how complete a run is.

    Ranking by coverage first, not mtime, because a *partial* rerun into the arm's
    primary file (e.g. results/d06/benchmark_cuda.json was overwritten 2026-09-04 by a
    `--benchmarks scaling --no-checkpoint` run) would otherwise silently drop every
    other benchmark the arm had already measured.
    """
    try:
        data = json.loads(path.read_text())
    except Exception:
        return (-1, -1, 0, 0.0)
    meta = data.get("meta", {})
    n_bench = len({r["benchmark"] for r in data.get("rows", [])})
    step = meta.get("checkpoint_step")
    return (n_bench, -1 if step is None else step, 1 if "_full" in path.stem else 0, path.stat().st_mtime)


def pick_benchmark_file(arm_dir: Path) -> Path | None:
    """Pick the most complete benchmark_cuda*.json for an arm.

    benchmark_cuda_finetune.json (written by eval_all.sh's second pass) only ever
    contains genomic_benchmarks/nt/gue rows, never lm/needle/scaling/species/lrb — it
    must never be picked as the primary file. It is merged in separately by main().

    Superseded runs (benchmark_cuda_old*.json) compete too, but only on coverage: they
    win only when the current file measured strictly fewer benchmarks, which is exactly
    the partial-overwrite case above.
    """
    candidates = [c for c in sorted(arm_dir.glob("benchmark_cuda*.json")) if c.stem != "benchmark_cuda_finetune"]
    if not candidates:
        return None
    return max(candidates, key=_coverage)


def finetune_file(arm_dir: Path) -> Path | None:
    """The eval_all.sh finetune-mode pass, if it has been run for this arm."""
    p = arm_dir / "benchmark_cuda_finetune.json"
    return p if p.exists() else None


def panel_file(arm_dir: Path) -> Path | None:
    """The scripts/baselines_panel.sh run (species + GLRB, frozen probe), if present.

    The panel deliberately writes a separate stem from benchmark_cuda so it never
    merges into the main suite's rows, and its numbers are NOT interchangeable with
    them (different ctx points and resampling — e.g. d06 vep_eqtl@32K is 0.641 in the
    suite and 0.578 in the panel). So it is never merged here either: the external
    checkpoints (hyenadna_*, caduceus_*) have only a panel file and become arms from
    it, while an arm that has both also gets a separate "<arm>_panel" entry, which is
    the one that may be read side-by-side with the external arms.
    """
    p = arm_dir / "panel_cuda.json"
    return p if p.exists() else None


def ctx_label(n: int) -> str:
    if n >= 1_000_000:
        return "1M" if n == 1_048_576 else f"{n / 1_000_000:.2g}M"
    return f"{round(n / 1024)}K"


def normalize(meta: dict, rows: list) -> dict:
    by_bench = {}
    for r in rows:
        by_bench.setdefault(r["benchmark"], []).append(r)

    def ok(bench):
        return [r for r in by_bench.get(bench, []) if r["status"].startswith("ok")]

    # --- lm: one row per context length ---
    lm = [
        {"ctx": ctx_label(r["ctx"]), "n": r["ctx"], **r["metrics"]}
        for r in sorted(ok("lm"), key=lambda r: r["ctx"])
    ]

    # --- species: one row per context length ---
    species = [
        {"ctx": ctx_label(r["ctx"]), "n": r["ctx"], **r["metrics"]}
        for r in sorted(ok("species"), key=lambda r: r["ctx"])
    ]

    # --- scaling: one row per context length ---
    scaling = [
        {"ctx": ctx_label(r["ctx"]), "n": r["ctx"], **r["metrics"]}
        for r in sorted(ok("scaling"), key=lambda r: r["ctx"])
    ]

    # --- needle: grouped by ctx, each with its distance rows. Mode ("recall"/"copy")
    #     comes straight from the row's own metrics.mode, set by benchmarks/needle.py.
    needle_rows = ok("needle")
    needle_mode = None
    needle_by_ctx = {}
    for r in needle_rows:
        m = r["metrics"]
        needle_mode = needle_mode or m.get("mode")
        needle_by_ctx.setdefault(ctx_label(r["ctx"]), []).append(
            {
                "dist": f"{m['distance_bp']}bp" if m["distance_bp"] < 1000 else f"{round(m['distance_bp'] / 1024)}Kbp",
                "distance_bp": m["distance_bp"],
                **{k: v for k, v in m.items() if k not in ("distance_bp", "mode")},
            }
        )
    for ctx in needle_by_ctx:
        needle_by_ctx[ctx].sort(key=lambda r: r["distance_bp"])

    # --- genomic_benchmarks / nt / gue: keyed by task name, single point each ---
    def fixed_length_suite(bench):
        return {r["task"]: {"ctx": r["ctx"], **r["metrics"]} for r in ok(bench)}

    genomic_benchmarks = fixed_length_suite("genomic_benchmarks")
    nt = fixed_length_suite("nt")
    gue = fixed_length_suite("gue")

    # --- lrb: keyed by task, one row per context length ---
    lrb = {}
    for r in ok("lrb"):
        if r["task"] == "lrb":
            continue
        lrb.setdefault(r["task"], []).append({"ctx": ctx_label(r["ctx"]), "n": r["ctx"], **r["metrics"]})
    for task in lrb:
        lrb[task].sort(key=lambda r: r["n"])

    return {
        "meta": {
            "id": meta["model_id"],
            "runName": meta["run_name"],
            "checkpoint": meta["checkpoint"],
            "checkpointStep": meta["checkpoint_step"],
            "gpu": meta["gpu"],
            "dtype": meta["dtype"].replace("torch.", ""),
            "trainedContext": meta["trained_context_length"],
            "params": meta["total_params"],
            "nonEmbeddingParams": meta["non_embedding_params"],
            "vocabSize": meta["vocab_size"],
            "benchmarkWallHours": meta["wall_time_s"] / 3600,
            "benchmarksRun": meta["benchmarks"],
        },
        "lm": lm,
        "species": species,
        "scaling": scaling,
        "needle": {"mode": needle_mode, "byCtx": needle_by_ctx},
        "genomic_benchmarks": genomic_benchmarks,
        "nt": nt,
        "gue": gue,
        "lrb": lrb,
    }


# Training progress. The benchmark JSON records the checkpoint step but not the token
# budget, so the budget is read from the arm's training config: one step consumes
# batch_size * context_length * grad_accum_steps tokens (scripts/train.py:249).
# Most arms are planned 5B-token runs; d01 (the original 120M-param model) and a few
# others were budgeted differently, which is why the percentage is always reported
# against that arm's OWN training_tokens rather than a shared denominator.
ARM_CONFIG = {
    "d01": "d01_kda.yaml",
    "d01_new": "d01_new.yaml",
    "d01_rope": "d01_rope.yaml",
    "d01_ropefix": "d01_ropefix.yaml",
    "d02": "d02_kda_local.yaml",
    "d03": "d03_kata.yaml",
    "d04": "d04_hola.yaml",
    "d05_rope": "d05_rope.yaml",
    "d06": "d06_kmer_fastweight.yaml",
    "d07": "d07_kmer_local.yaml",
}


def training_progress(arm_id: str, step: int | None) -> dict | None:
    """{tokensDone, tokensPlanned, pct} for one arm, or None if we can't say."""
    name = ARM_CONFIG.get(arm_id)
    if name is None or step is None:
        return None
    cfg_path = ROOT / "configs" / name
    if not cfg_path.exists():
        return None
    from training.config import load_config
    cfg = load_config(str(cfg_path))
    per_step = cfg["batch_size"] * cfg["context_length"] * cfg["grad_accum_steps"]
    planned = cfg["training_tokens"]
    done = step * per_step
    return {
        "tokensDone": done,
        "tokensPlanned": planned,
        "tokensPerStep": per_step,
        "pct": round(100.0 * done / planned, 1) if planned else None,
    }


def main():
    if not RESULTS_DIR.exists():
        raise SystemExit(f"no results/ dir at {RESULTS_DIR}")

    out = {}
    for arm_dir in sorted(RESULTS_DIR.iterdir()):
        if not arm_dir.is_dir() or arm_dir.name.startswith("_"):
            continue
        if OLD_ARM_DIR_RE.search(arm_dir.name):
            continue
        arm_files = []
        bpath = pick_benchmark_file(arm_dir)
        if bpath is not None:
            arm_files.append((arm_dir.name, bpath))

        ppath = panel_file(arm_dir)
        if ppath is not None:
            arm_files.append((arm_dir.name if bpath is None else f"{arm_dir.name}_panel", ppath))

        
        # Superseded runs opted into SHOW_SUPERSEDED get their own "<arm>_ckpt<step>"
        # entry so the dashboard can show the old and new checkpoint side by side.
        for old in superseded_files(arm_dir):
            step = json.loads(old.read_text())["meta"].get("checkpoint_step")
            arm_files.append((f"{arm_dir.name}_ckpt{step}", old))

        # Key by directory name, not meta["model_id"] — several dirs (e.g. results/d06
        # and results/d06_ctx32k) share a model_id, and keying by it silently drops all
        # but the last one.
        ft = finetune_file(arm_dir)
        for arm_id, bpath in arm_files:
            # Merge finetune-mode rows (appended after base rows) only for the arm's
            # primary entry, so fixed_length_suite's task-keyed dict lets finetune
            # numbers overwrite probe-mode ones for genomic_benchmarks/nt/gue while
            # lm/needle/scaling/species/lrb keep coming from the base file untouched.
            extra_rows = json.loads(ft.read_text())["rows"] if (ft is not None and bpath == arm_files[0][1]) else None
            emit(out, arm_id, bpath, extra_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH} — {len(out)} arm(s): {', '.join(out)}")




def emit(out: dict, arm_id: str, bpath: Path, extra_rows: list | None = None) -> None:
    data = json.loads(bpath.read_text())
    rows = data["rows"] + extra_rows if extra_rows else data["rows"]
    out[arm_id] = normalize(data["meta"], rows)
    out[arm_id]["_source"] = str(bpath.relative_to(ROOT))
    # "panel" rows come from scripts/baselines_panel.sh and are only comparable with
    # each other; "suite" rows come from scripts/benchmark.py. External arms are the
    # released HyenaDNA/Caduceus checkpoints run through our harness (§9) — they carry
    # no checkpoint of ours and their pretraining budget is not matched to ours.
    out[arm_id]["meta"]["group"] = "panel" if bpath.stem.startswith("panel_") else "suite"
    out[arm_id]["meta"]["external"] = arm_id.split("_")[0] in ("hyenadna", "caduceus")
    prog = training_progress(arm_id.removesuffix("_panel"), data["meta"].get("checkpoint_step"))
    if prog is not None:
        out[arm_id]["meta"]["training"] = prog


if __name__ == "__main__":
    main()
