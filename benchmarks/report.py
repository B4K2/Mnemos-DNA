"""Result table rendering + serialization.

Every measurement point appears in the table exactly once, including the ones that did not
run — a skipped row shows its status and the reason, so an OOM at 256K is visible as a result
rather than as a hole in the table.
"""
import csv
import json
import time
from pathlib import Path

from benchmarks.harness import OK, OK_OFFLOAD, SKIP_OOM, Row

HEADERS = ["benchmark", "task", "ctx", "status", "metrics", "time_s", "peak_GB", "note"]


def _fmt_ctx(ctx):
    if ctx is None:
        return "native"
    if ctx >= 1024 and ctx % 1024 == 0:
        return f"{ctx // 1024}K"
    return str(ctx)


def _fmt_metrics(metrics: dict) -> str:
    parts = []
    for key, value in metrics.items():
        if value is None:
            parts.append(f"{key}=n/a")
        elif isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _row_cells(row: Row) -> list[str]:
    return [row.benchmark, row.task, _fmt_ctx(row.ctx), row.status,
            _fmt_metrics(row.metrics), f"{row.seconds:.1f}",
            "-" if row.peak_mem_gb != row.peak_mem_gb else f"{row.peak_mem_gb:.2f}",
            row.note]


def render_table(rows: list[Row], max_metric_width: int = 78) -> str:
    cells = [HEADERS] + [_row_cells(r) for r in rows]
    metric_col = HEADERS.index("metrics")
    for line in cells[1:]:
        if len(line[metric_col]) > max_metric_width:
            line[metric_col] = line[metric_col][: max_metric_width - 1] + "…"
    widths = [max(len(line[i]) for line in cells) for i in range(len(HEADERS))]

    def fmt(line):
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(line)).rstrip()

    out = [fmt(cells[0]), "  ".join("-" * w for w in widths)]
    out.extend(fmt(line) for line in cells[1:])
    return "\n".join(out)


def render_extrapolation(rows: list[Row], trained_ctx: int) -> str:
    """Derived from the `lm` rows: how perplexity moves as context goes past the length the
    checkpoint was actually trained at (CLAUDE.md §5's length-generalization test)."""
    lm_rows = [r for r in rows if r.benchmark == "lm"
               and r.status in (OK, OK_OFFLOAD) and "ppl" in r.metrics]
    if not lm_rows:
        return ""
    baseline = next((r for r in lm_rows if r.ctx == trained_ctx), None) or min(
        lm_rows, key=lambda r: abs((r.ctx or 0) - trained_ctx))
    base_ppl = baseline.metrics["ppl"]
    lines = [f"Extrapolation (trained context = {_fmt_ctx(trained_ctx)}, "
             f"reference row ctx = {_fmt_ctx(baseline.ctx)}, ppl = {base_ppl:g})",
             f"  {'ctx':>8}  {'ppl':>10}  {'ppl / ref':>10}  {'bits/nt':>8}"]
    for row in sorted(lm_rows, key=lambda r: r.ctx or 0):
        lines.append(f"  {_fmt_ctx(row.ctx):>8}  {row.metrics['ppl']:>10g}  "
                     f"{row.metrics['ppl'] / base_ppl:>10.3f}  {row.metrics['bpn']:>8g}")
    return "\n".join(lines)


def render_summary(rows: list[Row]) -> str:
    total = len(rows)
    ok = sum(1 for r in rows if r.status in (OK, OK_OFFLOAD))
    skipped = {}
    for row in rows:
        if row.status not in (OK, OK_OFFLOAD):
            skipped[row.status] = skipped.get(row.status, 0) + 1
    parts = [f"{ok}/{total} measurement points completed"]
    offloaded = sum(1 for r in rows if r.status == OK_OFFLOAD)
    if offloaded:
        parts.append(f"{offloaded} x ok: lowmem (not VRAM-comparable to native rows)")
    for status, count in sorted(skipped.items()):
        parts.append(f"{count} x {status}")
    oom_ctx = sorted({r.ctx for r in rows if r.status == SKIP_OOM and r.ctx})
    if oom_ctx:
        parts.append(f"first OOM at ctx {_fmt_ctx(min(oom_ctx))}")
    return " | ".join(parts)


def _row_key(row: Row):
    return (row.benchmark, row.task, row.ctx)


def merge_rows(old: list[Row], new: list[Row]) -> list[Row]:
    """Old rows a run did not re-measure are kept; a point measured this run replaces its
    previous value. Order follows the new run, with anything only present in the old file
    appended after it — so an interrupted rerun can never shrink a finished result file."""
    fresh = {_row_key(r) for r in new}
    return list(new) + [r for r in old if _row_key(r) not in fresh]


def _comparable(a: dict, b: dict) -> bool:
    """Two result files describe the same measurement iff they came from the same weights,
    device and downstream protocol. If they do not, merging them would silently mix e.g.
    probe and fine-tune accuracies for the same task."""
    keys = ("model_id", "checkpoint", "checkpoint_step", "device", "downstream_mode")
    return all(a.get(k) == b.get(k) for k in keys)


def write_outputs(rows: list[Row], meta: dict, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    if json_path.exists():
        try:
            prev = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            prev = None
        if prev and _comparable(prev.get("meta", {}), meta):
            prev_rows = [Row(**r) for r in prev.get("rows", [])]
            before = len(rows)
            rows = merge_rows(prev_rows, rows)
            if len(rows) > before:
                print(f"merged {len(rows) - before} row(s) from the previous {json_path.name} "
                      "(points this run did not re-measure)")
        elif prev:
            # different checkpoint / device / downstream mode: keep the old file rather than
            # overwrite or merge it.
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = out_dir / f"{stem}.{stamp}.json"
            json_path.rename(backup)
            print(f"previous {json_path.name} describes a different run "
                  f"(checkpoint/device/mode changed) — kept as {backup.name}")

    json_path.write_text(json.dumps(
        {"meta": meta, "rows": [r.to_dict() for r in rows]}, indent=2, default=str))

    metric_keys = sorted({k for r in rows for k in r.metrics})
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "task", "ctx", "status", "note", "seconds",
                         "peak_mem_gb"] + metric_keys)
        for row in rows:
            writer.writerow([row.benchmark, row.task, row.ctx, row.status, row.note,
                             round(row.seconds, 2), round(row.peak_mem_gb, 4)]
                            + [row.metrics.get(k, "") for k in metric_keys])
    return json_path, csv_path


# --- resume support -------------------------------------------------------------------
# Every completed measurement point is appended to <stem>.partial.jsonl the moment it
# finishes, so a run killed (Ctrl-C, OOM-killer, power) can be relaunched and pick up where
# it stopped instead of re-measuring hours of length sweeps.
#
# The file is KEPT after a run completes. That is deliberate: a finished stage must still be
# free to relaunch (eval_all.sh reruns stage 1 whenever stage 2 was interrupted), and without
# a surviving ledger that relaunch re-measures everything and then overwrites the good result
# file with a shorter one. `--no-resume` is the way to force a clean re-measurement.
#
# The first line is a `_meta` header fingerprinting the run (weights, device, downstream
# mode). Rows are only reused when the fingerprint still matches, so a ledger from a
# different checkpoint is archived rather than silently replayed.

PARTIAL_META_KEYS = ("model_id", "checkpoint", "checkpoint_step", "device", "dtype",
                     "downstream_mode")


def partial_path(out_dir: Path, stem: str) -> Path:
    return Path(out_dir) / f"{stem}.partial.jsonl"


def partial_fingerprint(meta: dict) -> dict:
    return {k: meta.get(k) for k in PARTIAL_META_KEYS}


def read_partial_meta(path: Path) -> dict | None:
    """The `_meta` header, or None for a missing file / a ledger written before headers
    existed (those are treated as matching, since they predate the guard)."""
    path = Path(path)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj.get("_meta") if isinstance(obj, dict) else None
    return None


def init_partial(path: Path, meta: dict) -> None:
    """Write the fingerprint header if the ledger does not exist yet."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_meta": partial_fingerprint(meta)}) + "\n")


def archive_partial(path: Path) -> Path:
    """Move a ledger aside (stale fingerprint, or --no-resume) instead of deleting it."""
    path = Path(path)
    dest = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}")
    path.rename(dest)
    return dest


def load_partial(path: Path) -> list[Row]:
    """Rows from a previous run. The `_meta` header is skipped, and a truncated final line
    (killed mid-write) is ignored rather than treated as corruption."""
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "_meta" in obj:
            continue
        try:
            rows.append(Row(**obj))
        except TypeError:
            continue
    return rows


def append_partial(path: Path, rows: list[Row]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), default=str) + "\n")
        f.flush()
