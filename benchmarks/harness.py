"""Benchmark harness: runs each (benchmark, context-length) point in isolation, records
metrics/time/peak-memory, and turns failures into explicit SKIP rows instead of killing the run.

Nothing in here knows about any specific benchmark — benchmarks register themselves in
benchmarks/__init__.py and receive a BenchContext.
"""
import contextlib
import gc
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from models.shared import lowmem

OK = "ok"
OK_OFFLOAD = "ok: lowmem"   # succeeded only after the low-memory fallback kicked in
SKIP_OOM = "SKIP: OOM"
SKIP_CPU_CAP = "SKIP: cpu ctx cap"
SKIP_CTX_CAP = "SKIP: above benchmark ctx cap"
REUSED = "reused"        # loaded from a previous, interrupted run's .partial.jsonl

# Multi-task, non-length-swept benchmarks that checkpoint each sub-task themselves via
# BenchContext.checkpoint_task instead of only being recorded once their whole point returns —
# see run_suite. Their own append_partial calls are the only ones that touch the resume file;
# run_suite must not also append their point_rows or every already-checkpointed task would be
# duplicated in the .partial.jsonl.
SELF_CHECKPOINTING = {"genomic_benchmarks", "nt", "gue"}
SKIP_OOM_PRIOR = "SKIP: OOM at shorter ctx"
SKIP_MISSING = "SKIP: data unavailable"
ERROR = "ERROR"


class SkipBenchmark(Exception):
    """Raised by a benchmark when its data/prerequisites are missing. Becomes a SKIP row
    with the message as the note, rather than aborting the suite."""


# Benchmarks whose length sweep is capped by their own CLI flag rather than by --lengths.
# `needle` is capped because a 256K/512K/1M repeat-copy sweep costs hours per point and the
# recall question (§6.4) is already answered by <=128K, where the state capacity bound bites.
CTX_CAP_ARGS = {"needle": "needle_max_ctx"}


def parse_lengths(value: str) -> list[int]:
    """Accepts `8192,16384` or `8K,16K,1M`."""
    out = []
    for token in value.split(","):
        token = token.strip().upper()
        if not token:
            continue
        mult = {"K": 1024, "M": 1024 * 1024}.get(token[-1], 1)
        out.append(int(float(token[:-1] if mult > 1 else token)) * mult)
    return sorted(set(out))


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("can't allocate memory" in msg)


@dataclass
class Row:
    benchmark: str
    task: str
    ctx: int | None          # None => task runs at its own native sequence length
    status: str
    metrics: dict = field(default_factory=dict)
    note: str = ""
    seconds: float = 0.0
    peak_mem_gb: float = 0.0

    def to_dict(self):
        return asdict(self)


class _RSSSampler:
    """Peak resident-set sampler for CPU runs (CUDA has its own allocator counter).
    Linux-only; degrades to 0.0 elsewhere."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _rss_bytes() -> int:
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * 4096
        except OSError:
            return 0

    def __enter__(self):
        if self._rss_bytes() == 0:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._rss_bytes())

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._rss_bytes())
        return False


@dataclass
class BenchContext:
    """Everything a benchmark function needs. Passed to every registered benchmark."""
    model: torch.nn.Module
    config: dict
    device: str
    dtype: torch.dtype
    ctx: int | None                  # context length for this measurement point (None = native)
    args: object                     # parsed argparse namespace, for optional data paths
    trained_ctx: int = 0
    spec_name: str = ""
    resume_path: object = None
    prior_tasks: dict = field(default_factory=dict)   # task name -> Row, see SELF_CHECKPOINTING

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def task_cached(self, task: str):
        """A result for `task` persisted by a previous, interrupted run of this same
        multi-task benchmark (genomic_benchmarks/nt/gue), or None. Callers use this to skip
        finished tasks instead of retraining them."""
        return self.prior_tasks.get(task)

    def checkpoint_task(self, task: str, entry: dict) -> None:
        """Persist one finished sub-task immediately, so a multi-task benchmark survives a
        Ctrl-C partway through its task list instead of losing every task completed so far —
        genomic_benchmarks/nt/gue each run 6-18 independent fine-tunes per point and a single
        one can take well over an hour."""
        if self.resume_path is None:
            return
        from benchmarks.report import append_partial
        row = Row(self.spec_name, task, entry.get("_ctx", self.ctx),
                  entry.get("_status", OK),
                  {k: v for k, v in entry.items() if not k.startswith("_")},
                  entry.get("_note", ""), 0.0, 0.0)
        append_partial(self.resume_path, [row])


def _reset_mem(device: str):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def seed_point(seed: int, name: str, ctx) -> None:
    """Reseed numpy + torch deterministically per (benchmark, ctx).

    ShardLoader draws windows with the global numpy RNG, so without this every checkpoint
    (and every arm) is scored on *different* held-out windows and the lm/needle numbers
    carry sampling noise that looks like a change in the model. Keyed on the point rather
    than the run so the windows are the same no matter which subset of benchmarks/lengths
    was requested."""
    s = (int(seed) * 1_000_003 + zlib.crc32(f"{name}:{ctx}".encode())) % (2 ** 31)
    np.random.seed(s)
    torch.manual_seed(s)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class _Progress:
    """Elapsed / ETA line after every measurement point.

    Points are wildly different in cost (a 1M-token needle sweep vs a 200bp probe), so the
    ETA is a model, not a count: within a length-swept benchmark the next point is estimated
    at `t_last * ctx / ctx_last` (linear in ctx — exact for the O(N) arms, an underestimate
    for O(N^2)); a benchmark that has not started yet contributes nothing, and the ETA is
    printed as a lower bound (`>=`) while any such benchmark remains."""

    def __init__(self, specs, lengths):
        self.t0 = time.time()
        self.total = sum(len(lengths) if s.length_swept else 1 for s in specs)
        self.done = 0
        self.last = {}          # spec.name -> (ctx, seconds) of the last completed point
        self.pending = {s.name: (list(lengths) if s.length_swept else [None]) for s in specs}

    def update(self, spec, ctx, seconds, skipped: bool):
        self.done += 1
        self.pending[spec.name] = [c for c in self.pending[spec.name] if c != ctx]
        if not skipped:
            self.last[spec.name] = (ctx, seconds)

    def line(self) -> str:
        eta, unknown = 0.0, []
        for name, remaining in self.pending.items():
            if not remaining:
                continue
            if name not in self.last:
                unknown.append(name)
                continue
            ctx_last, t_last = self.last[name]
            for c in remaining:
                eta += t_last * (c / ctx_last) if (ctx_last and c) else t_last
        elapsed = time.time() - self.t0
        msg = (f"[progress] {self.done}/{self.total} points  elapsed {_fmt_hms(elapsed)}  "
               f"ETA {'>=' if unknown else '~'}{_fmt_hms(eta)}")
        if unknown:
            msg += f"  (+ not started: {', '.join(unknown)})"
        return msg


def _attempt(spec, ctx_obj: BenchContext, lowmem_ctx):
    """One execution attempt. Returns (status, note, result, seconds, peak, host_peak)."""
    _reset_mem(ctx_obj.device)
    seed_point(getattr(ctx_obj.args, "seed", 0), spec.name, ctx_obj.ctx)
    t0 = time.time()
    status, note, result = OK, "", {}
    with _RSSSampler() as rss:
        try:
            with lowmem_ctx:
                result = spec.fn(ctx_obj)
        except SkipBenchmark as exc:
            status, note = SKIP_MISSING, str(exc)
        except KeyboardInterrupt:
            raise            # a Ctrl-C is a stop request, not a failed measurement
        except BaseException as exc:  # noqa: BLE001 - deliberate catch-all, see module docstring
            if _is_oom(exc):
                status, note = SKIP_OOM, type(exc).__name__
            else:
                status = ERROR
                note = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:180]
                traceback.print_exc()
    seconds = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9) if ctx_obj.is_cuda else rss.peak / 1e9
    _reset_mem(ctx_obj.device)
    return status, note, result, seconds, peak, rss.peak / 1e9


def run_point(spec, ctx_obj: BenchContext) -> list[Row]:
    """Run one benchmark at one context length, capturing time, peak memory and failures.

    A benchmark returns either a metrics dict (one row) or a list of dicts (one row per
    sub-task, each optionally carrying `_task` / `_status` / `_note` keys) so multi-task
    suites can skip individual tasks without failing the whole point.

    When `--offload` is set, a point that OOMs natively is retried once in low-memory mode
    (chunked prefill + CPU activation offload; see models/shared/lowmem.py). The retry is
    reported as `ok: lowmem` rather than `ok`, because its peak-VRAM figure is the cost of a
    deliberately memory-limited execution and is NOT comparable to a native row — mixing the
    two would turn the §6.3 curve into two different measurements plotted as one line.
    """
    status, note, result, seconds, peak, host_peak = _attempt(
        spec, ctx_obj, contextlib.nullcontext())

    offload_enabled = getattr(ctx_obj.args, "offload", False)
    if status == SKIP_OOM and offload_enabled and ctx_obj.is_cuda:
        chunk = getattr(ctx_obj.args, "offload_chunk", lowmem.DEFAULT_CHUNK_SIZE)
        print(f"  .. native OOM; retrying in low-memory mode (chunk={chunk})", flush=True)
        r_status, r_note, r_result, r_seconds, r_peak, r_host = _attempt(
            spec, ctx_obj, lowmem.low_memory_mode(chunk=chunk, offload=True))
        if r_status == OK:
            status, result, seconds, peak, host_peak = (
                OK_OFFLOAD, r_result, r_seconds, r_peak, r_host)
            note = f"lowmem: chunked prefill (chunk={chunk}) + CPU activation offload"
        else:
            note = f"{note}; lowmem retry also failed ({r_note})"
            seconds += r_seconds

    if status not in (OK, OK_OFFLOAD):
        return [Row(spec.name, spec.name, ctx_obj.ctx, status, {}, note, seconds, peak)]

    entries = result if isinstance(result, list) else [result or {}]
    rows = []
    for entry in entries:
        entry = dict(entry)
        task = entry.pop("_task", spec.name)
        row_ctx = entry.pop("_ctx", ctx_obj.ctx)
        row_status = entry.pop("_status", status)
        row_note = entry.pop("_note", "") or (note if row_status == OK_OFFLOAD else "")
        if status == OK_OFFLOAD:
            # Host RAM is the resource the offload actually spends; without it the row looks
            # cheaper than the native one it replaced.
            entry["host_peak_gb"] = round(host_peak, 3)
        rows.append(Row(spec.name, task, row_ctx, row_status, entry, row_note,
                        seconds / len(entries), peak))
    return rows


def run_suite(specs, model, config, device, dtype, lengths, args,
              resume_path=None, done_rows=None) -> list[Row]:
    """Run every (benchmark, ctx) point, appending each finished point to `resume_path`.

    Points already present in `done_rows` (read from a previous run's .partial.jsonl) are
    replayed from disk instead of re-measured, and a Ctrl-C stops the sweep cleanly so the
    caller still writes the table for everything completed so far.
    """
    rows: list[Row] = []
    trained_ctx = config.get("context_length", 0)
    progress = _Progress(specs, lengths)
    args._interrupted = False

    done: dict[tuple, list[Row]] = {}
    done_by_name: dict[str, list[Row]] = {}
    for row in (done_rows or []):
        done.setdefault((row.benchmark, row.ctx), []).append(row)
        done_by_name.setdefault(row.benchmark, []).append(row)

    def record(new_rows: list[Row]):
        rows.extend(new_rows)
        if resume_path is not None:
            from benchmarks.report import append_partial
            append_partial(resume_path, new_rows)

    for spec in specs:
        points = lengths if spec.length_swept else [None]
        oom_hit = False
        for ctx in points:
            prior = done_by_name.get(spec.name) if ctx is None else done.get((spec.name, ctx))
            self_checkpointing = spec.name in SELF_CHECKPOINTING
            if prior and not self_checkpointing:
                for row in prior:
                    print(f"  -> [{spec.name} @ {ctx if ctx else 'native'}] {REUSED}: "
                          f"{row.status}  {row.metrics}", flush=True)
                rows.extend(prior)
                if any(r.status == SKIP_OOM for r in prior):
                    oom_hit = True
                progress.update(spec, ctx, 0.0, skipped=True)
                continue
            # self-checkpointing specs (genomic_benchmarks/nt/gue) are never skipped wholesale:
            # they persist and reuse task-by-task (BenchContext.task_cached/checkpoint_task) so
            # a prior partial run only needs to redo the tasks it hadn't finished yet.
            cap_arg = CTX_CAP_ARGS.get(spec.name)
            cap = getattr(args, cap_arg, 0) if cap_arg else 0
            if cap and ctx is not None and ctx > cap:
                record([Row(spec.name, spec.name, ctx, SKIP_CTX_CAP,
                            note=f"ctx > --{cap_arg.replace('_', '-')} ({cap})")])
                progress.update(spec, ctx, 0.0, skipped=True)
                continue
            if oom_hit:
                record([Row(spec.name, spec.name, ctx, SKIP_OOM_PRIOR,
                            note="skipped: a shorter context already OOM'd")])
                progress.update(spec, ctx, 0.0, skipped=True)
                continue
            if (not device.startswith("cuda")) and ctx is not None and ctx > args.cpu_max_ctx:
                record([Row(spec.name, spec.name, ctx, SKIP_CPU_CAP,
                            note=f"ctx > --cpu-max-ctx ({args.cpu_max_ctx}); "
                                 "CPU KDA runs a sequential scan and would take hours")])
                progress.update(spec, ctx, 0.0, skipped=True)
                continue

            label = f"{spec.name} @ {ctx if ctx else 'native'}"
            print(f"\n=== {label} ", "=" * max(0, 60 - len(label)), sep="", flush=True)
            prior_tasks = {r.task: r for r in prior} if prior else {}
            bctx = BenchContext(model, config, device, dtype, ctx, args, trained_ctx,
                                spec_name=spec.name, resume_path=resume_path,
                                prior_tasks=prior_tasks)
            try:
                point_rows = run_point(spec, bctx)
            except KeyboardInterrupt:
                args._interrupted = True
                if self_checkpointing and resume_path is not None:
                    from benchmarks.report import load_partial
                    recovered = [r for r in load_partial(resume_path) if r.benchmark == spec.name]
                    rows.extend(recovered)
                    print(f"\n  interrupted — {len(recovered)} task(s) already checkpointed to "
                          f"{resume_path}; rerun the same command to resume the rest", flush=True)
                else:
                    print("\n  interrupted — this point is not recorded; rerun the same command "
                          "to resume from here", flush=True)
                return rows
            for row in point_rows:
                print(f"  -> [{row.task}] {row.status}  {row.metrics}  "
                      f"[{row.seconds:.1f}s, peak {row.peak_mem_gb:.2f} GB]"
                      + (f"  ({row.note})" if row.note else ""), flush=True)
            if self_checkpointing:
                # newly finished tasks were already appended to resume_path one at a time via
                # BenchContext.checkpoint_task; reused tasks came from `prior` and were never
                # re-appended. Either way, only keep them in the in-memory table here — don't
                # append_partial the whole point again or every task would be duplicated on disk.
                rows.extend(point_rows)
            else:
                record(point_rows)
            if any(r.status == SKIP_OOM for r in point_rows):
                oom_hit = True
            point_seconds = sum(r.seconds for r in point_rows)
            progress.update(spec, ctx, point_seconds,
                            skipped=not any(r.status in (OK, OK_OFFLOAD) for r in point_rows))
            print("  " + progress.line(), flush=True)
    return rows
