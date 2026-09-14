"""Resume/merge semantics of the benchmark suite (no model, no GPU).

Guards the failure that lost ~24h of finished evaluation: a relaunched suite re-measuring
points it had already completed, and an interrupted rerun overwriting a complete
<stem>.json with a shorter one.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import report as R                      # noqa: E402
from benchmarks.harness import OK, Row, run_suite       # noqa: E402
from benchmarks.__init__ import BenchmarkSpec           # noqa: E402


def _args(**kw):
    return Namespace(seed=0, offload=False, cpu_max_ctx=10 ** 9, downstream_mode="probe", **kw)


def _spec(name, fn, swept=True):
    return BenchmarkSpec(name, fn, swept, name)


FP = {"model_id": "d06", "checkpoint": "results/d06/final.pt", "checkpoint_step": 100,
      "device": "cuda", "dtype": "torch.bfloat16", "downstream_mode": "finetune"}


def test_finished_points_are_replayed_not_remeasured(tmp_path):
    calls = []

    def fn(bctx):
        calls.append(bctx.ctx)
        return {"ppl": 3.0}

    ledger = R.partial_path(tmp_path, "bench")
    R.init_partial(ledger, FP)
    rows = run_suite([_spec("lm", fn)], None, {}, "cpu", None, [8192, 16384], _args(),
                     resume_path=ledger, done_rows=R.load_partial(ledger))
    assert calls == [8192, 16384] and len(rows) == 2

    # relaunch: same ledger, nothing re-measured
    calls.clear()
    rows2 = run_suite([_spec("lm", fn)], None, {}, "cpu", None, [8192, 16384], _args(),
                      resume_path=ledger, done_rows=R.load_partial(ledger))
    assert calls == []
    assert [(r.benchmark, r.ctx, r.metrics) for r in rows2] == \
           [(r.benchmark, r.ctx, r.metrics) for r in rows]

    # a newly requested length is measured, the old ones are not
    calls.clear()
    run_suite([_spec("lm", fn)], None, {}, "cpu", None, [8192, 16384, 32768], _args(),
              resume_path=ledger, done_rows=R.load_partial(ledger))
    assert calls == [32768]


def test_multi_task_suite_resumes_task_by_task(tmp_path):
    """genomic_benchmarks/nt/gue run many independent fine-tunes inside ONE point; a Ctrl-C
    partway through must not cost the tasks already finished."""
    trained = []
    killed = []

    def fn(bctx):
        out = []
        for task in ["a", "b", "c"]:
            cached = bctx.task_cached(task)
            if cached is not None:
                out.append(dict(cached.metrics, _task=task, _status=cached.status))
                continue
            if task == "c" and not killed:
                killed.append(task)
                trained.append(task)
                raise KeyboardInterrupt       # killed while fine-tuning the third task
            trained.append(task)
            entry = {"acc": 0.9, "_task": task}
            bctx.checkpoint_task(task, entry)
            out.append(entry)
        return out

    ledger = R.partial_path(tmp_path, "bench_ft")
    R.init_partial(ledger, FP)
    spec = _spec("nt", fn, swept=False)
    args = _args()
    rows = run_suite([spec], None, {}, "cpu", None, [], args,
                     resume_path=ledger, done_rows=R.load_partial(ledger))
    assert args._interrupted
    assert trained == ["a", "b", "c"]
    assert {r.task for r in rows} == {"a", "b"}          # the two that finished survive

    trained.clear()
    rows2 = run_suite([spec], None, {}, "cpu", None, [], _args(),
                      resume_path=ledger, done_rows=R.load_partial(ledger))
    assert trained == ["c"]                              # only the unfinished task reruns
    assert {r.task for r in rows2} == {"a", "b", "c"}
    assert all(r.status == OK for r in rows2)
    # and no task is duplicated on disk
    tasks = [r.task for r in R.load_partial(ledger)]
    assert sorted(tasks) == ["a", "b", "c"]


def test_interrupted_rerun_cannot_shrink_a_result_file(tmp_path):
    done = Row("nt", "promoter_all", 600, OK, {"acc": 0.85})
    R.write_outputs([done], dict(FP), tmp_path, "bench")

    partial = Row("nt", "splice_sites_all", 600, OK, {"acc": 0.82})
    json_path, _ = R.write_outputs([partial], dict(FP), tmp_path, "bench")
    rows = json.loads(json_path.read_text())["rows"]
    assert {r["task"] for r in rows} == {"promoter_all", "splice_sites_all"}

    redone = Row("nt", "promoter_all", 600, OK, {"acc": 0.91})
    json_path, _ = R.write_outputs([redone], dict(FP), tmp_path, "bench")
    rows = json.loads(json_path.read_text())["rows"]
    assert len(rows) == 2
    assert [r["metrics"]["acc"] for r in rows if r["task"] == "promoter_all"] == [0.91]


def test_a_different_checkpoint_is_archived_not_merged(tmp_path):
    row = Row("nt", "promoter_all", 600, OK, {"acc": 0.85})
    R.write_outputs([row], dict(FP), tmp_path, "bench")
    other = dict(FP, checkpoint_step=200)
    json_path, _ = R.write_outputs([row], other, tmp_path, "bench")
    assert len(json.loads(json_path.read_text())["rows"]) == 1
    assert list(tmp_path.glob("bench.2*.json"))          # previous run kept, not clobbered

    ledger = R.partial_path(tmp_path, "bench")
    R.init_partial(ledger, FP)
    assert R.read_partial_meta(ledger) != R.partial_fingerprint(other)
    moved = R.archive_partial(ledger)
    assert moved.exists() and not ledger.exists()


def test_legacy_headerless_ledger_still_loads(tmp_path):
    ledger = R.partial_path(tmp_path, "bench")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(Row("lm", "lm", 8192, OK, {"ppl": 3.0}).to_dict()) + "\n"
                      + '{"truncated": ')          # killed mid-write
    assert R.read_partial_meta(ledger) is None
    assert len(R.load_partial(ledger)) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
