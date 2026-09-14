"""Run the whole paper-revision job list on one GPU, in order, resumably.

Every job is a `train.py` or `benchmark.py` subprocess. Both are already resumable on their own
terms -- train.py picks up from its latest checkpoint, benchmark.py replays finished
(benchmark, ctx) / per-task points out of `<stem>.partial.jsonl` -- so this script's only extra
job is to remember which *jobs* finished, so a restart does not re-enter a 12-hour training run
that is already done.

    uv run python scripts/run_paper_queue.py --list          # what will run, in order, with ETAs
    uv run python scripts/run_paper_queue.py                 # run everything not yet done
    uv run python scripts/run_paper_queue.py --only A3 D1    # just those jobs
    uv run python scripts/run_paper_queue.py --from B1       # that job onwards
    uv run python scripts/run_paper_queue.py --redo A3       # forget A3's result and run it again

Pausing: Ctrl+C once. The signal reaches the child too, so train.py finishes its current step
and writes a checkpoint (minutes at 131k context) and benchmark.py flushes its ledger; this
script waits for that, records the job as *interrupted* rather than done, and exits. Re-running
the same command resumes that job mid-flight and continues down the list. A second Ctrl+C sends
SIGTERM, a third SIGKILL -- both lose everything since the last checkpoint/ledger flush.

State lives in results/_queue/paper_queue.json and is safe to hand-edit or delete.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

STATE_PATH = Path("results/_queue/paper_queue.json")

D06_CKPT = "results/d06/step_34177.pt"
D07_CKPT = "results/d07/step_33900.pt"
D06_CFG = "configs/d06_kmer_fastweight.yaml"
D07_CFG = "configs/d07_kmer_local.yaml"
D01_CFG = "configs/d01_new.yaml"
D02_CFG = "configs/d02_kda_local.yaml"

# The fine-tune protocol every downstream row must share. These are benchmark.py's current
# defaults, but they are passed explicitly so the command line -- and therefore the ledger --
# records the protocol even if a default changes later. The whole reason this queue exists is
# that d06 and d07 were previously fine-tuned under different caps, epochs and pooling heads.
FT = ["--downstream-mode", "finetune", "--ft-epochs", "6", "--ft-max-steps", "6000",
      "--probe-train-limit", "4000", "--probe-test-limit", "2000", "--ft-bins", "8", "--ft-rc"]


def bench(config, *rest, checkpoint=None, no_checkpoint=False):
    cmd = [sys.executable, "scripts/benchmark.py", "--config", config]
    if no_checkpoint:
        cmd.append("--no-checkpoint")
    elif checkpoint:
        cmd += ["--checkpoint", checkpoint]
    return cmd + list(rest)


def train(config):
    return [sys.executable, "scripts/train.py", "--config", config]


# id, hours (wall estimate), description, command, prerequisite paths
JOBS = [
    # ---- the long pole: the matched pure-KDA control that replaces D04 ------------------
    ("T1", 12.0, "train d01_new (pure KDA, NoPE, 131k ctx, 5B tokens)",
     train(D01_CFG), []),

    # ---- A: protocol repair. Nothing downstream is comparable until these land ----------
    ("A1", 3.0, "d07 GB+NT seed 0 at the matched protocol (pairs with d06/ft_seed0)",
     bench(D07_CFG, "--benchmarks", "genomic_benchmarks,nt", *FT,
           "--seed", "0", "--output-stem", "ft_seed0", checkpoint=D07_CKPT), [D07_CKPT]),
    ("A2", 0.5, "d06 seed 0 gap fill: NT enhancers + H3K4me3 (missing from ft_seed0)",
     bench(D06_CFG, "--benchmarks", "nt", "--nt-tasks", "enhancers,H3K4me3", *FT,
           "--seed", "0", "--output-stem", "ft_seed0", checkpoint=D06_CKPT), [D06_CKPT]),
    ("A3", 12.0, "d06 GUE seed 0 at the matched protocol (d07's existing GUE run already is)",
     bench(D06_CFG, "--benchmarks", "gue", *FT,
           "--seed", "0", "--output-stem", "ft_seed0_gue", checkpoint=D06_CKPT), [D06_CKPT]),

    # ---- B: seeds. mean +/- std across 3 seeds on the 14 GB/NT tasks -------------------
    ("B1", 3.0, "d06 GB+NT seed 1",
     bench(D06_CFG, "--benchmarks", "genomic_benchmarks,nt", *FT,
           "--seed", "1", "--output-stem", "ft_seed1", checkpoint=D06_CKPT), [D06_CKPT]),
    ("B2", 3.0, "d06 GB+NT seed 2",
     bench(D06_CFG, "--benchmarks", "genomic_benchmarks,nt", *FT,
           "--seed", "2", "--output-stem", "ft_seed2", checkpoint=D06_CKPT), [D06_CKPT]),
    ("B3", 3.0, "d07 GB+NT seed 1",
     bench(D07_CFG, "--benchmarks", "genomic_benchmarks,nt", *FT,
           "--seed", "1", "--output-stem", "ft_seed1", checkpoint=D07_CKPT), [D07_CKPT]),
    ("B4", 3.0, "d07 GB+NT seed 2",
     bench(D07_CFG, "--benchmarks", "genomic_benchmarks,nt", *FT,
           "--seed", "2", "--output-stem", "ft_seed2", checkpoint=D07_CKPT), [D07_CKPT]),

    # ---- C: random-init control under the SAME head the arms use ----------------------
    # The published control used a different (locus-aware) head, so "worse than random init"
    # currently compares two different heads on one task.
    ("C1", 1.5, "random-init d06 architecture, NT suite, seed 0 (no pretrained weights)",
     bench(D06_CFG, "--benchmarks", "nt", *FT, "--seed", "0",
           "--output-stem", "ft_randinit_seed0", no_checkpoint=True), []),
    ("C2", 1.5, "random-init d06 architecture, NT suite, seed 1",
     bench(D06_CFG, "--benchmarks", "nt", *FT, "--seed", "1",
           "--output-stem", "ft_randinit_seed1", no_checkpoint=True), []),

    # ---- D: put the mutation sweep on one frozen checkpoint ---------------------------
    # m=0 is step 34,177 (the paper's checkpoint) but m=1,2,3 were run at 34,178.
    ("D1", 0.7, "d06 needle copy, 1 substitution, at step 34,177",
     bench(D06_CFG, "--benchmarks", "needle", "--needle-copy-mutations", "1",
           "--lengths", "131072", "--needle-trials", "128",
           "--output-stem", "needle_mut1_34177", checkpoint=D06_CKPT), [D06_CKPT]),
    ("D2", 0.7, "d06 needle copy, 2 substitutions, at step 34,177",
     bench(D06_CFG, "--benchmarks", "needle", "--needle-copy-mutations", "2",
           "--lengths", "131072", "--needle-trials", "128",
           "--output-stem", "needle_mut2_34177", checkpoint=D06_CKPT), [D06_CKPT]),
    ("D3", 0.7, "d06 needle copy, 3 substitutions, at step 34,177",
     bench(D06_CFG, "--benchmarks", "needle", "--needle-copy-mutations", "3",
           "--lengths", "131072", "--needle-trials", "128",
           "--output-stem", "needle_mut3_34177", checkpoint=D06_CKPT), [D06_CKPT]),

    # ---- E: the missing recall control ------------------------------------------------
    # d02's existing needle rows are zero-shot mode (no ce_repeat), which is why the
    # window-only arm is absent from the recall figure.
    ("E1", 0.5, "d02 needle in COPY mode at 131,072 (its current rows are zero-shot)",
     bench(D02_CFG, "--benchmarks", "needle", "--needle-mode", "copy",
           "--lengths", "131072", "--needle-trials", "128",
           "--output-stem", "needle_copy", checkpoint="results/d02/final.pt"),
     ["results/d02/final.pt"]),

    # ---- F: baseline harness fixes ----------------------------------------------------
    # The external "OOM ceilings" all come from `species` at --probe-batch-size 8, which is
    # eight long sequences at once, not a batch-1 inference forward. `scaling` is the batch-1
    # measurement and costs about a minute per arm.
    ("F1", 0.1, "batch-1 scaling sweep for the 4 external checkpoints (real OOM ceiling + latency)",
     None, []),   # expanded below into one sub-command per config
    ("F2", 4.0, "species at --probe-batch-size 1 for external + our arms",
     None, []),

    # ---- G: the new pure-KDA arm's numbers --------------------------------------------
    ("G1", 8.0, "d01_new full probe suite (lm / needle / scaling / species / lrb)",
     bench(D01_CFG, "--benchmarks", "lm,needle,scaling,species,lrb",
           checkpoint="results/d01_new/final.pt"), ["results/d01_new/final.pt"]),
    ("G2", 3.0, "d01_new GB+NT seed 0 at the matched protocol",
     bench(D01_CFG, "--benchmarks", "genomic_benchmarks,nt", *FT, "--seed", "0",
           "--output-stem", "ft_seed0", checkpoint="results/d01_new/final.pt"),
     ["results/d01_new/final.pt"]),

    # ---- H: does the pretraining-damage finding generalize beyond our architecture? ----
    # Pretrained HyenaDNA-1M vs a random-init copy of the same architecture, both fine-tuned
    # on the NT suite under the exact matched protocol used for every other downstream row.
    ("H1", 2.0, "HyenaDNA-1M (pretrained ckpt) NT suite, matched fine-tune protocol, seed 0",
     bench("configs/h01_hyenadna_1m.yaml", "--benchmarks", "nt", *FT, "--seed", "0",
           "--output-stem", "ft_seed0"), []),
    ("H2", 2.0, "HyenaDNA-1M random-init control, NT suite, seed 0",
     bench("configs/h03_hyenadna_1m_randinit.yaml", "--benchmarks", "nt", *FT, "--seed", "0",
           "--output-stem", "ft_randinit_seed0"), []),
    ("H3", 2.0, "HyenaDNA-1M random-init control, NT suite, seed 1",
     bench("configs/h03_hyenadna_1m_randinit.yaml", "--benchmarks", "nt", *FT, "--seed", "1",
           "--output-stem", "ft_randinit_seed1"), []),
]

EXTERNAL_CFGS = ["configs/h01_hyenadna_1m.yaml", "configs/h02_hyenadna_450k.yaml",
                 "configs/c01_caduceus_ph.yaml", "configs/c02_caduceus_ps.yaml"]


def _expand(jobs):
    """F1/F2 are one logical job each but several subprocesses; give each its own id so a
    restart does not redo the arms that already finished.

    External arms carry `hf_model_id`, so `resolve_checkpoint` short-circuits and no checkpoint
    flag is passed at all -- and no `--offload`, because chunked prefill is exact only for the
    KDA recurrence (HyenaDNA's implicit convolution and Caduceus' bidirectional scan both read
    the whole sequence at once, so their OOM ceiling is the measurement, not an obstacle)."""
    out = []
    for job in jobs:
        jid, hours, desc, cmd, needs = job
        if jid == "F1":
            for i, cfg in enumerate(EXTERNAL_CFGS, 1):
                out.append((f"F1.{i}", hours / len(EXTERNAL_CFGS),
                            f"batch-1 scaling: {Path(cfg).stem}",
                            bench(cfg, "--benchmarks", "scaling",
                                  "--output-stem", "scaling_b1"), []))
        elif jid == "F2":
            # No --offload for our arms either: the low-memory chunked retry at 1M context costs
            # 5-10h per point. A native OOM is recorded as `SKIP: OOM` and the sweep moves on.
            arms = [(cfg, None) for cfg in EXTERNAL_CFGS] + [
                (D06_CFG, D06_CKPT), (D07_CFG, D07_CKPT),
                (D02_CFG, "results/d02/final.pt")]
            for i, (cfg, ck) in enumerate(arms, 1):
                out.append((f"F2.{i}", hours / len(arms),
                            f"species @ probe-batch-size 1: {Path(cfg).stem}",
                            bench(cfg, "--benchmarks", "species", "--probe-batch-size", "1",
                                  "--downstream-mode", "probe", "--bootstrap", "1000",
                                  "--species-dir", "data/species",
                                  "--output-stem", "species_b1", checkpoint=ck),
                            [ck] if ck else []))
        else:
            out.append(job)
    return out


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(STATE_PATH)      # atomic, so a Ctrl+C mid-write cannot corrupt the ledger


LOCK_PATH = Path("results/_queue/paper_queue.lock")


def acquire_lock():
    """Refuse to start a second queue on the same GPU.

    Two instances sharing one 8GB card do not queue -- they collide, and the loser's points all
    come back `SKIP: OOM`, which (before the check in `verify_job`) looked like a clean run and
    got written to the ledger as done. The lock file holds the owning pid so a stale one is
    readable rather than mysterious.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown pid"
        print(f"[queue] another run_paper_queue.py is already running ({owner}). "
              f"Wait for it, or stop it first -- two queues on one GPU just OOM each other.")
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid {os.getpid()} started {datetime.now().isoformat(timespec='seconds')}\n")
    handle.flush()
    return handle


def result_path(cmd) -> Path | None:
    """Where the benchmark job for `cmd` writes its rows, or None for a training job."""
    if "scripts/benchmark.py" not in " ".join(cmd):
        return None
    config = cmd[cmd.index("--config") + 1]
    stem = cmd[cmd.index("--output-stem") + 1] if "--output-stem" in cmd else "benchmark_cuda"
    from training.config import load_config
    return Path("results") / load_config(config)["model_id"] / f"{stem}.json"


def verify_job(cmd) -> str | None:
    """Return a complaint if a job exited 0 without actually producing usable numbers.

    A benchmark run whose every point OOMed or errored still exits 0. Marking that `done`
    silently bakes an empty row set into the paper's inputs, so treat it as a failure and make
    the queue stop on it.
    """
    if "scripts/train.py" in " ".join(cmd):
        config = cmd[cmd.index("--config") + 1]
        from training.config import load_config
        final = Path("results") / load_config(config)["model_id"] / "final.pt"
        return None if final.exists() else f"training finished but {final} was not written"

    path = result_path(cmd)
    if path is None or not path.exists():
        return f"no result file at {path}"
    rows = json.loads(path.read_text()).get("rows", [])
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        statuses = sorted({r.get("status", "?") for r in rows}) or ["no rows at all"]
        return f"{path} has 0 usable rows ({', '.join(statuses)})"
    return None


_interrupted = False


def run_child(cmd) -> int:
    """Run one job, forwarding Ctrl+C as a graceful-stop request rather than dying on it.

    The child runs in its own process group (`start_new_session`), so a terminal Ctrl+C reaches
    only this parent, which then forwards the signal deliberately. That makes an interactive
    Ctrl+C and a `kill -INT <queue pid>` from a script behave identically -- without it, the
    scripted form stops the parent while the child keeps running.

    The forwarded SIGINT starts the child's own shutdown: train.py checkpoints after the current
    step, benchmark.py has already flushed each finished point to its ledger. The parent then
    stays alive and waits -- dying here orphans the child mid-step and throws away the work
    since its last checkpoint. Second interrupt escalates to SIGTERM, third to SIGKILL.
    """
    global _interrupted
    proc = subprocess.Popen(cmd, start_new_session=True)
    escalation = 0

    def signal_child(sig):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def on_signal(signum, frame):
        nonlocal escalation
        global _interrupted
        _interrupted = True
        escalation += 1
        if escalation == 1:
            print("\n[queue] pause requested -- waiting for the job to checkpoint and exit. "
                  "This can take several minutes at 131k context. Ctrl+C again to force.",
                  flush=True)
            signal_child(signal.SIGINT)
        elif escalation == 2:
            print("\n[queue] second interrupt -- SIGTERM to the child.", flush=True)
            signal_child(signal.SIGTERM)
        else:
            print("\n[queue] third interrupt -- SIGKILL. Work since the last checkpoint is lost.",
                  flush=True)
            signal_child(signal.SIGKILL)

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                continue
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def fmt_h(h: float) -> str:
    return f"{h:4.1f}h" if h >= 1 else f"{int(round(h * 60)):3d}m "


def main() -> int:
    global STATE_PATH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the queue and exit")
    ap.add_argument("--dry-run", action="store_true", help="print each command instead of running")
    ap.add_argument("--only", nargs="*", default=None, help="run just these job ids")
    ap.add_argument("--skip", nargs="*", default=(), help="skip these job ids")
    ap.add_argument("--from", dest="start_at", default=None, help="start at this job id")
    ap.add_argument("--redo", nargs="*", default=(), help="forget these jobs' results first")
    ap.add_argument("--state", default=str(STATE_PATH), help="ledger path")
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="stop the queue at the first failed job (default: record it and move on)")
    args = ap.parse_args()

    STATE_PATH = Path(args.state)
    state = load_state()
    for jid in args.redo:
        state.pop(jid, None)
    if args.redo:
        save_state(state)
        print(f"[queue] forgot {list(args.redo)}")

    jobs = _expand(JOBS)
    ids = [j[0] for j in jobs]
    for jid in list(args.only or ()) + list(args.skip) + list(args.redo):
        if jid not in ids:
            print(f"[queue] unknown job id {jid!r}; known ids: {' '.join(ids)}")
            return 2
    if args.start_at:
        if args.start_at not in ids:
            print(f"[queue] unknown --from {args.start_at!r}")
            return 2
        jobs = jobs[ids.index(args.start_at):]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    jobs = [j for j in jobs if j[0] not in args.skip]

    if args.list or args.dry_run:
        total = pending = 0.0
        for jid, hours, desc, cmd, needs in jobs:
            done = state.get(jid, {}).get("status") == "done"
            missing = [p for p in needs if not Path(p).exists()]
            mark = "done  " if done else ("BLOCKED" if missing else "todo  ")
            total += hours
            pending += 0.0 if done else hours
            print(f"  [{mark}] {jid:6s} {fmt_h(hours)}  {desc}")
            if missing:
                print(f"           waiting on: {', '.join(missing)}")
            if args.dry_run:
                print(f"           $ {' '.join(cmd)}")
        print(f"\n  {len(jobs)} jobs, {total:.1f}h total, {pending:.1f}h still to run")
        return 0

    lock = acquire_lock()
    if lock is None:
        return 3

    started = time.time()
    failed = []
    remaining = sum(h for jid, h, *_ in jobs if state.get(jid, {}).get("status") != "done")
    print(f"[queue] {len(jobs)} jobs queued, ~{remaining:.1f}h of work outstanding. "
          f"Ctrl+C once to pause; re-run this command to resume.\n")

    for n, (jid, hours, desc, cmd, needs) in enumerate(jobs, 1):
        entry = state.get(jid, {})
        if entry.get("status") == "done":
            print(f"[queue] ({n}/{len(jobs)}) {jid} already done "
                  f"({entry.get('seconds', 0) / 3600:.1f}h on {entry.get('finished', '?')[:10]}) "
                  f"-- skipping")
            continue
        missing = [p for p in needs if not Path(p).exists()]
        if missing:
            print(f"[queue] ({n}/{len(jobs)}) {jid} BLOCKED, missing {', '.join(missing)} "
                  f"-- skipping for now, re-run once it exists")
            state[jid] = {"status": "blocked", "missing": missing}
            save_state(state)
            continue

        eta = datetime.now() + timedelta(hours=remaining)
        print(f"\n[queue] ({n}/{len(jobs)}) {jid}: {desc}")
        print(f"[queue]   est {fmt_h(hours)} | ~{remaining:.1f}h left | queue ETA {eta:%a %H:%M}")
        print(f"[queue]   $ {' '.join(cmd)}", flush=True)

        t0 = time.time()
        rc = run_child(cmd)
        elapsed = time.time() - t0
        remaining = max(0.0, remaining - hours)

        if _interrupted:
            state[jid] = {"status": "interrupted", "seconds": round(elapsed, 1),
                          "returncode": rc, "stopped": datetime.now().isoformat(timespec="seconds")}
            save_state(state)
            print(f"\n[queue] paused during {jid} after {elapsed / 3600:.2f}h. Re-run this "
                  f"command to resume it and continue the remaining {len(jobs) - n} job(s).")
            return 0
        if rc != 0:
            state[jid] = {"status": "failed", "returncode": rc, "seconds": round(elapsed, 1),
                          "finished": datetime.now().isoformat(timespec="seconds")}
            save_state(state)
            failed.append((jid, f"exited {rc}"))
            if args.stop_on_failure:
                print(f"[queue] {jid} exited {rc} after {elapsed / 3600:.2f}h -- stopping. Fix it, "
                      f"then re-run (finished jobs are skipped). --skip {jid} to move past it.")
                return rc
            print(f"[queue] {jid} FAILED (exited {rc}) after {elapsed / 3600:.2f}h -- "
                  f"continuing with the next job; re-run later to retry it.", flush=True)
            continue

        complaint = verify_job(cmd)
        if complaint:
            state[jid] = {"status": "failed", "returncode": 0, "seconds": round(elapsed, 1),
                          "problem": complaint,
                          "finished": datetime.now().isoformat(timespec="seconds")}
            save_state(state)
            print(f"[queue] {jid} exited 0 but produced nothing usable: {complaint}")
            failed.append((jid, complaint))
            if args.stop_on_failure:
                print(f"[queue] stopping. Investigate, then re-run (add --redo {jid} to re-measure; "
                      f"a stale <stem>.partial.jsonl may need deleting first).")
                return 1
            print(f"[queue] {jid} FAILED -- continuing with the next job. Retry later with "
                  f"--redo {jid} (a stale <stem>.partial.jsonl may need deleting first).", flush=True)
            continue

        state[jid] = {"status": "done", "returncode": 0, "seconds": round(elapsed, 1),
                      "finished": datetime.now().isoformat(timespec="seconds")}
        save_state(state)
        print(f"[queue] ({n}/{len(jobs)}) {jid} done in {elapsed / 3600:.2f}h "
              f"(estimated {hours:.1f}h)")

    hours_run = (time.time() - started) / 3600
    if failed:
        print(f"\n[queue] finished in {hours_run:.1f}h with {len(failed)} FAILED job(s):")
        for jid, why in failed:
            print(f"[queue]   {jid}: {why}")
        print("[queue] fix them and re-run this command; finished jobs are skipped, failed ones retried.")
        return 1
    print(f"\n[queue] all jobs complete in {hours_run:.1f}h.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
