"""Run a sequence of training configs one after another, in-process order, on this single GPU.

Each config runs as its own `train.py` subprocess (so a crash in one arm doesn't kill the
queue), and only starts once the previous one exits. train.py is idempotent — an arm that
already has results/{model_id}/final.pt is skipped instantly, and a partially-trained arm
resumes automatically from its latest checkpoint — so re-running this script after a crash or
a Ctrl+C just picks up where the whole queue left off.

Usage:
    uv run python scripts/run_queue.py                          # all configs/d*.yaml, in order
    uv run python scripts/run_queue.py --configs configs/d01_kda.yaml configs/d02_kda_local.yaml
    uv run python scripts/run_queue.py --force                  # retrain every arm from scratch
"""
import argparse
import signal
import subprocess
import sys
from pathlib import Path

_interrupted = False


def _run_child(cmd) -> int:
    """Run train.py, forwarding Ctrl+C as a graceful-stop request rather than dying on it.

    Ctrl+C goes to the whole foreground process group, so train.py already sees the SIGINT
    and sets its own _stop_requested flag. The parent's job is simply to stay alive and wait
    while the child finishes its current step and writes a checkpoint — at ~400s/step that
    can take several minutes. Dying here (the subprocess.run default) orphans the child
    mid-step and throws away everything since the last checkpoint.

    A second Ctrl+C escalates to SIGTERM, a third to SIGKILL, for when you really mean now.
    """
    global _interrupted
    proc = subprocess.Popen(cmd)
    escalation = 0

    def on_signal(signum, frame):
        nonlocal escalation
        global _interrupted
        _interrupted = True
        escalation += 1
        if escalation == 1:
            print("\n[queue] interrupt received — waiting for train.py to checkpoint and exit "
                  "(this can take a few minutes at long step times). Ctrl+C again to force.")
        elif escalation == 2:
            print("\n[queue] second interrupt — sending SIGTERM to train.py.")
            proc.terminate()
        else:
            print("\n[queue] third interrupt — killing train.py. Progress since the last "
                  "checkpoint will be lost.")
            proc.kill()

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                continue  # handler already ran; keep waiting on the child
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def default_configs() -> list[Path]:
    return sorted(Path("configs").glob("d*.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None, help="config paths, in run order")
    parser.add_argument("--force", action="store_true", help="passed through to train.py")
    args = parser.parse_args()

    configs = [Path(c) for c in args.configs] if args.configs else default_configs()
    if not configs:
        print("[queue] no configs found (looked in configs/d*.yaml)")
        sys.exit(1)

    print(f"[queue] {len(configs)} model(s) queued: {[c.stem for c in configs]}")

    for i, config_path in enumerate(configs, start=1):
        print(f"\n[queue] ({i}/{len(configs)}) starting {config_path}")
        cmd = [sys.executable, "scripts/train.py", "--config", str(config_path)]
        if args.force:
            cmd.append("--force")

        returncode = _run_child(cmd)

        if _interrupted:
            print(f"\n[queue] stopped by interrupt during {config_path}. Re-run this script to "
                  f"resume it from its latest checkpoint and continue the remaining "
                  f"{len(configs) - i} model(s).")
            sys.exit(0)

        if returncode != 0:
            print(f"[queue] {config_path} exited with code {returncode}, stopping queue.")
            print(f"[queue] re-run this script to retry {config_path} and continue the remaining "
                  f"{len(configs) - i} model(s) — completed/in-progress ones will be skipped/resumed.")
            sys.exit(returncode)

        print(f"[queue] ({i}/{len(configs)}) finished {config_path}")

    print(f"\n[queue] all {len(configs)} model(s) complete.")


if __name__ == "__main__":
    main()
