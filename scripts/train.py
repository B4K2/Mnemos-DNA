"""Train a single model arm (see CLAUDE.md model matrix, §3).

Usage:
    uv run python scripts/train.py --config configs/d01_kda.yaml

Checkpoint handling is automatic:
  - If results/{model_id}/final.pt already exists, training is considered complete and the
    script exits immediately (use --force to retrain from scratch).
  - Otherwise, the latest results/{model_id}/step_N.pt is picked up and resumed from
    automatically. Pass --resume explicitly only to resume from a specific checkpoint.
  - On successful completion, final.pt is written and all step_*.pt checkpoints are deleted
    except the last one (step_{total}.pt), which is kept so the run can be extended later.

Extending a finished run (e.g. 5B -> 10B tokens) without editing the yaml:

    uv run python scripts/train.py --config configs/d06_kmer_fastweight.yaml \
        --resume results/d06/final.pt --training-tokens 10_000_000_000

  The old final.pt is archived as final_step_{N}.pt first, so the 5B result is never lost.
  Note the LR schedule is recomputed for the new total: a cosine run that had decayed to
  lr_min re-warms to the cosine value at (old_steps / new_steps) — a warm restart, printed
  at startup. MLflow continues the same run.

Ctrl+C (SIGINT) or SIGTERM triggers a graceful stop: the current step finishes, a checkpoint is
saved, and the process exits 0. Resume by just re-running the same command.
"""
import argparse
import importlib
import math
import os
import signal
import sys
import time
from pathlib import Path

# D06's fast-weight scan allocates data-dependent (P, C, dv) blocks whose size swings with the
# k-mer collision structure of the window, which fragments the caching allocator badly enough to
# OOM with hundreds of MiB reserved-but-unallocated. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mlflow
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from dataset import ShardLoader
from training.config import load_config
from training.curriculum import curriculum_enabled, curriculum_plan, micro_batch_size


# How many optimizer steps in a row may be lost to an OOM that even a single-sequence
# micro-batch cannot absorb before the run is aborted with an actionable message.
MAX_CONSECUTIVE_OOM_STEPS = 3


def _split_micro(mb: int, factor: int) -> list[int]:
    """Split one micro-batch of `mb` sequences into `factor` smaller ones (same total)."""
    part = max(1, mb // factor)
    out, left = [], mb
    while left > 0:
        out.append(min(part, left))
        left -= part
    return out

load_dotenv()

KEEP_N_CHECKPOINTS = 2

_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print(f"\n[train] received signal {signum}, will checkpoint and stop after this step...")


def build_param_groups(model, weight_decay: float):
    """Split parameters into decayed / non-decayed groups.

    Parameters carrying `_no_weight_decay` (the KDA forget-gate `A_log` and `dt_bias`, see
    models/d01/attention.py) must be excluded: decay drags the gate back toward fast forgetting,
    which is precisely the failure that left the state with a ~1.5-token half-life in the first
    d01_rope run. Norm gains and biases are excluded for the usual reason — they are not weights
    whose magnitude should be penalized.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "_no_weight_decay", False) or p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_lr_scheduler(optimizer, config: dict, total_steps: int):
    """Linear warmup -> cosine decay, opt-in via `lr_schedule: cosine`.

    Default is `none` (constant LR) because that is what D01 was actually trained under —
    turning a schedule on globally would silently make later arms non-comparable to it.
    Note that `warmup_steps` had no effect at all before this function existed; a run whose
    total step count is below its warmup_steps was, and under `none` still is, running at
    full LR from step 0.
    """
    schedule = config.get("lr_schedule", "none")
    if schedule == "none":
        return None
    if schedule != "cosine":
        raise ValueError(f"unknown lr_schedule {schedule!r} (expected 'none' or 'cosine')")

    warmup = min(config.get("warmup_steps", 0), max(total_steps - 1, 0))
    min_ratio = config.get("lr_min_ratio", 0.1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(config: dict):
    module = importlib.import_module(f"models.{config.get('model_module', config['model_id'])}")
    return module.build_model(config)


def find_latest_checkpoint(results_dir: Path) -> Path | None:
    ckpts = sorted(results_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def prune_checkpoints(results_dir: Path, keep_n: int | None = KEEP_N_CHECKPOINTS):
    """keep_n=None deletes every step_*.pt (used once final.pt is written)."""
    ckpts = sorted(results_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    stale = ckpts if keep_n is None else ckpts[:-keep_n]
    for path in stale:
        path.unlink()
        print(f"[train] pruned checkpoint {path.name}")


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]


@torch.no_grad()
def estimate_val_loss(model, val_loader, config, num_batches: int | None = None,
                      block_size: int | None = None, batch_size: int | None = None):
    num_batches = num_batches or int(config.get("eval_batches", 40))
    block_size = block_size or config["context_length"]
    batch_size = batch_size or config["batch_size"]
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = val_loader.get_batch(batch_size, block_size)
        use_autocast = torch.cuda.is_available() and config.get("precision") == "bf16"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="path to a specific step_N.pt checkpoint")
    parser.add_argument("--force", action="store_true", help="retrain from scratch, ignoring existing checkpoints")
    parser.add_argument("--max-steps", type=int, default=None, help="override total steps for smoke tests")
    parser.add_argument("--training-tokens", type=int, default=None,
                        help="override config training_tokens (use with --resume results/<id>/final.pt "
                             "to extend a finished run; the yaml is left untouched)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.training_tokens is not None:
        config["training_tokens"] = int(args.training_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = config["model_id"]
    results_dir = Path("results") / model_id
    results_dir.mkdir(parents=True, exist_ok=True)

    final_path = results_dir / "final.pt"
    if final_path.exists() and not args.force and not args.resume:
        print(f"[train] {final_path} already exists, {model_id} is complete. Skipping "
              f"(use --force to redo, or --resume {final_path} --training-tokens N to extend).")
        return

    # Parameters (and the optimizer state) always stay fp32; `precision: bf16` means bf16 autocast
    # for the forward/backward only. Pure-bf16 parameters silently froze every parameter with
    # |x| >~ 0.25 (bf16 ulp at |x|=4 is 0.031 > an lr=1e-3 Adam step): the D05 k-mer read_gate
    # and D04 HOLA cache_gate sat at their -4.0 init for the entire run (found 2026-08-18).
    # Checkpoints are saved in fp32; evaluation still casts to bf16 (benchmarks/model_io.py).
    if config.get("precision") == "bf16" and device != "cuda":
        print("[train] precision bf16 requested but no CUDA; running fp32")
    model_dtype = torch.float32
    model = build_model(config).to(device, dtype=model_dtype)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, config["weight_decay"]),
        lr=config["learning_rate"],
    )

    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
    elif not args.force:
        resume_path = find_latest_checkpoint(results_dir)

    start_step = 0
    resumed_tokens = None
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, model, optimizer)
        resumed_tokens = torch.load(resume_path, map_location="cpu", weights_only=False)["config"].get("training_tokens")
        print(f"[train] resumed from {resume_path} at step {start_step}")
        if final_path.exists() and resume_path.resolve() == final_path.resolve():
            archive = results_dir / f"final_step_{start_step}.pt"
            if not archive.exists():
                final_path.rename(archive)
                print(f"[train] archived previous {final_path.name} -> {archive.name}")

    train_loader = ShardLoader(config["dataset_dir"], "train", device=device)
    val_loader = ShardLoader(config["dataset_dir"], "valid", device=device)

    tokens_per_step = config["batch_size"] * config["context_length"] * config["grad_accum_steps"]
    total_steps = config.get("max_steps") or (config["training_tokens"] // tokens_per_step)
    scheduler = build_lr_scheduler(optimizer, config, total_steps)
    if scheduler is not None:
        # Optimizer state is restored from the checkpoint but the scheduler's is not, so replay
        # it up to start_step; without this a resumed run restarts warmup from zero.
        for _ in range(start_step):
            scheduler.step()
    if start_step >= total_steps:
        raise SystemExit(f"[train] checkpoint step {start_step} >= total_steps {total_steps}; "
                         "pass --training-tokens N (N > current) to extend the run")
    if scheduler is not None and resumed_tokens and resumed_tokens != config["training_tokens"]:
        old_total = resumed_tokens // tokens_per_step
        print(f"[train] EXTENDING run: {resumed_tokens:,} -> {config['training_tokens']:,} tokens "
              f"({old_total:,} -> {total_steps:,} steps). Cosine schedule recomputed for the new "
              f"total: LR restarts at {optimizer.param_groups[0]['lr']:.2e} (warm restart) "
              f"instead of the {config['learning_rate'] * config.get('lr_min_ratio', 0.1):.2e} it "
              "had decayed to. Expect a transient loss bump.")
    if config.get("warmup_steps", 0) >= total_steps:
        print(f"[train] WARNING: warmup_steps={config['warmup_steps']} >= total_steps={total_steps} "
              f"({tokens_per_step:,} tokens/step). Lower tokens/step or warmup_steps.")

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    run_id_file = results_dir / "mlflow_run_id.txt"
    resume_run_id = run_id_file.read_text().strip() if (resume_path and run_id_file.exists()) else None

    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    with mlflow.start_run(run_id=resume_run_id, run_name=config["run_name"]) as run:
        run_id_file.write_text(run.info.run_id)
        if resume_run_id is None:
            mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})
            mlflow.set_tags({"model_id": model_id, "attention_type": config.get("attention_type", "?")})
            mlflow.log_metric("total_params", model.num_params(), step=0)

        model.train()
        step = start_step
        oom_batches = 0
        consecutive_oom_steps = 0
        oom_split = 1          # current micro-batch subdivision factor (reset after a good step)
        t_start = time.time()

        pbar = tqdm(
            total=total_steps,
            initial=start_step,
            desc=model_id,
            unit="step",
            dynamic_ncols=True,
        )
        try:
          while step < total_steps and not _stop_requested:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            use_autocast = device == "cuda" and config.get("precision") == "bf16"
            oomed = False
            # Length + micro-batch plan for this step (training/curriculum.py). Without a
            # `seq_curriculum` block this is exactly the old context_length x grad_accum loop.
            seq_len, micros = curriculum_plan(config, step, total_steps, tokens_per_step)
            # An OOM at this step's shape is usually deterministic (the planned micro-batch simply
            # does not fit), so re-planning it unchanged would spin forever. Halve the micro-batch
            # and retry; only give up on the step once single-sequence micro-batches also OOM.
            if oom_split > 1:
                split = [m for mb in micros for m in _split_micro(mb, oom_split)]
                micros = split
            for mb in micros:
                x, y = train_loader.get_batch(mb, seq_len)
                weight = (mb * seq_len) / tokens_per_step   # token-weighted mean over micro-batches
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                        _, loss = model(x, y)
                    (loss * weight).backward()
                    accum_loss += loss.item() * weight
                except torch.OutOfMemoryError:
                    # Peak memory here is data-dependent (d06's per-slot chain lengths scale with the
                    # window's k-mer skew), so an unlucky repeat-rich window can OOM a config that is
                    # comfortable on average. Drop the batch rather than losing the whole run.
                    oomed = True
                    oom_batches += 1
                    del x, y
                    loss = None
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    pbar.write(f"[train] OOM at step {step + 1}, batch skipped "
                               f"({oom_batches} total)")
                    break

            if oomed:
                mlflow.log_metric("oom_batches", oom_batches, step=step)
                if max(micros) > 1:
                    # Retry the same step with smaller micro-batches (same tokens/step, so the
                    # LR schedule and token budget are untouched).
                    oom_split *= 2
                    pbar.write(f"[train] retrying step {step + 1} with micro-batches "
                               f"split {oom_split}x")
                    continue
                consecutive_oom_steps += 1
                if consecutive_oom_steps >= MAX_CONSECUTIVE_OOM_STEPS:
                    raise RuntimeError(
                        f"OOM at step {step + 1} with micro-batch 1 x {seq_len} tokens on "
                        f"{consecutive_oom_steps} consecutive steps -- this shape does not fit. "
                        f"Lower context_length / seq_curriculum.max_micro_tokens and resume."
                    )
                # Single-sequence micro-batches still OOM: skip the step rather than spin on it.
                pbar.write(f"[train] skipping step {step + 1} (OOM at 1 x {seq_len})")
                step += 1
                pbar.update(1)
                if scheduler is not None:
                    scheduler.step()
                continue

            consecutive_oom_steps = 0
            oom_split = 1

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_now = optimizer.param_groups[0]["lr"]  # read before the scheduler advances it
            if scheduler is not None:
                scheduler.step()
            step += 1

            elapsed = time.time() - t_start
            tok_per_sec = tokens_per_step * (step - start_step) / max(elapsed, 1e-6)
            metrics = {
                    "train_loss": accum_loss,
                    "grad_norm": grad_norm.item(),
                    "learning_rate": lr_now,
                    "tokens_per_sec": tok_per_sec,
                    "tokens_seen": step * tokens_per_step,
                    "seq_len": seq_len,
                    "peak_vram_gb": (
                        torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                    ),
            }
            if curriculum_enabled(config):
                # per-length train loss so the short/long regimes can be read separately in the
                # mix phase (train_loss alone blends whatever length this step drew)
                metrics[f"train_loss_len{seq_len}"] = accum_loss
                metrics["train_loss_short" if seq_len <= int(config["seq_curriculum"].get("start_len", 1024))
                        else "train_loss_long" if seq_len >= config["context_length"]
                        else "train_loss_mid"] = accum_loss
            mlflow.log_metrics(metrics, step=step)

            if hasattr(model, "kmer_metrics") and (step % 10 == 0 or step == 1):
                km = model.kmer_metrics()
                if km:
                    mlflow.log_metrics(km, step=step)
                    if "router_entropy" in km and km["router_entropy"] < 0.3 and step > 1000:
                        pbar.write(f"[train] WARNING step {step}: router entropy "
                                   f"{km['router_entropy']:.3f} (norm.) -- set-associativity collapsing")
            pbar.set_postfix(loss=f"{accum_loss:.4f}", tok_s=f"{tok_per_sec:,.0f}")
            pbar.update(1)

            if step % config["eval_interval_steps"] == 0 or step == total_steps:
                val_loss = estimate_val_loss(model, val_loader, config)
                mlflow.log_metric("val_loss", val_loss, step=step)
                msg = f"[train] step {step}  val_loss {val_loss:.4f}"
                if curriculum_enabled(config):
                    # short-context val: the "did the short-range grammar survive" monitor,
                    # same token count per batch as the long one (tokens_per_step / short).
                    cur = config["seq_curriculum"]
                    short = int(cur.get("start_len", 1024))
                    bs_short = max(1, min(tokens_per_step // short, micro_batch_size(config, short)))
                    val_short = estimate_val_loss(model, val_loader, config,
                                                  block_size=short, batch_size=bs_short)
                    mlflow.log_metric("val_loss_short", val_short, step=step)
                    msg += f"  val_loss_short@{short} {val_short:.4f}"
                pbar.write(msg)

            if step % config["checkpoint_interval_steps"] == 0 or step == total_steps or _stop_requested:
                ckpt_path = results_dir / f"step_{step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, config)
                pbar.write(f"[train] saved checkpoint {ckpt_path}")
                prune_checkpoints(results_dir, keep_n=KEEP_N_CHECKPOINTS)
        except BaseException as exc:
            # Anything that kills the loop (an OOM the per-batch handler could not absorb, a
            # KeyboardInterrupt that landed mid-step, a hardware fault) would otherwise throw away
            # every step since the last periodic checkpoint. Save first, then re-raise.
            if step > start_step:
                pbar.close()
                try:
                    torch.cuda.empty_cache()
                    ckpt_path = results_dir / f"step_{step}.pt"
                    save_checkpoint(ckpt_path, model, optimizer, step, config)
                    print(f"[train] {type(exc).__name__} at step {step}: "
                          f"saved emergency checkpoint {ckpt_path}")
                    prune_checkpoints(results_dir, keep_n=KEEP_N_CHECKPOINTS)
                except Exception as save_exc:  # never mask the original failure
                    print(f"[train] emergency checkpoint FAILED: {save_exc!r}")
            mlflow.set_tag("status", "crashed")
            raise

        pbar.close()

        if step >= total_steps:
            save_checkpoint(final_path, model, optimizer, step, config)
            # keep final.pt + the last step_{total}.pt (identical weights; the step file is the
            # one --resume picks up when the run is later extended with --training-tokens)
            last_step = results_dir / f"step_{step}.pt"
            if not last_step.exists():
                save_checkpoint(last_step, model, optimizer, step, config)
            prune_checkpoints(results_dir, keep_n=1)
            mlflow.set_tag("status", "complete")
            print(f"[train] training complete at step {step}, final checkpoint: {final_path} "
                  f"(+ {last_step.name} kept for extension)")
        elif _stop_requested:
            mlflow.set_tag("status", "stopped")
            print(f"[train] stopped early at step {step}. Re-run the same command to resume.")
            sys.exit(0)


if __name__ == "__main__":
    main()
