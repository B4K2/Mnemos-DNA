"""Run the full evaluation suite for one model arm across a sweep of context lengths.

    # GPU, everything, default 8K -> 1M sweep
    uv run python scripts/benchmark.py --config configs/d01_kda.yaml \
        --checkpoint results/d01/final.pt

    # CPU (system RAM, no VRAM ceiling — but the KDA CPU path is a sequential scan, so the
    # per-context cap below keeps it from running for days)
    uv run python scripts/benchmark.py --config configs/d01_kda.yaml \
        --checkpoint results/d01/final.pt --device cpu --cpu-max-ctx 16384

    # just the scaling curve, architecture only (no trained weights needed)
    uv run python scripts/benchmark.py --config configs/d03_kata.yaml \
        --benchmarks scaling --no-checkpoint

Any measurement point that runs out of memory is recorded as `SKIP: OOM` and the suite moves
on; once a benchmark OOMs at one context length, longer lengths for that benchmark are marked
`SKIP: OOM at shorter ctx` without being attempted. Everything lands in one table at the end,
plus <output-dir>/<stem>.json and .csv.

Runs are resumable. Each finished point — and, for the multi-task downstream suites, each
finished task — is appended to <output-dir>/<stem>.partial.jsonl the moment it completes, so a
suite stopped with Ctrl-C or killed continues from where it left off when the same command is
relaunched. The ledger is kept after a successful run too, so relaunching a *finished* stage
(as eval_all.sh does whenever its second stage was interrupted) replays it in seconds instead
of re-measuring it. <stem>.json/.csv are merged with what is already on disk rather than
overwritten, so an interrupted rerun can never shrink a finished result file, and a ledger
written for a different checkpoint/device/mode is archived instead of replayed.
`--no-resume` forces a clean re-measurement (the old ledger is archived, not deleted).
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import bootstrap  # noqa: E402
from benchmarks import select  # noqa: E402
from benchmarks.harness import parse_lengths, run_suite  # noqa: E402
from benchmarks.model_io import count_params, load_model, resolve_dtype  # noqa: E402
from benchmarks.report import (archive_partial, init_partial, load_partial,  # noqa: E402
                               partial_fingerprint, partial_path, read_partial_meta,
                               render_extrapolation, render_summary, render_table,
                               write_outputs)
from training.config import load_config  # noqa: E402

DEFAULT_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="arm config, e.g. configs/d01_kda.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="defaults to results/<model_id>/final.pt, then the latest step_*.pt")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="benchmark the randomly-initialised architecture (scaling curve only; "
                        "accuracy numbers from an untrained model are meaningless)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--benchmarks", default="all",
                   help="comma-separated: lm,needle,scaling,lrb,species,genomic_benchmarks,nt,gue")
    p.add_argument("--lengths", default=",".join(str(v) for v in DEFAULT_LENGTHS),
                   help="context lengths for the length-swept benchmarks")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--offload", action="store_true",
                   help="on a native OOM, retry the point in low-memory mode: chunked KDA "
                        "prefill with the recurrent state carried across chunks (lossless, not "
                        "a sliding window) plus CPU activation offload for the train step. "
                        "Such rows are reported as `ok: lowmem`, never as `ok`, since their "
                        "peak-VRAM figure is not comparable to a native measurement")
    p.add_argument("--offload-chunk", type=int, default=16384,
                   help="sequence chunk size for the low-memory prefill path")

    p.add_argument("--cpu-max-ctx", type=int, default=16384,
                   help="on CPU, skip length-swept points above this (the KDA CPU path is an "
                        "O(T) Python scan; 256K would take many hours)")

    p.add_argument("--lm-batches", type=int, default=8,
                   help="minimum held-out windows per context length for the lm benchmark")
    p.add_argument("--lm-eval-tokens", type=int, default=4_000_000,
                   help="minimum held-out tokens scored per context length (windows = "
                        "max(--lm-batches, tokens/ctx)). 65K tokens resolved CE to only "
                        "~0.03 nats, enough to make checkpoints look non-monotonic")
    p.add_argument("--needle-mode", default="copy", choices=["copy", "trained", "zeroshot"],
                   help="`copy` (default): zero-shot repeat-copy delta-perplexity — how many "
                        "nats/token easier a repeated segment is than a paired novel one; "
                        "works on pretrained checkpoints because genomes naturally contain "
                        "repeats. `trained` fine-tunes on the synthetic key->value format "
                        "first (MQAR protocol; weights restored afterwards) — but note the "
                        "induction circuit did not form within thousands of steps in testing. "
                        "`zeroshot` is the raw key->value probe (chance even for pretrained "
                        "HyenaDNA; kept for reference)")
    p.add_argument("--needle-seg-len", type=int, default=256,
                   help="repeated-segment length for --needle-mode copy")
    p.add_argument("--needle-train-steps", type=int, default=300)
    p.add_argument("--needle-train-batch", type=int, default=2)
    p.add_argument("--needle-train-ctx", type=int, default=8192,
                   help="context length for the fine-tune; training distances are sampled "
                        "log-uniformly up to what fits it, so eval distances beyond it probe "
                        "capacity/extrapolation rather than format familiarity")
    p.add_argument("--needle-train-lr", type=float, default=3e-4)
    p.add_argument("--needle-train-max-d", type=int, default=0,
                   help="cap the training key->query distance (0 = up to what fits "
                        "--needle-train-ctx). Useful to check a model learns the format at "
                        "short range at all before blaming state capacity")
    p.add_argument("--needle-train-pairs", type=int, default=16,
                   help="key->value pairs (each with its own in-sequence query) planted per "
                        "training sequence — supervision per step scales with this")
    p.add_argument("--needle-max-ctx", type=int, default=131072,
                   help="skip needle points above this context length (0 = no cap). The "
                        "256K/512K/1M repeat-copy points cost hours each and add nothing to "
                        "the recall picture, which is already resolved by <=128K")
    p.add_argument("--needle-trials", type=int, default=128,
                   help="paired repeat/control windows per (ctx, distance); stderr ~ 1/sqrt(n)")
    p.add_argument("--needle-key-len", type=int, default=12)
    p.add_argument("--needle-copy-mutations", type=int, default=0,
                   help="copy mode only: point substitutions applied to the EARLIER occurrence "
                        "of the repeated segment. The scored copy is untouched, so "
                        "copy_advantage stays comparable across mutation levels — sweep "
                        "0,1,2,3 to measure how much of the copy benefit survives a degenerate "
                        "rather than exact motif match.")
    p.add_argument("--needle-distances", default=None,
                   help="comma-separated key->query distances in bp (default: 64,256,1K,...,1M "
                        "filtered to those that fit each context). d=64 is the positive "
                        "control — if it is at chance the probe says nothing about state capacity")

    p.add_argument("--downstream-mode", default="probe", choices=["probe", "finetune"],
                   help="classifier for genomic_benchmarks / nt / gue: `probe` = frozen backbone "
                        "+ logistic regression (fast, a lower bound); `finetune` = whole model "
                        "trained per task (benchmarks/finetune.py) — the protocol HyenaDNA/NT/"
                        "DNABERT-2 report, so use it for any table placed next to theirs. The "
                        "pretrained weights are restored after every task. Output stem gains "
                        "a `_finetune` suffix so probe results are never overwritten")
    p.add_argument("--ft-epochs", type=int, default=6)
    p.add_argument("--ft-max-steps", type=int, default=6000,
                   help="cap on optimizer steps per task (0 = epochs only)")
    p.add_argument("--ft-batch-size", type=int, default=32,
                   help="effective batch per optimizer step (gradient-accumulated)")
    p.add_argument("--ft-micro-batch", type=int, default=8,
                   help="sequences per forward/backward; halved automatically on OOM. The "
                        "k-mer-memory arms need ~0.6 GB per sequence in training mode")
    p.add_argument("--ft-lr", type=float, default=2e-4, help="backbone learning rate")
    p.add_argument("--ft-head-lr", type=float, default=1e-3, help="classifier head learning rate")
    p.add_argument("--ft-weight-decay", type=float, default=0.01)
    p.add_argument("--ft-bins", type=int, default=8,
                   help="positional-bin means in the fine-tune head (0 = mean+last only)")
    p.add_argument("--ft-rc", dest="ft_rc", action="store_true", default=True,
                   help="also encode the reverse complement and concatenate its features (default)")
    p.add_argument("--no-ft-rc", dest="ft_rc", action="store_false")
    p.add_argument("--ft-grad-checkpoint", action="store_true",
                   help="gradient checkpointing during fine-tuning (only needed for long tasks "
                        "such as mouse_enhancers at large batch sizes)")

    p.add_argument("--probe-train-limit", type=int, default=4000,
                   help="cap on probe training examples per task (None-like 0 = no cap)")
    p.add_argument("--probe-test-limit", type=int, default=2000)
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--probe-layers", default=None,
                   help="probe mode only: fit a separate linear probe per transformer block "
                        "(comma list, negatives count from the end, or 'all') off a single "
                        "embedding pass. Rows gain acc_layer{i}/mcc_layer{i}/f1_layer{i} per block plus "
                        "best_layer. Default (unset) keeps the last-three-blocks readout.")
    p.add_argument("--probe-max-ctx", type=int, default=8192,
                   help="upper bound on the native sequence length used by fixed-length tasks")
    p.add_argument("--gb-tasks", default=None,
                   help="comma-separated Genomic Benchmarks task names (default: all 8). "
                        "human_vs_worm,coding_vs_intergenomic is the fast sanity pair")
    p.add_argument("--nt-tasks", default=None,
                   help="comma-separated NT task names, or `all` (default: 6-task subset)")
    p.add_argument("--gue-dir", default="data/GUE",
                   help="extracted GUE/ directory (skipped if missing)")
    p.add_argument("--gue-tasks", default=None, help="comma-separated GUE task names")
    p.add_argument("--skip-gue", action="store_true",
                   help="drop the GUE (DNABERT-2) suite from this run, even when it is part "
                        "of --benchmarks all. 37 tasks x a full fine-tune dominates the "
                        "fine-tune stage's runtime; the GB/NT tables are the ones compared "
                        "against HyenaDNA/NT")

    p.add_argument("--hg38-fasta", default="data/hg38/hg38.ml.fa",
                   help="FASTA the LRB coordinate windows are cut from")
    p.add_argument("--lrb-tasks", default=None,
                   help="comma-separated: vep_eqtl,vep_clinvar,chromatin,enhancer,promoter")
    p.add_argument("--lrb-train-limit", type=int, default=256)
    p.add_argument("--lrb-test-limit", type=int, default=128)
    p.add_argument("--lrb-batch-size", type=int, default=4)
    p.add_argument("--lrb-max-ctx", type=int, default=65536,
                   help="LRB embeds hundreds of full-length windows per task; above this the "
                        "task is skipped rather than dominating the suite's runtime")

    p.add_argument("--species-dir", default="data/species",
                   help="directory of per-species .fa genomes (skipped if missing)")
    p.add_argument("--species-per-class", type=int, default=200)

    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap resamples of the test predictions per reported number "
                        "(mean + 95%% CI alongside every point estimate). 0 disables. Costs "
                        "seconds of CPU — the model is never re-run")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore (and overwrite) <stem>.partial.jsonl and re-measure every "
                        "point from scratch. By default a run appends each finished point to "
                        "that file and a relaunch of the same command replays it instead of "
                        "re-running, so an interrupted suite continues where it stopped")
    p.add_argument("--output-dir", default=None,
                   help="defaults to results/<model_id>/")
    p.add_argument("--output-stem", default=None,
                   help="defaults to benchmark_<device>")
    return p


def authenticate_hf():
    """.env holds the token as HUGGING_FACE_TOKEN (see CLAUDE.md "Secrets"), but
    huggingface_hub only looks at HF_TOKEN / HUGGINGFACE_HUB_TOKEN — bridge the two so the
    dataset-backed benchmarks download authenticated (higher rate limits, faster)."""
    load_dotenv()
    token = os.environ.get("HUGGING_FACE_TOKEN")
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    return bool(os.environ.get("HF_TOKEN"))


def _csv_list(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def resolve_checkpoint(args, config) -> str | None:
    if config.get("hf_model_id"):
        # Pretrained external arm (e.g. hyenadna): build_model() downloads the released
        # HF weights itself — there is no local checkpoint file to resolve.
        return None
    if args.no_checkpoint:
        return None
    if args.checkpoint:
        return args.checkpoint
    results_dir = Path("results") / config["model_id"]
    final = results_dir / "final.pt"
    if final.exists():
        return str(final)
    steps = sorted(results_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if steps:
        return str(steps[-1])
    raise SystemExit(
        f"no checkpoint found in {results_dir}. Pass --checkpoint, or --no-checkpoint to "
        "benchmark the untrained architecture (scaling curve only).")


def main():
    args = build_parser().parse_args()
    hf_authed = authenticate_hf()
    args.gb_tasks = _csv_list(args.gb_tasks)
    args.nt_tasks = _csv_list(args.nt_tasks)
    args.gue_tasks = _csv_list(args.gue_tasks)
    args.lrb_tasks = _csv_list(args.lrb_tasks)
    args.probe_train_limit = args.probe_train_limit or None
    args.probe_test_limit = args.probe_test_limit or None

    bootstrap.configure(args.bootstrap, args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   # ShardLoader samples with the global numpy RNG
    config = load_config(args.config)
    lengths = parse_lengths(args.lengths)
    specs = select(_csv_list(args.benchmarks) or ["all"])
    if args.skip_gue:
        specs = [s for s in specs if s.name != "gue"]
        if not specs:
            raise SystemExit("--skip-gue removed every requested benchmark")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    dtype = resolve_dtype(device, config.get("precision", "bf16"))

    checkpoint = resolve_checkpoint(args, config)
    model, step = load_model(config, checkpoint, device, dtype)
    params = count_params(model)

    print(f"arm         : {config['model_id']} ({config.get('run_name')})")
    ckpt_label = checkpoint or (
        f"pretrained HF weights ({config['hf_model_id']})" if config.get("hf_model_id")
        else "RANDOM INIT (untrained)")
    print(f"checkpoint  : {ckpt_label}" + (f"  step {step}" if step is not None else ""))
    print(f"params      : {params['total_params']:,} total, "
          f"{params['non_embedding_params']:,} non-embedding")
    print(f"device      : {device} / {dtype}"
          + (f" / {torch.cuda.get_device_name(0)}" if device == "cuda" else
             f" / {platform.processor() or platform.machine()}"))
    print(f"trained ctx : {config.get('context_length')}")
    print(f"sweep       : {lengths}")
    print(f"benchmarks  : {[s.name for s in specs]}")
    print(f"downstream  : {args.downstream_mode}"
          + (f" (epochs={args.ft_epochs}, max_steps={args.ft_max_steps}, bs={args.ft_batch_size}, "
             f"lr={args.ft_lr}/{args.ft_head_lr})" if args.downstream_mode == "finetune" else
             " (frozen backbone; lower bound on fine-tuned accuracy)"))
    print(f"lm eval     : >= {args.lm_eval_tokens:,} tokens per ctx; needle trials {args.needle_trials}"
          + (f"; needle capped at ctx {args.needle_max_ctx}" if args.needle_max_ctx else ""))
    print(f"offload     : {'on (chunked prefill fallback after native OOM)' if args.offload else 'off — pass --offload to reach contexts that OOM natively'}")
    print(f"bootstrap   : "
          + (f"{args.bootstrap} resamples per reported number (95% CI)" if args.bootstrap
             else "off — point estimates only"))
    print(f"hf auth     : {'HF_TOKEN set' if hf_authed else 'anonymous (rate-limited)'}")
    if checkpoint is None and not config.get("hf_model_id"):
        print("\nWARNING: no checkpoint loaded — only the `scaling` benchmark is meaningful; "
              "all accuracy/perplexity rows describe an untrained model.")
    if device == "cpu":
        print(f"\nNOTE: CPU run — everything lives in system RAM (no 8GB VRAM ceiling), and "
              f"length-swept points above --cpu-max-ctx={args.cpu_max_ctx} are skipped because "
              "the KDA CPU path is a sequential per-token scan.")

    out_dir = Path(args.output_dir or (Path("results") / config["model_id"]))
    stem = args.output_stem or (f"benchmark_{device}"
                                + ("_finetune" if args.downstream_mode == "finetune" else ""))
    resume_file = partial_path(out_dir, stem)
    fingerprint = {
        "model_id": config["model_id"],
        "checkpoint": checkpoint or config.get("hf_model_id"),   # external arms have no local file
        "checkpoint_step": step,
        "device": device,
        "dtype": str(dtype),
        "downstream_mode": args.downstream_mode,
    }
    if args.no_resume and resume_file.exists():
        moved = archive_partial(resume_file)
        print(f"resume      : --no-resume; previous ledger kept as {moved.name}, "
              "every point will be re-measured")
    else:
        prior_meta = read_partial_meta(resume_file)
        if prior_meta is not None and prior_meta != partial_fingerprint(fingerprint):
            changed = [k for k, v in partial_fingerprint(fingerprint).items()
                       if prior_meta.get(k) != v]
            moved = archive_partial(resume_file)
            print(f"resume      : ledger in {resume_file.name} was written for a different run "
                  f"({', '.join(changed)} changed) — kept as {moved.name}; starting fresh")
    init_partial(resume_file, fingerprint)
    done_rows = load_partial(resume_file)
    if done_rows:
        pts = len({(r.benchmark, r.ctx) for r in done_rows})
        tasks = len({(r.benchmark, r.task, r.ctx) for r in done_rows})
        print(f"resume      : {pts} point(s) / {tasks} task(s) already done in "
              f"{resume_file.name} — they will be replayed, not re-measured "
              "(pass --no-resume to redo them)")
    else:
        print(f"resume      : none; progress is appended to {resume_file.name} after every "
              "point (and after every fine-tuned task), so an interrupted run can be "
              "relaunched with the same command")

    t0 = time.time()
    rows = run_suite(specs, model, config, device, dtype, lengths, args,
                     resume_path=resume_file, done_rows=done_rows)
    elapsed = time.time() - t0
    interrupted = bool(getattr(args, "_interrupted", False))

    print("\n" + "=" * 100)
    print(f"RESULTS  arm={config['model_id']}  device={device}  "
          f"checkpoint={Path(checkpoint).name if checkpoint else ckpt_label}")
    print("=" * 100)
    print(render_table(rows))
    extrap = render_extrapolation(rows, config.get("context_length", 0))
    if extrap:
        print("\n" + extrap)
    print("\n" + render_summary(rows))
    print(f"total wall time: {elapsed / 60:.1f} min")

    meta = {
        "model_id": config["model_id"],
        "run_name": config.get("run_name"),
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "device": device,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "trained_context_length": config.get("context_length"),
        "lengths": lengths,
        "benchmarks": [s.name for s in specs],
        "downstream_mode": args.downstream_mode,
        "lm_eval_tokens": args.lm_eval_tokens,
        "needle_trials": args.needle_trials,
        "seed": args.seed,
        "offload": bool(args.offload),
        "wall_time_s": round(elapsed, 1),
        "interrupted": interrupted,
        **params,
    }
    json_path, csv_path = write_outputs(rows, meta, out_dir, stem)
    print(f"wrote {json_path}\nwrote {csv_path}")
    if interrupted:
        print(f"\nINTERRUPTED — {len(rows)} row(s) kept in {resume_file.name}. Rerun the same "
              f"command to continue from the point that was cut short; {json_path.name} "
              "carries everything finished so far (rows from earlier runs are merged in, "
              "never dropped).")
    else:
        print(f"\ncomplete — the ledger {resume_file.name} is kept, so relaunching this same "
              "command replays these points instead of re-measuring them (use --no-resume "
              "to force a fresh run).")


if __name__ == "__main__":
    main()
