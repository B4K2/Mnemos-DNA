"""Run every experiment the 2026-09-15 review asked for, on one GPU, in order, resumably.

Three controls around the central recall claim (d01_new = no memory, d06 = k-mer memory):

  R  addressing   -- is it the *k-mer addressing* that buys recall, or just having a big memory?
                     d06_poshash (same 4^8 table, slot = hash(position)) and d06_shared (one 16x16
                     matrix for every event). A random permutation of k-mer addresses is NOT a
                     control: it relabels slots and is indistinguishable from d06.
  K  k sweep      -- k in {4, 6, 7} next to the existing k=8 (d06). k=9 (2 GiB table) and k=10
                     (8 GiB) do not fit the 5060. The overwrite-rate analysis predicts the recall
                     cliff at D ~ 4^k * d_k.
  S  KDA state    -- pure KDA with 4x / 16x the recurrent state of d01_new (state per layer
                     = d_model^2 / num_heads): 128/h1 (param-matched), 256/h1. 16x is the card's
                     ceiling: d_model 512 (32x) OOMs at 131072 in the smoke test (2026-09-15).

Every arm is matched to d06/d01_new on context (131072), tokens (5B), LR schedule and data order.
Each arm runs: VRAM+speed smoke on real hg38 (fails the arm before a doomed multi-day run), full
training, then needle copy (+ 1-3 SNP mutations for the addressing/k arms), lm, and scaling.
d06 (paper checkpoint step 34,177) and d01_new (final.pt, step 34,408) are re-measured under the
same output stems, so every row in the review tables comes from one command line.

    uv run python scripts/run_review_queue.py --list
    uv run python scripts/run_review_queue.py                 # everything not yet done
    uv run python scripts/run_review_queue.py --only R1.smoke R1.train   # ids from --list
    uv run python scripts/run_review_queue.py --smoke configs/d01_state16x.yaml   # one smoke check

Pause / resume / --redo / --skip / --stop-on-failure behave exactly as in run_paper_queue.py
(this script reuses its runner). State: results/_queue/review_queue.json. The GPU lock is shared
with run_paper_queue.py, so the two queues can never run on the card at the same time.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_paper_queue as q  # noqa: E402

CTX = "131072"
NEEDLE = ["--benchmarks", "needle", "--needle-mode", "copy", "--lengths", CTX,
          "--needle-trials", "128", "--bootstrap", "1000"]

# (model_id, config, train hours, description). Hours = smoke-measured s/step x steps on the 5060
# (2026-09-15), excluding evals and checkpoints.
ARMS = [
    ("d01_state4x", "configs/d01_state4x.yaml", 14.0, "S: pure KDA 4x state (128/h1, param-matched)"),
    ("d06_poshash", "configs/d06_poshash.yaml", 40.0, "R: k-mer table, position-hashed addresses"),
    ("d06_k6", "configs/d06_k6.yaml", 28.0, "K: k=6 (4,096 slots)"),
    ("d01_state16x", "configs/d01_state16x.yaml", 32.0, "S: pure KDA 16x state (256/h1)"),
    ("d06_k4", "configs/d06_k4.yaml", 32.0, "K: k=4 (256 slots)"),
    ("d06_k7", "configs/d06_k7.yaml", 36.0, "K: k=7 (16,384 slots)"),
    # last: one 262k-event chain -> ~86h even at the largest scan_chunk that fits (512)
    ("d06_shared", "configs/d06_shared.yaml", 86.0, "R: one shared fast-weight matrix, no addressing"),
]
MEMORY_ARMS = {"d06_poshash", "d06_shared", "d06_k6", "d06_k7", "d06_k4"}

# Existing reference arms: (model_id, config, checkpoint). d06 never wrote final.pt; 34,177 is the
# paper's checkpoint and the same ~4.5B-token budget as a new arm's final.pt.
REFERENCE = [("d06", q.D06_CFG, q.D06_CKPT), ("d01_new", q.D01_CFG, "results/d01_new/final.pt")]


def smoke_cmd(config):
    return [sys.executable, os.path.abspath(__file__), "--smoke", config]


def eval_jobs(prefix, mid, cfg, memory, ck=None):
    ck = ck or f"results/{mid}/final.pt"
    b = lambda *rest: q.bench(cfg, *rest, checkpoint=ck)
    jobs = [(f"{prefix}.needle", 0.5, f"{mid}: needle copy @131072",
             b(*NEEDLE, "--output-stem", "review_needle"), [ck])]
    if memory:
        for m in (1, 2, 3):
            jobs.append((f"{prefix}.mut{m}", 0.5, f"{mid}: needle copy, {m} substitution(s)",
                         b(*NEEDLE, "--needle-copy-mutations", str(m),
                           "--output-stem", f"review_needle_mut{m}"), [ck]))
    jobs += [
        (f"{prefix}.lm", 0.5, f"{mid}: held-out lm @131072",
         b("--benchmarks", "lm", "--lengths", CTX, "--bootstrap", "1000",
           "--output-stem", "review_lm"), [ck]),
        (f"{prefix}.scaling", 0.2, f"{mid}: batch-1 VRAM/throughput scaling sweep",
         b("--benchmarks", "scaling", "--output-stem", "review_scaling"), [ck]),
    ]
    return jobs


def build_jobs():
    jobs = []
    for i, (mid, cfg, ck) in enumerate(REFERENCE, 1):
        jobs += eval_jobs(f"REF{i}", mid, cfg, memory=mid == "d06", ck=ck)
    for i, (mid, cfg, hours, desc) in enumerate(ARMS, 1):
        p = f"{desc[0]}{i}"
        jobs.append((f"{p}.smoke", 0.1, f"{mid}: VRAM + step-time smoke on real hg38 ({desc})",
                     smoke_cmd(cfg), []))
        jobs.append((f"{p}.train", hours, f"{mid}: train 5B tokens @131072 ({desc})",
                     q.train(cfg), [f"results/_queue/smoke_{mid}.ok"]))
        jobs += eval_jobs(p, mid, cfg, memory=mid in MEMORY_ARMS)
    return jobs


def smoke(config_path: str) -> int:
    """One real-data fwd+bwd+AdamW step at the training shape, then a timed one. Writes
    results/_queue/smoke_<id>.ok only if it fits, which gates that arm's training job. Real
    tokens, not random ones: random tokens underestimate the k-mer arms' VRAM ~2x (d06 memory)."""
    import torch
    from dataset import ShardLoader
    from training.config import load_config
    from train import build_model, build_param_groups

    cfg = load_config(config_path)
    mid, ctx = cfg["model_id"], int(cfg["context_length"])
    ok_path = Path(f"results/_queue/smoke_{mid}.ok")
    ok_path.parent.mkdir(parents=True, exist_ok=True)
    ok_path.unlink(missing_ok=True)
    model = build_model(cfg).cuda().train()
    opt = torch.optim.AdamW(build_param_groups(model, cfg["weight_decay"]), lr=1e-4)
    loader = ShardLoader(cfg["dataset_dir"], "train", device="cuda")
    n_params = sum(p.numel() for p in model.parameters())
    times = []
    try:
        for _ in range(2):
            x, y = loader.get_batch(batch_size=1, block_size=ctx)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    except torch.OutOfMemoryError:
        print(f"[smoke] {mid}: OOM at ctx {ctx} -- this arm does not fit the card as configured")
        return 1
    if not torch.isfinite(loss):
        print(f"[smoke] {mid}: non-finite loss {loss.item()}")
        return 1
    peak = torch.cuda.max_memory_allocated() / 2**30
    steps = int(cfg["training_tokens"]) // (ctx * int(cfg["batch_size"]) * int(cfg["grad_accum_steps"]))
    proj_h = steps * times[-1] / 3600
    print(f"[smoke] {mid}: {n_params / 1e6:.2f}M params | peak {peak:.2f} GiB @ ctx {ctx} | "
          f"{times[-1]:.2f}s/step | {steps} steps -> ~{proj_h:.1f}h training (no evals)")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if peak > 0.9 * total:
        print(f"[smoke] {mid}: peak is within 10% of the {total:.1f} GiB card -- refusing; "
              f"allocator fragmentation over a long run will OOM")
        return 1
    ok_path.write_text(f"peak_gib {peak:.3f}\nsec_per_step {times[-1]:.3f}\nprojected_hours {proj_h:.1f}\n")
    return 0


def main() -> int:
    if "--smoke" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--smoke", required=True)
        return smoke(ap.parse_args().smoke)

    q.JOBS = build_jobs()
    q.STATE_PATH = Path("results/_queue/review_queue.json")
    # q.LOCK_PATH is left as paper_queue.lock on purpose: one lock per GPU, not per queue.
    original_verify = q.verify_job

    def verify(cmd):
        if os.path.abspath(__file__) in cmd:           # smoke job: its .ok file is the product
            from training.config import load_config
            mid = load_config(cmd[-1])["model_id"]
            ok = Path(f"results/_queue/smoke_{mid}.ok")
            return None if ok.exists() else f"smoke did not write {ok}"
        return original_verify(cmd)

    q.verify_job = verify
    sys.argv[0] = "run_review_queue.py"
    return q.main()


if __name__ == "__main__":
    sys.exit(main())
