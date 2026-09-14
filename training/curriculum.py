"""Sequence-length curriculum for pretraining (D07 onward; off unless `seq_curriculum` is set).

Why: a gated recurrence has one set of timescales per channel. Trained only at 131K, the decay
gates settle on long half-lives and 300-600bp grammar (NT promoter / splice tasks) rides on a
state that is mostly remembering the last 50kb. Measured 2026-08-30: with a locus-aware head,
a *random-init* d06 fine-tunes to 0.86 on NT splice_sites_all, the 4.5B-token 131K checkpoint
to 0.82. Two phases, both fixed in the yaml (no loss-driven switching -- a bang-bang controller
oscillates, changes batch shape every switch and makes runs irreproducible across arms):

  1. warmup   : length doubles from `start_len` to `context_length` over the first
                `warmup_frac` of steps (short grammar first, HyenaDNA-style length warmup).
  2. mix      : every step draws its length from `mix` ({length: weight}); both regimes stay
                in the loss for the rest of training so neither timescale can drift away.

Tokens per optimizer step are constant (`batch_size * context_length * grad_accum_steps`), so
the LR schedule and the token budget are untouched; short lengths just mean more sequences per
step, processed in micro-batches of at most `max_micro_batch` sequences AND `max_micro_tokens`
tokens. Both caps are needed: peak VRAM tracks the tokens in flight (`mb * seq_len`), with a
further ~0.6 GB per *sequence* on top for the k-mer memory. Measured for d07 on the 5060
(fwd+bwd, hg38 windows, bf16 + grad checkpointing):

    131072x1  4.35 GiB   65536x1  2.54     32768x2  2.98    16384x2  1.55    1024x4  0.49
     65536x2  5.52 GiB   32768x4  6.17 (OOMs in-run)        16384x4  3.29

i.e. a micro-batch of 131072 tokens is affordable as ONE sequence and not as four. Without the
token cap the warmup stage at seq_len 32768 planned 4x32768 and OOMed every step (observed
2026-08-30, run d07 step ~2396: the handler skips the batch, so the stage made no progress).

yaml:
  seq_curriculum:
    start_len: 1024
    warmup_frac: 0.1
    mix: {1024: 0.25, 16384: 0.25, 131072: 0.5}
    max_micro_batch: 4
    max_micro_tokens: 65536     # default: context_length // 2
"""
import math

import numpy as np


def curriculum_enabled(config: dict) -> bool:
    cur = config.get("seq_curriculum")
    return bool(cur) and bool(cur.get("enabled", True))


def curriculum_length(config: dict, step: int, total_steps: int) -> int:
    """Sequence length for optimizer step `step` (0-based). Deterministic in (seed, step), so a
    resumed run replays the same lengths."""
    ctx = int(config["context_length"])
    if not curriculum_enabled(config):
        return ctx
    cur = config["seq_curriculum"]
    start_len = int(cur.get("start_len", 1024))
    assert ctx % start_len == 0 and (ctx // start_len) & (ctx // start_len - 1) == 0, \
        "context_length must be start_len * 2^n"
    warm = int(float(cur.get("warmup_frac", 0.1)) * total_steps)
    if step < warm:
        stages = int(math.log2(ctx // start_len)) + 1          # start_len, 2*start_len, ..., ctx
        return start_len << min(stages - 1, int(stages * step / warm))
    mix = cur.get("mix") or {ctx: 1.0}
    lens = sorted(int(k) for k in mix)
    probs = np.array([float(mix[k]) for k in mix], dtype=np.float64)
    probs = probs[np.argsort([int(k) for k in mix])]
    probs /= probs.sum()
    rng = np.random.default_rng([int(cur.get("seed", 0)), int(step)])
    return int(rng.choice(lens, p=probs))


def micro_batch_size(config: dict, seq_len: int) -> int:
    """Largest number of sequences of `seq_len` to run in one micro-batch: capped both by
    `max_micro_batch` (per-sequence k-mer cost) and `max_micro_tokens` (activation cost)."""
    cur = config.get("seq_curriculum") or {}
    max_micro = max(1, int(cur.get("max_micro_batch", 8)))
    max_tokens = int(cur.get("max_micro_tokens", int(config["context_length"]) // 2))
    return max(1, min(max_micro, max_tokens // seq_len))


def curriculum_plan(config: dict, step: int, total_steps: int, tokens_per_step: int):
    """-> (seq_len, [micro_batch sizes]) for one optimizer step. Without a curriculum this is
    the legacy plan: context_length x grad_accum_steps micro-batches of batch_size."""
    if not curriculum_enabled(config):
        return int(config["context_length"]), [int(config["batch_size"])] * int(config["grad_accum_steps"])
    seq_len = curriculum_length(config, step, total_steps)
    n_seq = max(1, tokens_per_step // seq_len)
    max_micro = micro_batch_size(config, seq_len)
    micros = [max_micro] * (n_seq // max_micro)
    if n_seq % max_micro:
        micros.append(n_seq % max_micro)
    return seq_len, micros
