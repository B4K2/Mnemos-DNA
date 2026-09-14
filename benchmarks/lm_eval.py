"""Next-nucleotide language-modelling quality on held-out hg38 chromosomes, swept over
context length.

Uses the `test` split of the pretokenized shards (the Basenji/Enformer chromosome split that
HyenaDNA also uses — see dataset/prepare.py), never a random split.

Reported per context length:
  ce          cross-entropy in nats/token
  ppl         perplexity = exp(ce)
  bpn         bits per nucleotide = ce / ln(2) — the unit the genomics LM literature uses;
              5-token uniform chance is ~2.32 bits, a 4-base uniform baseline is 2.0
  acc         top-1 next-nucleotide accuracy

Sweeping ctx past config["context_length"] is exactly the extrapolation test (§5 of CLAUDE.md);
scripts/benchmark.py derives the ppl-vs-trained-length ratio from these rows.
"""
import math

import torch

from dataset import ShardLoader
from benchmarks.bootstrap import bootstrap_mean
from benchmarks.harness import SkipBenchmark

_LOADER_CACHE: dict[tuple, ShardLoader] = {}


def _loader(config, split, device):
    key = (config["dataset_dir"], split, device)
    if key not in _LOADER_CACHE:
        try:
            _LOADER_CACHE[key] = ShardLoader(config["dataset_dir"], split, device=device)
        except FileNotFoundError as exc:
            raise SkipBenchmark(str(exc)) from exc
    return _LOADER_CACHE[key]


@torch.no_grad()
def run(bctx, split: str = "test") -> dict:
    ctx = bctx.ctx
    loader = _loader(bctx.config, split, bctx.device)
    if min(loader.shard_lengths) <= ctx + 1:
        raise SkipBenchmark(f"no test shard longer than ctx={ctx}")

    # Fixed *token* budget per context length, not a fixed batch count: 8 windows at 8K is
    # 65K tokens, which resolves CE to only ~+-0.03 nats and made successive checkpoints look
    # like they were getting worse. --lm-eval-tokens (default 4M) is the floor; --lm-batches
    # still applies as a minimum window count for the long contexts.
    n_batches = max(1, bctx.args.lm_batches, math.ceil(bctx.args.lm_eval_tokens / ctx))
    total_nll, total_correct, total_tokens = 0.0, 0, 0
    window_ce, window_acc = [], []   # per-window, the independent unit for the bootstrap
    for _ in range(n_batches):
        x, y = loader.get_batch(1, ctx)
        logits, _ = bctx.model(x)
        logits = logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += nll.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += y.numel()
        window_ce.append(nll.item() / y.numel())
        window_acc.append((logits.argmax(dim=-1) == y).float().mean().item())

    ce = total_nll / total_tokens
    # CE/accuracy vary far more across genomic windows than within one, so the bootstrap
    # resamples *windows*; the ppl CI is the CE CI pushed through exp, not a separate resample.
    boot = bootstrap_mean(window_ce, "ce")
    boot.update({k.replace("ce_", "ppl_"): round(math.exp(min(v, 20)), 3)
                 for k, v in boot.items() if k.startswith("ce_")})
    boot.update(bootstrap_mean(window_acc, "acc"))
    return {
        "ce": round(ce, 4),
        "ppl": round(math.exp(min(ce, 20)), 3),
        "bpn": round(ce / math.log(2), 4),
        "acc": round(total_correct / total_tokens, 4),
        "eval_tokens": total_tokens,
        "eval_windows": len(window_ce),
        **boot,
    }
