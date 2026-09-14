"""Memory / throughput scaling curve (§6.3) — the figure where O(N) vs O(N^2) becomes visible.

Three separate measurements per context length, each isolated so one OOM does not hide the
others:
  infer_*   forward only, no_grad, batch 1 — what an inference deployment costs
  decode_*  latency of producing one more token given a full ctx-length prefix. These arms have
            no KV/recurrent-state cache implemented yet, so this is a full recompute and is
            labelled `nocache`; it is an upper bound, and the KDA arms' true cached decode cost
            is O(1) in ctx once a cache exists.
  train_*   forward + backward with gradient checkpointing, batch 1 — the number that decides
            what context length is actually trainable on the 8GB card

Peak memory here is measured by this benchmark itself (CUDA allocator counter / RSS sampler)
rather than taken from the harness row, so it reflects only the measured region.
"""
import gc
import time
from contextlib import contextmanager

import torch

from benchmarks.harness import _RSSSampler


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@contextmanager
def _measure_peak(device, out: dict, key: str):
    """CUDA allocator counter on GPU, sampled RSS on CPU — so the CPU rows carry a real
    number instead of nan."""
    _reset(device)
    with _RSSSampler() as rss:
        yield
        _sync(device)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") \
        else rss.peak / 1e9
    out[key] = round(peak, 3)


def _reset(device):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _timed(fn, device, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.time()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.time() - t0) / iters


def run(bctx) -> dict:
    ctx, device, model = bctx.ctx, bctx.device, bctx.model
    ids = torch.randint(3, 8, (1, ctx), device=device)
    warmup = 1 if ctx >= 65536 or not bctx.is_cuda else 2
    iters = 1 if ctx >= 65536 or not bctx.is_cuda else 3
    out = {}
    # Bidirectional MLM wrappers (Caduceus) refuse forward() so lm/needle can't compute a causal
    # CE from them. VRAM/throughput don't depend on the objective, so time the backbone instead.
    causal = getattr(model, "causal", True)
    fwd = model if causal else model.encode

    # --- inference: forward only
    model.eval()
    with _measure_peak(device, out, "infer_peak_gb"), torch.no_grad():
        sec = _timed(lambda: fwd(ids), device, warmup, iters)
    out["prefill_tok_per_s"] = round(ctx / sec)
    out["prefill_s"] = round(sec, 4)

    # --- decode: one extra token, no cache (full recompute) -> upper bound
    ids1 = torch.randint(3, 8, (1, ctx + 1), device=device)
    with _measure_peak(device, out, "decode_peak_gb"), torch.no_grad():
        sec1 = _timed(lambda: fwd(ids1), device, 0, 1)
    out["decode_s_per_tok_nocache"] = round(sec1, 4)

    # --- training step: forward + backward with gradient checkpointing
    # The released external checkpoints are inference wrappers: they return logits and a None
    # loss, so there is no training step to measure. Report the inference figures and say so,
    # rather than letting the missing loss surface as an ERROR row that also discards them.
    probe_loss = None
    if causal:
        with torch.no_grad():
            _, probe_loss = model(ids[:, :2], ids[:, :2])
    if probe_loss is None:
        out["train_peak_gb"] = None
        out["train_tok_per_s"] = None
        out["_note"] = "no training step: this arm is an inference-only wrapper"
        return out

    try:
        model.train()
        model.gradient_checkpointing = True

        def train_step():
            model.zero_grad(set_to_none=True)
            _, loss = model(ids[:, :-1], ids[:, 1:])
            loss.backward()

        with _measure_peak(device, out, "train_peak_gb"):
            sec_train = _timed(train_step, device, 0, 1)
        out["train_tok_per_s"] = round(ctx / sec_train)
    except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        out["train_peak_gb"] = None
        out["train_tok_per_s"] = None
        out["_note"] = "train step OOM (inference figures above are still valid)"
        # The backward pass is the one place where moving activations to host RAM genuinely
        # helps: unlike the no_grad forward, it retains every saved tensor until consumed.
        # Reported under separate keys so an offloaded train step is never mistaken for a
        # native one in the §6.3 curve.
        if getattr(bctx.args, "offload", False) and device.startswith("cuda"):
            _reset(device)
            try:
                with _measure_peak(device, out, "train_peak_gb_offload"), \
                        torch.autograd.graph.save_on_cpu(pin_memory=True):
                    sec_off = _timed(train_step, device, 0, 1)
                out["train_tok_per_s_offload"] = round(ctx / sec_off)
                out["_note"] += "; train step succeeded with CPU activation offload"
            except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError) as exc2:
                if "out of memory" not in str(exc2).lower():
                    raise
                out["_note"] += "; CPU activation offload also OOM'd"
    finally:
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing = False
        model.eval()
        _reset(device)

    return out
