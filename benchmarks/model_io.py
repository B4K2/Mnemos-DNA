"""Checkpoint loading + the two primitives every benchmark is built on:
`embed_sequences` (pooled multi-layer representations for linear probes) and `lm_logits`.
"""
import importlib

import numpy as np
import torch

from dataset.tokenizer import PAD_ID, TOKEN_TO_ID, VOCAB_SIZE, encode


def build_model(config: dict):
    module = importlib.import_module(f"models.{config.get('model_module', config['model_id'])}")
    return module.build_model(config)


def load_model(config: dict, checkpoint: str | None, device: str, dtype: torch.dtype):
    """Builds the arm from its config and loads weights. `checkpoint=None` gives a randomly
    initialised model — only meaningful for the memory/throughput benchmark, which measures
    the architecture, not the weights."""
    model = build_model(config)
    loaded_step = None
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        loaded_step = ckpt.get("step")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.gradient_checkpointing = False
    return model, loaded_step


def tokenize_batch(seqs, ctx: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequences -> (ids, mask). Longer than ctx: centre-crop (keeps the annotated locus,
    which sits mid-window in every dataset used here). Shorter: right-pad with [PAD].

    Width is the longest sequence *in this batch*, not ctx: padding is wasted compute, and on
    the CPU path (a sequential per-token scan) it is wasted linearly in the padding. Pooling
    masks the padding out either way, so this changes speed, not results.
    """
    toks = [encode(s.upper()) for s in seqs]
    toks = [t if len(t) <= ctx else t[(len(t) - ctx) // 2:][:ctx] for t in toks]
    width = max((len(t) for t in toks), default=1)

    ids = np.full((len(toks), width), PAD_ID, dtype=np.int64)
    mask = np.zeros((len(toks), width), dtype=bool)
    for i, tok in enumerate(toks):
        ids[i, : len(tok)] = tok
        mask[i, : len(tok)] = True
    return (torch.from_numpy(ids).to(device),
            torch.from_numpy(mask).to(device))


def reverse_complement(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, T) right-padded token ids -> reverse complement, still right-padded. A<->T, C<->G,
    everything else (N, specials) maps to itself. This is DNA's natural "reverse direction":
    the other strand read 5'->3', so a causal model gets a second view whose *last* token is
    the original 5' end."""
    comp = torch.arange(VOCAB_SIZE, device=ids.device)
    for a, b in (("A", "T"), ("C", "G")):
        comp[TOKEN_TO_ID[a]], comp[TOKEN_TO_ID[b]] = TOKEN_TO_ID[b], TOKEN_TO_ID[a]
    lengths = mask.sum(1)                                                    # (B,)
    pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)         # (1, T)
    src = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)                      # reversed index
    out = torch.gather(comp[ids], 1, src)
    return torch.where(mask, out, torch.full_like(out, PAD_ID))


DEFAULT_LAYERS = (-3, -2, -1)
DEFAULT_BINS = 8


def _pool(h: torch.Tensor, mask: torch.Tensor, pool: str, n_bins: int) -> torch.Tensor:
    """(B, T, d) hidden states + (B, T) bool mask -> (B, F) pooled features, fp32.

    pool="mean": masked mean (d).  pool="last": final real token (d).
    pool="rich": [mean | max | std | n_bins positional-bin means] (3d + n_bins*d). The bin
    means split each sequence's *real* length into n_bins equal chunks, which restores the
    "where in the window" signal that global mean pooling discards — the thing promoter /
    splice / regulatory-grammar tasks hinge on.
    """
    h = h.float()
    if pool == "last":
        last = mask.sum(dim=1).clamp(min=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

    m = mask.unsqueeze(-1).float()
    cnt = m.sum(dim=1).clamp(min=1.0)
    mean = (h * m).sum(dim=1) / cnt
    if pool == "mean":
        return mean
    assert pool == "rich", pool

    mx = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    var = ((h - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / cnt
    std = var.clamp(min=0).sqrt()

    B, T, _ = h.shape
    lengths = mask.sum(dim=1).clamp(min=1)                      # (B,)
    pos = torch.arange(T, device=h.device).unsqueeze(0)          # (1, T)
    bin_idx = (pos * n_bins) // lengths.unsqueeze(1)             # (B, T), >= n_bins on pad
    bin_idx = bin_idx.clamp(max=n_bins - 1)
    onehot = torch.nn.functional.one_hot(bin_idx, n_bins).float() * m   # (B, T, n_bins)
    bin_sum = torch.einsum("btk,btd->bkd", onehot, h)
    bin_cnt = onehot.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    bins = (bin_sum / bin_cnt).reshape(B, n_bins * h.shape[-1])
    return torch.cat([mean, mx, std, bins], dim=1)


def embed_dim(model, pool: str = "rich", layers=DEFAULT_LAYERS, n_bins: int = DEFAULT_BINS) -> int:
    d = model.config["d_model"]
    per_layer = d if pool in ("mean", "last") else (3 + n_bins) * d
    return per_layer * (1 if layers is None else len(set(layers)))


@torch.no_grad()
def embed_sequences(model, seqs, ctx: int, device: str, batch_size: int = 8,
                    pool: str = "rich", layers=DEFAULT_LAYERS,
                    n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """Frozen features for the linear probes -> (N, F) float32.

    Default is the "rich" readout: for each of the last three blocks (final-layer features
    in a causal LM are specialised for next-token prediction; the layers just below probe
    better), concatenate mean / max / std pooling plus `n_bins` positional-bin means (see
    `_pool`). With d_model=768 that is 3 x 11 x 768 = 25,344 features.
    `pool="mean", layers=None` reproduces the old HyenaDNA-style mean-of-final-layer readout;
    `pool="last"` takes the final real token instead.

    Pooling is applied inside `model.encode` per captured layer, so the chunked long-context
    path never holds more than one layer's full hidden state at a time.

    Sequences are length-sorted before batching (and restored afterwards) so each batch is
    nearly uniform in length — on a variable-length task like mouse_enhancers (331-4776bp)
    this alone removes most of the padding compute.
    """
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.empty((len(seqs), embed_dim(model, pool, layers, n_bins)), dtype=np.float32)

    for start in range(0, len(order), batch_size):
        idx_batch = order[start:start + batch_size]
        ids, mask = tokenize_batch([seqs[i] for i in idx_batch], ctx, device)
        reduce = lambda h: _pool(h, mask, pool, n_bins)  # noqa: E731
        if layers is None:
            pooled = reduce(model.encode(ids))
        else:
            pooled = torch.cat(model.encode(ids, layers=layers, reduce=reduce), dim=1)
        # `encode(layers=...)` returns captures in ascending block order, so the per-layer
        # feature blocks in `out` are contiguous and sliceable — that is what lets the
        # layer-wise probe (benchmarks/probe.py) fit one probe per layer off a single pass.
        out[idx_batch] = pooled.cpu().numpy()
    return out


@torch.no_grad()
def lm_forward(model, ids: torch.Tensor):
    logits, _ = model(ids)
    return logits


def resolve_dtype(device: str, precision: str) -> torch.dtype:
    """bf16 on CUDA; CPU stays fp32 — bf16 matmuls on CPU are both slow and, for the KDA
    reference scan, numerically poor."""
    if device.startswith("cuda") and precision == "bf16":
        return torch.bfloat16
    return torch.float32


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"total_params": total, "non_embedding_params": total - model.tok_emb.weight.numel(),
            "vocab_size": VOCAB_SIZE}
