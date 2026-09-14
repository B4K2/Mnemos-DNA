"""D04/HOLA correctness gates (models/d04/attention.py), CPU reference path unless CUDA is
free. Three properties:

  1. gate closed => HOLA output == plain KDA output on the same weights (the cache is a pure
     additive overlay; lambda -> 0 must recover d01 exactly).
  2. the q := k surprise identity: beta*||v - S_t^T k|| / (1-beta) equals the directly-computed
     beta*||e|| from the sequential scan.
  3. forward_chunked over pieces == forward over the whole sequence — the carried
     (state, cache) tuple must make the block-level chunked path exact.
"""
import torch

from models.d01.attention import KDAAttention
from models.d04.attention import HOLAAttention

CONFIG = {"d_model": 64, "num_heads": 4, "kda_conv_kernel": 4,
          "hola_window": 8, "hola_chunk": 16}


def _pair(seed=0):
    torch.manual_seed(seed)
    hola = HOLAAttention(CONFIG)
    kda = KDAAttention(CONFIG)
    kda.load_state_dict({k: v for k, v in hola.state_dict().items()
                         if k in kda.state_dict()})
    return hola, kda


def test_gate_closed_matches_kda():
    hola, kda = _pair()
    with torch.no_grad():
        hola.cache_gate.fill_(-30.0)  # sigmoid ~ 0: cache contribution vanishes
    x = torch.randn(2, 48, 64)
    diff = (hola(x) - kda(x)).abs().max().item()
    print(f"gate-closed max abs diff: {diff:.2e}")
    assert diff < 1e-5, diff


def test_surprise_identity():
    hola, _ = _pair()
    x = torch.randn(2, 48, 64)
    q, k, v, g, beta = hola._qkv_alpha_beta(x, None)
    _, score_direct, _ = hola._state_and_surprise_reference(q, k, v, g, beta, None)
    # the fused-path formula, evaluated with the reference scan standing in for the kernel
    o_kk, _, _ = hola._state_and_surprise_reference(k, k, v, g, beta, None)
    e_norm = (v - o_kk).float().norm(dim=-1) / (1.0 - beta.float()).clamp_min(1e-3)
    score_identity = beta.float() * e_norm
    diff = (score_direct - score_identity).abs().max().item()
    print(f"surprise identity max abs diff: {diff:.2e}")
    assert diff < 1e-4, diff


def test_chunked_matches_full():
    hola, _ = _pair()
    x = torch.randn(2, 64, 64)
    with torch.no_grad():
        full = hola(x)
        lead = hola.left_context
        state, pieces = None, []
        for start in range(0, 64, 16):
            ctx_start = max(0, start - lead)
            out, state = hola.forward_chunked(
                x[:, ctx_start:start + 16], None, state, start, start - ctx_start)
            pieces.append(out)
    diff = (full - torch.cat(pieces, dim=1)).abs().max().item()
    print(f"chunked-vs-full max abs diff: {diff:.2e}")
    assert diff < 1e-4, diff


if __name__ == "__main__":
    test_gate_closed_matches_kda()
    test_surprise_identity()
    test_chunked_matches_full()
    print("PASS")
