"""The KATA kernel must be a function of *relative* position under RoPE.

Regression test for the cross-block rotate-half bug: the shared RoPE rotates dim pairs
(i, i + head_dim/2), so contiguous feature blocks split every pair across two blocks and the
per-block-normalized squared kernel picked up absolute-position dependence (~5x swing at a
fixed offset). KATAAttention.block_perm groups whole pairs into each block; with it, the
kernel at a fixed offset must not move as the absolute position slides.
"""
import torch
import torch.nn.functional as F

from models.d03.attention import KATAAttention
from models.shared.pos import NTKScaledRoPE


def _kernel(attn, q, k):
    pq, pk = attn._features(q), attn._features(k)
    return (pq * pk).sum(-1)


def test_kata_kernel_is_relative_under_rope():
    torch.manual_seed(0)
    config = {"d_model": 128, "num_heads": 4, "kata_blocks": 4}
    attn = KATAAttention(config)
    rope = NTKScaledRoPE(dim=attn.head_dim, base=10000.0, trained_length=65536)

    q0 = torch.randn(1, attn.num_heads, 1, attn.head_dim)
    k0 = torch.randn(1, attn.num_heads, 1, attn.head_dim)
    delta = 100

    vals = []
    for p in [0, 1000, 5000, 20000, 60000]:
        qr, _ = rope(q0, q0, positions=p + delta)
        kr, _ = rope(k0, k0, positions=p)
        vals.append(_kernel(attn, qr, kr))
    vals = torch.stack(vals)
    spread = (vals.max(dim=0).values - vals.min(dim=0).values).max().item()
    # Relative threshold: _features no longer unit-normalizes blocks (RMSNorm on q/k moved
    # into _qkv), so kernel magnitude — and fp32 RoPE rounding — scales with ||q_a||^2||k_a||^2.
    scale = vals.abs().max().item()
    assert spread < 1e-4 * scale, \
        f"kernel varies with absolute position (spread={spread}, scale={scale})"


def test_block_perm_preserves_block_norms_under_rope():
    torch.manual_seed(1)
    config = {"d_model": 128, "num_heads": 4, "kata_blocks": 4}
    attn = KATAAttention(config)
    rope = NTKScaledRoPE(dim=attn.head_dim, base=10000.0, trained_length=65536)

    u = torch.randn(1, attn.num_heads, 1, attn.head_dim)
    g, m = attn.num_blocks, attn.block_dim

    def block_norms(x):
        return x[..., attn.block_perm].view(1, attn.num_heads, 1, g, m).norm(dim=-1)

    ur, _ = rope(u, u, positions=12345)
    assert torch.allclose(block_norms(u), block_norms(ur), atol=1e-5)
