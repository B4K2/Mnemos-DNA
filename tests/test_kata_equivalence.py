"""Verify KATA's chunked-state fast path produces the same recurrence as the plain per-token
reference scan, that results are invariant to kata_chunk_size, and that the block-level
forward_chunked state-carry path matches a whole-sequence pass. Correctness gate before
trusting D03 for a real run — same role as tests/test_kda_equivalence.py for D01."""
import torch

from models.d03.attention import KATAAttention

CONFIG = {"d_model": 64, "num_heads": 4, "kda_conv_kernel": 4,
          "kata_blocks": 4, "kata_chunk_size": 16}


def _attn(seed=0):
    torch.manual_seed(seed)
    return KATAAttention(CONFIG)


def test_chunked_scan_matches_reference():
    attn = _attn()
    x = torch.randn(2, 100, 64, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()

    out_ref = attn.o_proj(attn._forward_reference(x, pos_scheme=None))
    q, k, v = attn._qkv(x2, None)
    out_fast, _ = attn._scan(q, k, v, None)
    out_fast = attn.o_proj(out_fast.to(x2.dtype))

    assert torch.allclose(out_ref, out_fast, atol=1e-4, rtol=1e-4), \
        (out_ref - out_fast).abs().max().item()

    out_ref.sum().backward()
    out_fast.sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-4, rtol=1e-4), \
        (x.grad - x2.grad).abs().max().item()


def test_invariant_to_chunk_size():
    attn = _attn()
    x = torch.randn(1, 130, 64)
    q, k, v = attn._qkv(x, None)
    out16, _ = attn._scan(q, k, v, None)
    attn.chunk_size = 64
    out64, _ = attn._scan(q, k, v, None)
    attn.chunk_size = 1000  # single chunk
    out_full, _ = attn._scan(q, k, v, None)
    assert torch.allclose(out16, out64, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out16, out_full, atol=1e-4, rtol=1e-4)


def test_forward_chunked_state_carry_matches_whole_sequence():
    attn = _attn()
    x = torch.randn(1, 96, 64)
    with torch.no_grad():
        whole = attn(x)  # CPU path -> reference scan

        lead = attn.left_context
        state, pieces = None, []
        for start in range(0, 96, 32):
            stop = start + 32
            ctx_start = max(0, start - lead)
            piece, state = attn.forward_chunked(
                x[:, ctx_start:stop], None, state,
                pos_offset=start, drop_lead=start - ctx_start)
            pieces.append(piece)
        chunked = torch.cat(pieces, dim=1)

    assert torch.allclose(whole, chunked, atol=1e-4, rtol=1e-4), \
        (whole - chunked).abs().max().item()


if __name__ == "__main__":
    test_chunked_scan_matches_reference()
    test_invariant_to_chunk_size()
    test_forward_chunked_state_carry_matches_whole_sequence()
    print("PASS")
