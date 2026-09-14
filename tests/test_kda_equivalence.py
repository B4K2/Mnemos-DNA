"""Verify the fused chunk_kda kernel produces the same recurrence as the plain sequential
reference scan, on the same weights and inputs. This is the correctness gate before trusting
the fused path for a real training run -- see models/d01/attention.py docstring."""
import torch

from models.d01.attention import KDAAttention


def test_fused_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    config = {"d_model": 64, "num_heads": 4, "kda_conv_kernel": 4}

    attn_ref = KDAAttention(config).to("cuda")
    attn_fused = KDAAttention(config).to("cuda")
    attn_fused.load_state_dict(attn_ref.state_dict())

    attn_ref = attn_ref.to(torch.bfloat16)
    attn_fused = attn_fused.to(torch.bfloat16)

    x_ref = torch.randn(2, 96, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_()

    out_ref = attn_ref._forward_reference(x_ref, pos_scheme=None)
    out_fused = attn_fused._forward_fused(x_fused, pos_scheme=None)

    max_abs_diff = (out_ref.float() - out_fused.float()).abs().max().item()
    print(f"forward max abs diff: {max_abs_diff:.6f}")
    assert torch.allclose(out_ref.float(), out_fused.float(), atol=5e-2, rtol=5e-2), max_abs_diff

    out_ref.float().sum().backward()
    out_fused.float().sum().backward()
    grad_diff = (x_ref.grad.float() - x_fused.grad.float()).abs().max().item()
    print(f"grad max abs diff: {grad_diff:.6f}")
    assert torch.allclose(x_ref.grad.float(), x_fused.grad.float(), atol=5e-2, rtol=5e-2), grad_diff

    print("PASS")


if __name__ == "__main__":
    test_fused_matches_reference_forward_and_backward()
