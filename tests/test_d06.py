"""D06 per-k-mer fast-weight memory: chunked WY scan == sequential delta rule; chunk carry;
induction recall; key-disambiguation; model builds."""
import torch
import pytest

from models.d05.kmer_memory import kmer_addresses, shift_write_addresses
from models.d06.fastweight_memory import FastWeightKmerMemory
from dataset.tokenizer import encode
from training.config import load_config


def _addr(idx, k):
    r = kmer_addresses(idx, k)
    return r, shift_write_addresses(r)


@pytest.mark.parametrize("scan_chunk", [2, 4, 16])
def test_scan_matches_reference(scan_chunk):
    torch.manual_seed(0)
    mem = FastWeightKmerMemory(k=2, d_in=16, d_k=4, d_v=6, gate_init=0.5, scan_chunk=scan_chunk)
    B, T = 2, 300
    idx = torch.randint(3, 8, (B, T))      # ACGT + N
    addr = _addr(idx, 2)
    h = torch.randn(B, T, 16, requires_grad=True)
    out, tab = mem(h, addr, None)
    h2 = h.detach().clone().requires_grad_(True)
    ref_out, ref_tab = mem.reference(h2, addr)
    assert torch.allclose(out, ref_out, atol=1e-4), (out - ref_out).abs().max()
    assert torch.allclose(tab, ref_tab, atol=1e-4)
    out.sum().backward(); ref_out.sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-3), (h.grad - h2.grad).abs().max()


def test_carry_across_chunks_equals_full():
    torch.manual_seed(1)
    mem = FastWeightKmerMemory(k=2, d_in=16, d_k=4, d_v=4, gate_init=0.0, scan_chunk=4)
    idx = torch.randint(3, 7, (1, 200))
    r, w = _addr(idx, 2)
    h = torch.randn(1, 200, 16)
    with torch.no_grad():
        full, _ = mem(h, (r, w), None)
        a, tab = mem(h[:, :90], (r[:, :90], w[:, :90]), None)
        b, _ = mem(h[:, 90:], (r[:, 90:], w[:, 90:]), tab)
        c, none = mem(h, (r, w), None, return_table=False)
    assert none is None
    assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-5)
    assert torch.allclose(full, c, atol=1e-6)


def test_induction_recall():
    """beta ~ 1, one write: the next occurrence of the k-mer, queried with the same context,
    reads back the stored value (W_r v). Same semantics as d05's offset test."""
    torch.manual_seed(2)
    mem = FastWeightKmerMemory(k=2, d_in=4, d_k=4, d_v=4, gate_init=12.0, scan_chunk=8)
    idx = torch.tensor([encode("ACGTACGT")])
    h = torch.randn(1, 8, 4)
    h[0, 5] = h[0, 2]                        # query context at the 2nd "AC" == write context
    mem.q_proj.weight.data.copy_(mem.k_proj.weight.data)   # q == k for equal h
    with torch.no_grad():
        out, _ = mem(h, _addr(idx, 2), None)
        v2 = mem.v_proj(h)[0, 2]
        expect = mem.read_proj(v2) * torch.sigmoid(mem.read_gate)
    assert torch.allclose(out[0, 5], expect, atol=1e-5), (out[0, 5], expect)
    assert torch.allclose(out[0, 1], torch.zeros(4))


def test_orthogonal_keys_do_not_overwrite():
    """Two writes to the same slot with orthogonal keys: querying with the first key still returns
    the first value -- the property d05's EMA slot lacks."""
    mem = FastWeightKmerMemory(k=1, d_in=2, d_k=2, d_v=2, gate_init=30.0, scan_chunk=8)
    tab = mem.empty_table(1, torch.zeros(1))
    q = torch.tensor([[[1., 0.], [0., 1.], [1., 0.]]])       # events: write k1, write k2, read k1
    k = q.clone()
    v = torch.tensor([[[1., 2.], [3., 4.], [0., 0.]]])
    beta = torch.tensor([[1., 1., 0.]])
    addr = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        out, _ = mem.scan(addr, q, k, v, beta, tab)
    assert torch.allclose(out[0, 2], torch.tensor([1., 2.]), atol=1e-6)


def test_model_builds_and_runs_cpu():
    cfg = load_config("configs/d06_kmer_fastweight.yaml")
    cfg.update(d_model=32, num_layers=2, num_heads=2, kmer_mem_k=3, kmer_fw_dk=4, kmer_fw_dv=4,
               kda_use_fused_kernel=False, gradient_checkpointing=False)
    from models.d06 import build_model
    m = build_model(cfg)
    idx = torch.randint(3, 8, (2, 64))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 64, cfg["vocab_size"]) and torch.isfinite(loss)
    loss.backward()
    assert m.kmer_aux_loss() is None
    assert "read_gate_mean" in m.kmer_metrics()
