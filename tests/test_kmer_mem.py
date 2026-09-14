import torch
import pytest

from models.d05.kmer_memory import KmerMemory, kmer_addresses, shift_write_addresses
from dataset.tokenizer import encode
from training.config import load_config


def test_addresses():
    idx = torch.tensor([encode("ACGTNACGTACG")])
    a = kmer_addresses(idx, 3)
    # first 2 invalid; "ACG" at t=2 -> G*1 + C*4 + A*16 = 6 (latest base least significant)
    assert a[0, :2].tolist() == [-1, -1]
    assert a[0, 2].item() == 6
    assert a[0, 4:7].tolist() == [-1, -1, -1]
    assert a[0, 7].item() == 6 and a[0, 8].item() == 27  # "ACG", "CGT" again after the N gap


@pytest.mark.parametrize("scan_chunk", [4, 64])
def test_scan_matches_reference(scan_chunk):
    torch.manual_seed(0)
    mem = KmerMemory(k=2, d_in=16, mem_dim=8, gate_init=0.0, scan_chunk=scan_chunk).float()
    B, T = 2, 300
    idx = torch.randint(3, 8, (B, T))  # ACGT + N
    r = kmer_addresses(idx, 2)
    addr = (r, shift_write_addresses(r))
    h = torch.randn(B, T, 16, requires_grad=True)
    out, tab = mem(h, addr, None)
    h2 = h.detach().clone().requires_grad_(True)
    ref_out, ref_tab = mem.reference(h2, addr)
    assert torch.allclose(out, ref_out, atol=1e-4), (out - ref_out).abs().max()
    assert torch.allclose(tab, ref_tab, atol=1e-4)
    out.sum().backward(); ref_out.sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-3)


def test_carry_across_chunks_equals_full():
    torch.manual_seed(1)
    mem = KmerMemory(k=2, d_in=16, mem_dim=8, gate_init=0.0, scan_chunk=8).float()
    B, T = 1, 200
    idx = torch.randint(3, 7, (B, T))
    r = kmer_addresses(idx, 2)
    addr = (r, shift_write_addresses(r))
    h = torch.randn(B, T, 16)
    with torch.no_grad():
        full, _ = mem(h, addr, None)
        a, tab = mem(h[:, :90], (addr[0][:, :90], addr[1][:, :90]), None)
        b, _ = mem(h[:, 90:], (addr[0][:, 90:], addr[1][:, 90:]), tab)
    assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-5)


def test_model_builds_and_runs_cpu():
    cfg = load_config("configs/d05_kmer_mem.yaml")
    cfg.update(d_model=32, num_layers=2, num_heads=2, kmer_mem_k=3, kmer_mem_dim=8,
               kda_use_fused_kernel=False, gradient_checkpointing=False)
    from models.d05 import build_model
    m = build_model(cfg)
    idx = torch.randint(3, 8, (2, 64))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 64, cfg["vocab_size"]) and torch.isfinite(loss)
    loss.backward()


def test_write_lands_in_previous_kmer_slot():
    """Induction semantics: the slot of the k-mer ending at t-1 receives h_t, so at the next
    occurrence of that k-mer the read carries information about the base that followed it."""
    torch.manual_seed(2)
    mem = KmerMemory(k=2, d_in=4, mem_dim=4, gate_init=10.0, scan_chunk=8).float()  # gate ~1: overwrite
    idx = torch.tensor([encode("ACGTACGT")])
    r = kmer_addresses(idx, 2)
    h = torch.randn(1, 8, 4)
    with torch.no_grad():
        out, tab = mem(h, (r, shift_write_addresses(r)), None)
        u = mem.write_proj(h)[0]
        # "AC" ends at t=1 and t=5; base t=2 (G) follows it -> slot["AC"] holds u_2 after t=2,
        # so the read at t=5 must be u_2 (not u_1, the old same-address behaviour).
        read5 = mem.read_proj(u[2]) * torch.sigmoid(mem.read_gate)
        assert torch.allclose(out[0, 5], read5, atol=1e-5)
        assert torch.allclose(out[0, 1], torch.zeros(4))          # first occurrence: empty slot


@pytest.mark.parametrize("scan_chunk", [4, 64])
def test_surprise_scan_matches_reference(scan_chunk):
    """The two-pass surprise gate must agree with the sequential loop, values and gradients."""
    torch.manual_seed(0)
    mem = KmerMemory(k=2, d_in=16, mem_dim=8, gate_init=0.0, scan_chunk=scan_chunk,
                     write_gate="surprise").float()
    B, T = 2, 300
    idx = torch.randint(3, 8, (B, T))
    r = kmer_addresses(idx, 2)
    addr = (r, shift_write_addresses(r))
    h = torch.randn(B, T, 16, requires_grad=True)
    out, tab = mem(h, addr, None)
    h2 = h.detach().clone().requires_grad_(True)
    ref_out, ref_tab = mem.reference(h2, addr)
    assert torch.allclose(out, ref_out, atol=1e-4), (out - ref_out).abs().max()
    assert torch.allclose(tab, ref_tab, atol=1e-4)
    out.sum().backward(); ref_out.sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-3)
    assert mem.surprise_w.grad is not None and mem.surprise_w.grad.abs() > 0


def test_surprise_suppresses_redundant_writes():
    """The point of the gate: a slot that already holds u_t must take a much smaller write than a
    slot seeing novel content. Compare gamma on a repeated-content stream vs a novel-content one."""
    torch.manual_seed(3)
    mem = KmerMemory(k=2, d_in=8, mem_dim=8, gate_init=0.0, scan_chunk=16,
                     write_gate="surprise").float()
    idx = torch.tensor([encode("AC" * 64)])          # one k-mer chain, hammered repeatedly
    r = kmer_addresses(idx, 2)
    aw = shift_write_addresses(r)
    h_rep = torch.randn(1, 2, 8).repeat(1, 64, 1)    # identical content every occurrence
    h_new = torch.randn(1, 128, 8)                   # novel content every occurrence
    with torch.no_grad():
        logit_rep = mem.write_gate(h_rep).float().squeeze(-1)
        logit_new = mem.write_gate(h_new).float().squeeze(-1)
        tab = mem.empty_table(1, h_rep)
        g_rep = mem._surprise(logit_rep, mem.write_proj(h_rep).float(), r, aw, tab, None)
        g_new = mem._surprise(logit_new, mem.write_proj(h_new).float(), r, aw, tab, None)
    # Repeated content drives the slot to u, so s -> 0 and gamma decays onto the *baseline* gate
    # sigmoid(W_g h + b_g) with no surprise bonus at all. Novel content keeps paying the bonus.
    base_rep = torch.sigmoid(logit_rep)[0, -16:]
    assert torch.allclose(g_rep[0, -16:], base_rep, atol=2e-2), (g_rep[0, -16:], base_rep)
    assert (g_new[0, 8:] - torch.sigmoid(logit_new)[0, 8:]).mean() > 0.1
    assert g_rep[0, 8:].mean() < g_new[0, 8:].mean()


def test_surprise_model_builds_and_runs_cpu():
    cfg = load_config("configs/d05b_kmer_surprise.yaml")
    cfg.update(d_model=32, num_layers=2, num_heads=2, kmer_mem_k=3, kmer_mem_dim=8,
               kda_use_fused_kernel=False, gradient_checkpointing=False)
    from models.d05 import build_model
    m = build_model(cfg)
    assert all(mem.write_gate_mode == "surprise"
               for blk in m.blocks for mem in blk.attn.memories)
    idx = torch.randint(3, 8, (2, 64))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 64, cfg["vocab_size"]) and torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("scan_chunk,write_gate", [(4, "content"), (64, "content"), (8, "surprise")])
def test_set_associative_scan_matches_reference(scan_chunk, write_gate):
    """S-way sets: the folded-address event stream must agree with the sequential loop."""
    torch.manual_seed(0)
    mem = KmerMemory(k=2, d_in=16, mem_dim=8, gate_init=0.0, scan_chunk=scan_chunk,
                     write_gate=write_gate, slots_per_kmer=3).float()
    B, T = 2, 200
    idx = torch.randint(3, 8, (B, T))
    r = kmer_addresses(idx, 2)
    addr = (r, shift_write_addresses(r))
    h = torch.randn(B, T, 16, requires_grad=True)
    out, tab = mem(h, addr, None)
    h2 = h.detach().clone().requires_grad_(True)
    ref_out, ref_tab = mem.reference(h2, addr)
    assert torch.allclose(out, ref_out, atol=1e-4), (out - ref_out).abs().max()
    assert torch.allclose(tab, ref_tab, atol=1e-4)
    out.sum().backward(); ref_out.sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-3)
    assert mem.router.weight.grad is not None


def test_single_slot_is_unchanged_by_the_set_code():
    """S=1 must stay byte-identical to the pre-set-associative memory: no router, no aux loss."""
    torch.manual_seed(0)
    mem = KmerMemory(k=2, d_in=16, mem_dim=8, gate_init=0.0, scan_chunk=16).float()
    assert not hasattr(mem, "router") and mem.n_slots == 4 ** 2
    idx = torch.randint(3, 8, (1, 120))
    r = kmer_addresses(idx, 2)
    h = torch.randn(1, 120, 16)
    with torch.no_grad():
        out, _ = mem(h, (r, shift_write_addresses(r)), None)
        ref, _ = mem.reference(h, (r, shift_write_addresses(r)))
    assert mem.aux_loss is None and torch.allclose(out, ref, atol=1e-5)


def test_sets_disambiguate_colliding_contexts():
    """The point of S>1: two different continuations written to the SAME address under different
    contexts must both survive. A single slot can only ever hold the more recent one."""
    d = 8
    T = 3
    a = torch.zeros(1, T, dtype=torch.long)         # every position hits the same k-mer address
    gamma = torch.ones(1, T)                        # full overwrite: the hardest case for S=1
    u = torch.zeros(1, T, d)
    v_a, v_b = torch.randn(d), torch.randn(d)
    u[0, 0], u[0, 1] = v_a, v_b                     # t=0 writes under context A, t=1 under B
    got = {}
    for S in (1, 4):
        mem = KmerMemory(k=2, d_in=4, mem_dim=d, gate_init=0.0, scan_chunk=8,
                         slots_per_kmer=S).float()
        p = None
        if S > 1:                                   # context A -> entry 0, context B -> entry 1
            p = torch.zeros(1, T, S)
            p[0, 0, 0], p[0, 1, 1], p[0, 2, 0] = 1.0, 1.0, 1.0
        with torch.no_grad():
            reads, _, _ = mem._scan_rw(a, a, gamma, u, mem.empty_table(1, u), p)
            reads = reads.squeeze(2) if p is None else torch.einsum("bts,btsd->btd", p, reads)
        got[S] = reads[0, 2]                        # read at t=2, which is context A again
    # S=1: the context-B write at t=1 clobbered the context-A write at t=0.
    assert torch.allclose(got[1], v_b, atol=1e-5)
    assert (got[1] - v_a).norm() > 1e-2
    # S=4: the two contexts occupied different entries, so context A's value survived.
    assert torch.allclose(got[4], v_a, atol=1e-5), (got[4], v_a)


def test_load_balance_loss_penalises_collapse():
    mem = KmerMemory(k=2, d_in=8, mem_dim=8, gate_init=0.0, scan_chunk=8, slots_per_kmer=4).float()
    uniform = torch.full((2, 10, 4), 0.25)
    collapsed = torch.zeros(2, 10, 4); collapsed[..., 0] = 1.0
    assert torch.isclose(mem.load_balance_loss(uniform), torch.tensor(1.0))
    assert mem.load_balance_loss(collapsed).item() == pytest.approx(4.0)


def test_set_associative_model_builds_and_runs_cpu():
    cfg = load_config("configs/d05c_kmer_setassoc.yaml")
    cfg.update(d_model=32, num_layers=2, num_heads=2, kmer_mem_k=3, kmer_mem_dim=8,
               kda_use_fused_kernel=False, gradient_checkpointing=False)
    from models.d05 import build_model
    m = build_model(cfg)
    S = cfg["kmer_mem_slots_per_kmer"]
    assert all(mem.S == S for blk in m.blocks for mem in blk.attn.memories)
    idx = torch.randint(3, 8, (2, 64))
    logits, loss = m(idx, idx)
    assert torch.isfinite(loss) and m.kmer_aux_loss() is not None
    loss.backward()
    assert m.blocks[0].attn.memories[0].router.weight.grad is not None
