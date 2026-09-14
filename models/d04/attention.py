"""HOLA — Hippocampal Linear Attention (arXiv 2607.02303) on the KDA backbone.

Keeps the KDA recurrence of models/d01 unchanged as the "neocortex" (lossy compressive state)
and adds a bounded exact-KV cache ("hippocampus") per layer:

  what to store   the top-w tokens of the causal history ranked by the delta rule's own write
                  magnitude m_t = beta_t * ||e_t|| (Eq. 6 in the paper) — the surprise signal
                  the recurrence already computes. Parameter-free; no learned eviction module.
  how to read     a sharpened softmax over the visible set V_t = {top-w cache} ∪ {current
                  chunk, causal} ∪ {null sink}, with Qwen3-style RMSNorm-gamma applied to the
                  *cache-path* q/k only (paper §3.4). The q/k feeding the recurrent state stay
                  unit-L2 — a sqrt(d) norm inside the state update would make the delta-rule
                  operator I - beta*k*k^T unstable, which is why the norm is decoupled.
  read-out        o_t = S_t^T q_t + lambda_h * cache_read(q_t), lambda_h = sigmoid(per-head
                  scalar, init -4) so training starts as plain KDA and opts into the cache.

Surprise without a custom kernel: the fused path never materializes e_t, but running the same
chunk_kda kernel a second time with q := k gives o_t^{q=k} = S_t^T k_t, and since S_t already
contains this token's write and ||k_t|| = 1,
    S_t^T k_t = pred_t + beta_t * e_t = v_t - (1 - beta_t) * e_t
so ||e_t|| = ||v_t - o_t^{q=k}|| / (1 - beta_t). The second call runs under no_grad — the
eviction score is a score, not a differentiable path (matching the paper, where selection is
non-differentiable top-w).

Complexity stays O(T): the cache read costs O(T * (w + C)) with chunk size C, and the cache
itself is O(w) state per layer. Training and inference use the same cache semantics — the
top-w set over history is order-independent because scores are fixed at write time.

Memory: each chunk's cache read is checkpointed (nested inside the block-level checkpoint, so
only the read *output* survives per chunk, not its softmax/logits) — without this the fp32
probs alone are ~T*(w+C+1)*H*4 bytes per layer during backward recompute, which OOMs the 5060
at 262144. Validated fwd+bwd at full length on 2026-08-13.
"""
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from models.d01.attention import KDAAttention


class HOLAAttention(KDAAttention):
    def __init__(self, config: dict):
        super().__init__(config)
        H, Dh = self.num_heads, self.head_dim
        self.cache_window = config.get("hola_window", 64)   # w
        self.cache_chunk = config.get("hola_chunk", 256)    # C

        # Cache-path-only sharpening (paper §3.4): RMSNorm with learned per-channel gain takes
        # the unit-L2 q/k back to norm ~sqrt(d), so cache logits reach ~sqrt(d)*cos instead of
        # ~cos and the read is near-argmax retrieval rather than a soft average. tau fixed at 1.
        self.cache_q_norm = nn.RMSNorm(Dh)
        self.cache_k_norm = nn.RMSNorm(Dh)

        # Per-head read gate lambda = sigmoid(.), init -4 (~0.018): start as pure KDA. Per-head
        # null-sink logit lets a query attend to nothing. 2H scalars total, matching the paper's
        # parameter accounting; excluded from weight decay like A_log/dt_bias (decay would drag
        # the gate back toward "no cache").
        self.cache_gate = nn.Parameter(torch.full((H,), float(config.get("hola_gate_init", -4.0))))
        self.sink_logit = nn.Parameter(torch.zeros(H))
        self.cache_gate._no_weight_decay = True
        self.sink_logit._no_weight_decay = True

    # --- state path: KDA output + surprise score ----------------------------------------------

    def _run_kernel(self, q, k, v, g, beta, initial_state, output_final_state):
        """chunk_kda on pre-computed (B, H, T, ·) tensors — same dtype/layout contract as
        KDAAttention._kernel, but without recomputing the projections."""
        from fla.ops.kda import chunk_kda

        kernel_dtype = torch.bfloat16 if q.is_cuda else torch.float32
        out, final_state = chunk_kda(
            q.transpose(1, 2).to(kernel_dtype),
            k.transpose(1, 2).to(kernel_dtype),
            v.transpose(1, 2).to(kernel_dtype),
            g.transpose(1, 2).to(kernel_dtype),
            beta.transpose(1, 2).to(kernel_dtype),
            scale=1.0,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
        )
        return out.transpose(1, 2), final_state  # (B, H, T, Dh)

    def _surprise_fused(self, k, v, g, beta, initial_state):
        """m_t = beta_t * ||e_t|| via the q := k identity in the module docstring."""
        with torch.no_grad():
            o_kk, _ = self._run_kernel(k, k, v, g, beta, initial_state, False)
            e_norm = (v - o_kk.to(v.dtype)).float().norm(dim=-1)
            e_norm = e_norm / (1.0 - beta.float()).clamp_min(1e-3)
            return beta.float() * e_norm  # (B, H, T)

    def _state_and_surprise_reference(self, q, k, v, g, beta, initial_state):
        """Sequential scan (CPU / correctness path) returning output, surprise, final state."""
        alpha = torch.exp(g).to(q.dtype)
        batch, H, seq_len, Dh = q.shape
        state = initial_state if initial_state is not None else q.new_zeros(batch, H, Dh, Dh)
        outputs, scores = [], []
        for t in range(seq_len):
            q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
            state = alpha[:, :, t].unsqueeze(-1) * state
            pred_v = torch.einsum("bhd,bhdv->bhv", k_t, state)
            e_t = v_t - pred_v
            state = state + torch.einsum("bhd,bhv->bhdv", k_t, beta[:, :, t].unsqueeze(-1) * e_t)
            outputs.append(torch.einsum("bhd,bhdv->bhv", q_t, state))
            scores.append(beta[:, :, t].float() * e_t.detach().float().norm(dim=-1))
        return torch.stack(outputs, dim=2), torch.stack(scores, dim=2), state

    # --- cache path ---------------------------------------------------------------------------

    def _empty_cache(self, like):
        B, H, _, Dh = like.shape
        return (like.new_zeros(B, H, 0, Dh),          # keys (unit-L2, pre cache-norm)
                like.new_zeros(B, H, 0, Dh),          # values
                like.new_zeros(B, H, 0, dtype=torch.float32))  # surprise scores

    def _cache_read_update(self, q, k, v, score, cache):
        """Chunked cache read + top-w merge. For each chunk: read against the cache as selected
        from *strictly previous* history, plus causal intra-chunk tokens, plus the null sink —
        then fold the chunk's (k, v, score) into the top-w set.

        Cached entries are detached at write time: the cache is test-time state (the paper's
        framing), not a differentiable path back into earlier chunks; gradients flow through
        the current chunk's q/k/v and through the RMSNorm gains applied at read.
        """
        B, H, T, Dh = q.shape
        w, C = self.cache_window, self.cache_chunk
        scale = 1.0 / math.sqrt(Dh)
        ck, cv, cs = cache
        sink = self.sink_logit.float().view(1, H, 1, 1)

        def read(q_raw, k_raw, v_raw, mem_k, mem_v):
            n = q_raw.shape[2]
            qc = self.cache_q_norm(q_raw)
            logits_cur = torch.einsum(
                "bhid,bhjd->bhij", qc, self.cache_k_norm(k_raw)).float() * scale
            causal = torch.ones(n, n, dtype=torch.bool, device=q_raw.device).tril_()
            logits_cur = logits_cur.masked_fill(~causal, float("-inf"))

            if mem_k.shape[2] > 0:
                logits_mem = torch.einsum(
                    "bhid,bhjd->bhij", qc, self.cache_k_norm(mem_k)).float() * scale
                logits = torch.cat([logits_mem, logits_cur, sink.expand(B, H, n, 1)], dim=-1)
                values = torch.cat([mem_v, v_raw], dim=2)
            else:
                logits = torch.cat([logits_cur, sink.expand(B, H, n, 1)], dim=-1)
                values = v_raw
            attn = torch.softmax(logits, dim=-1).to(v_raw.dtype)[..., :-1]  # sink's value is 0
            return torch.einsum("bhij,bhjd->bhid", attn, values)

        out = torch.empty_like(v)
        for start in range(0, T, C):
            stop = min(start + C, T)
            chunk_args = (q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], ck, cv)
            if torch.is_grad_enabled():
                # Nested checkpoint: only the read's output survives the layer recompute; the
                # per-chunk softmax/logits (the OOM term at 262144) are rebuilt on backward.
                out[:, :, start:stop] = torch.utils.checkpoint.checkpoint(
                    read, *chunk_args, use_reentrant=False)
            else:
                out[:, :, start:stop] = read(*chunk_args)

            ck = torch.cat([ck, k[:, :, start:stop].detach()], dim=2)
            cv = torch.cat([cv, v[:, :, start:stop].detach()], dim=2)
            cs = torch.cat([cs, score[:, :, start:stop]], dim=2)
            if ck.shape[2] > w:
                _, idx = cs.topk(w, dim=-1)
                gather = idx.unsqueeze(-1).expand(-1, -1, -1, Dh)
                ck, cv, cs = ck.gather(2, gather), cv.gather(2, gather), cs.gather(2, idx)
        return out, (ck, cv, cs)

    # --- forward ------------------------------------------------------------------------------

    def _mix(self, o_state, o_cache):
        lam = torch.sigmoid(self.cache_gate.float()).view(1, self.num_heads, 1, 1)
        return o_state + (lam * o_cache.float()).to(o_state.dtype)

    def _forward_full(self, x, pos_scheme, state, pos_offset: int = 0, drop_lead: int = 0,
                      output_final_state: bool = False):
        """Shared body: state = None or (S, cache) tuple carried across chunks."""
        q, k, v, g, beta = self._qkv_alpha_beta(x, pos_scheme, pos_offset, drop_lead)
        S, cache = state if state is not None else (None, self._empty_cache(q))

        if self.use_fused_kernel and x.is_cuda:
            o_state, S_new = self._run_kernel(q, k, v, g, beta, S, output_final_state)
            score = self._surprise_fused(k, v, g, beta, S)
        else:
            o_state, score, S_new = self._state_and_surprise_reference(q, k, v, g, beta, S)

        o_cache, cache = self._cache_read_update(q, k, v.to(o_state.dtype), score, cache)
        out = self._mix(o_state, o_cache)
        B, H, T, Dh = out.shape
        return out.transpose(1, 2).reshape(B, T, H * Dh), (S_new, cache)

    def forward(self, x, pos_scheme=None):
        out, _ = self._forward_full(x, pos_scheme, None)
        return self.o_proj(out)

    def forward_chunked(self, x_ext, pos_scheme, state, pos_offset: int, drop_lead: int):
        """Block-level chunked path (models/shared/gpt.py). The carried state is the
        (recurrent S, exact-KV cache) pair — both survive chunk boundaries, so a needle cached
        in chunk 0 is still readable at 1M tokens. Inference only (caller has grad disabled)."""
        out, state = self._forward_full(
            x_ext, pos_scheme, state, pos_offset, drop_lead, output_final_state=True)
        return self.o_proj(out), state
