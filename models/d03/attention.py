"""KATA — Kernelized Linear Attention with blocked rank-one PSD features."""
import torch
import torch.nn as nn

from models.d01.attention import ShortCausalConv
from models.shared import lowmem


class KATAAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.num_blocks = config.get("kata_blocks", 4)  # g
        assert self.head_dim % self.num_blocks == 0, "head_dim must divide into kata_blocks"
        self.block_dim = self.head_dim // self.num_blocks  # m
        self.feat_dim = self.num_blocks * self.block_dim**2  # F = g * m^2

        self.chunk_size = config.get("kata_chunk_size", 512)
        self.eps = config.get("kata_eps", 1e-6)

        # RMSNorm with learned per-channel gains on q/k instead of plain per-block l2
        # normalization (paper App. A.1: gains + inverse-temperature absorb into the lifted
        # bilinear form; their Table 3/4 ablation shows plain l2-norm is strictly worse).
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        conv_kernel = config.get("kda_conv_kernel", 4)
        self.left_context = conv_kernel - 1  # read by the block-level chunked path in gpt.py
        self.q_conv = ShortCausalConv(d_model, conv_kernel)
        self.k_conv = ShortCausalConv(d_model, conv_kernel)
        self.v_conv = ShortCausalConv(d_model, conv_kernel)

        # Group rotate-half pairs (i, i + head_dim/2) into the same block: block a takes
        # pairs a*m/2 .. (a+1)*m/2. A no-op relabeling under NoPE.
        assert self.block_dim % 2 == 0, "block_dim must be even to hold whole RoPE pairs"
        half, pairs = self.head_dim // 2, self.block_dim // 2
        perm = torch.cat([
            torch.cat([torch.arange(a * pairs, (a + 1) * pairs),
                       torch.arange(a * pairs, (a + 1) * pairs) + half])
            for a in range(self.num_blocks)
        ])
        self.register_buffer("block_perm", perm, persistent=False)

    # --- feature map -------------------------------------------------------------------------

    def _features(self, u):
        """psi: (B, H, T, Dh) -> (B, H, T, g*m^2), fp32.

        Inputs are RMSNorm-ed in _qkv (learned gains set block magnitude, acting as an
        inverse temperature); the outer product is the rank-one PSD lift.
        Kernel = sum_a (u_a^T v_a)^2 >= 0."""
        B, H, T, _ = u.shape
        g, m = self.num_blocks, self.block_dim
        u = u[..., self.block_perm].float().view(B, H, T, g, m)
        outer = u.unsqueeze(-1) * u.unsqueeze(-2)  # (B, H, T, g, m, m)
        return outer.reshape(B, H, T, g * m * m)

    def _qkv(self, x, pos_scheme, pos_offset: int = 0, drop_lead: int = 0):
        """Same conv/left-context contract as KDAAttention._qkv_alpha_beta."""
        H, Dh = self.num_heads, self.head_dim

        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))
        if drop_lead:
            q, k, v = q[:, drop_lead:], k[:, drop_lead:], v[:, drop_lead:]

        batch, seq_len, _ = q.shape

        def split_heads(t):
            return t.view(batch, seq_len, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q, k = self.q_norm(q), self.k_norm(k)
        if pos_scheme is not None:
            q, k = pos_scheme(q, k, positions=pos_offset)
        return q, k, v

    # --- reference scan (test ground truth / CPU) ----------------------------------------------

    def _forward_reference(self, x, pos_scheme):
        q, k, v = self._qkv(x, pos_scheme)
        phi_q, phi_k = self._features(q), self._features(k)
        v = v.float()
        B, H, T, Dv = v.shape

        S = v.new_zeros(B, H, self.feat_dim, Dv)
        Z = v.new_zeros(B, H, self.feat_dim)
        outs = []
        for t in range(T):
            S = S + torch.einsum("bhf,bhv->bhfv", phi_k[:, :, t], v[:, :, t])
            Z = Z + phi_k[:, :, t]
            num = torch.einsum("bhf,bhfv->bhv", phi_q[:, :, t], S)
            den = torch.einsum("bhf,bhf->bh", phi_q[:, :, t], Z)
            outs.append(num / (den.unsqueeze(-1) + self.eps))
        out = torch.stack(outs, dim=2)  # (B, H, T, Dv)
        return out.transpose(1, 2).reshape(B, T, H * Dv).to(x.dtype)

    # --- chunked-state form (the fast path, paper eqs. 18-20) ---------------------------------

    def _scan(self, q, k, v, state):
        """Run the chunked recurrence over (q, k, v), resuming from `state` (S, Z) or None.
        Returns (out (B, T, H*Dv) in fp32, new state). State is always fp32."""
        B, H, T, Dv = v.shape
        C = self.chunk_size
        v = v.float()

        if state is None:
            S = v.new_zeros(B, H, self.feat_dim, Dv)
            Z = v.new_zeros(B, H, self.feat_dim)
        else:
            S, Z = state

        out = v.new_empty(B, H, T, Dv)
        for s in range(0, T, C):
            e = min(s + C, T)
            pq = self._features(q[:, :, s:e])   # (B, H, c, F)
            pk = self._features(k[:, :, s:e])
            vc = v[:, :, s:e]
            c = e - s

            P = pq @ pk.transpose(-1, -2)  # (B, H, c, c), entries >= 0
            P = P.tril()                   # causal (inclusive of self)
            num = pq @ S + P @ vc
            den = (pq * Z.unsqueeze(2)).sum(-1) + P.sum(-1)  # (B, H, c)
            out[:, :, s:e] = num / (den.unsqueeze(-1) + self.eps)

            S = S + pk.transpose(-1, -2) @ vc
            Z = Z + pk.sum(2)

        return out.transpose(1, 2).reshape(B, T, H * Dv), (S, Z)

    def forward_chunked(self, x_ext, pos_scheme, state, pos_offset: int, drop_lead: int):
        """Block-level chunked path (same contract as KDAAttention.forward_chunked): `x_ext`
        carries `drop_lead` tokens of conv left context, state is carried across chunks."""
        q, k, v = self._qkv(x_ext, pos_scheme, pos_offset, drop_lead)
        out, state = self._scan(q, k, v, state)
        return self.o_proj(out.to(x_ext.dtype)), state

    def forward(self, x, pos_scheme=None):
        if not x.is_cuda:
            return self.o_proj(self._forward_reference(x, pos_scheme))

        chunk = lowmem.chunk_size()
        if chunk is not None and x.shape[1] > chunk:
            # Outer lowmem chunking bounds q/k/v materialization; _scan's own chunk_size
            # bounds the feature/P working set within each piece.
            B, T, _ = x.shape
            lead = self.left_context
            out_full, state = None, None
            for start in range(0, T, chunk):
                stop = min(start + chunk, T)
                ctx_start = max(0, start - lead)
                q, k, v = self._qkv(
                    x[:, ctx_start:stop], pos_scheme,
                    pos_offset=start, drop_lead=start - ctx_start)
                piece, state = self._scan(q, k, v, state)
                piece = piece.to(x.dtype)
                if out_full is None:
                    out_full = piece.new_empty(B, T, piece.shape[-1])
                out_full[:, start:stop] = piece
            return self.o_proj(out_full)

        q, k, v = self._qkv(x, pos_scheme)
        out, _ = self._scan(q, k, v, None)
        return self.o_proj(out.to(x.dtype))
