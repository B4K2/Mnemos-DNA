"""D01-rope: the same pure-KDA operator as models/d01, with NTK-scaled RoPE instead of NoPE.

The positional scheme is entirely config-driven — GPT builds it via
models/shared/pos.build_pos_scheme from `pos_type`, and KDAAttention applies it to q/k — so the
builder is identical to D01's and the NoPE-vs-NTK-RoPE axis stays a pure config change (CLAUDE.md
§5). This module exists only because scripts/train.py resolves the builder by model_id.
"""
from models.d01 import build_model

__all__ = ["build_model"]
