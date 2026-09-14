"""D05-rope: the same KDA + k-mer-memory operator as models/d05, with NTK-scaled RoPE.

Identical situation to models/d01_rope: the positional scheme is entirely config-driven
(models/shared/gpt builds it from `pos_type`; KDAAttention applies it to q/k), and the k-mer
memory is *content*-addressed — its address depends only on the nucleotides, never on position —
so RoPE and the memory are orthogonal and this stays a one-axis change. This module exists only
because scripts/train.py resolves the builder by model_id.
"""
from models.d05 import build_model

__all__ = ["build_model"]
