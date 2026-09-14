"""D01-ropefix: pure KDA + NTK-scaled RoPE with the corrected forget gate (configs/d01_ropefix.yaml).

Same builder as models/d01 — the gate fix lives in models/d01/attention.py and so is shared by
every arm; the positional scheme is config-driven. This module exists only because
scripts/train.py resolves the builder by model_id.
"""
from models.d01 import build_model

__all__ = ["build_model"]
