"""D03: pure KATA (kernelized linear attention, arXiv 2607.17419) — every layer is
KATAAttention, no hybrid quadratic layers.

This arm replaced the old D03 (KDA + full-softmax hybrid). The recall problem the matrix
keeps hitting (§6.4 needle probe at chance for pure KDA) has two candidate fixes: spend
quadratic budget back (old D03 / current D02's local layers), or change the linear-attention
kernel itself so stored associations stop interfering (KATA). D03 now tests the second.

Comparisons this arm supports:
  d01_rope -> d03 : same backbone/pos/config, mixer KDA -> KATA (does the kernel fix recall?)
  d02      -> d03 : local-attention patch vs KATA kernel, both O(T) (which recall fix wins?)
"""
from models.d03.attention import KATAAttention
from models.shared.gpt import GPT


def build_model(config: dict) -> GPT:
    return GPT(KATAAttention, config)
