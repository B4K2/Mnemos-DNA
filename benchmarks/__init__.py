"""Benchmark registry.

`length_swept=True` means the benchmark is re-run at every context length in --lengths; the
fixed-length downstream suites (Genomic Benchmarks, NT, GUE) run once at each task's own native
sequence length, because padding a 200bp promoter to 1M tokens measures padding, not genomics.
"""
from dataclasses import dataclass
from typing import Callable

from benchmarks import (genomic_benchmarks, gue, lm_eval, lrb, needle, nt_bench,
                        scaling, species)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fn: Callable
    length_swept: bool
    description: str


REGISTRY: list[BenchmarkSpec] = [
    BenchmarkSpec("lm", lm_eval.run, True,
                  "next-nucleotide CE/perplexity/bits-per-nucleotide/accuracy on held-out hg38"),
    BenchmarkSpec("needle", needle.run, True,
                  "long-range exact-recall probe (key->value associative recall, chance 0.25)"),
    BenchmarkSpec("scaling", scaling.run, True,
                  "peak VRAM/RAM and prefill/decode/train throughput vs context length"),
    BenchmarkSpec("lrb", lrb.run, True,
                  "Genomics Long-Range Benchmark: VEP eQTL/ClinVar, chromatin, enhancer, promoter"),
    BenchmarkSpec("species", species.run, True,
                  "multi-species window classification from local reference genomes"),
    BenchmarkSpec("genomic_benchmarks", genomic_benchmarks.run, False,
                  "Genomic Benchmarks, 8 tasks (HyenaDNA Table 4.1 row-for-row)"),
    BenchmarkSpec("nt", nt_bench.run, False,
                  "Nucleotide Transformer downstream tasks (position-sensitive subset)"),
    BenchmarkSpec("gue", gue.run, False,
                  "GUE / DNABERT-2 multi-species understanding (needs --gue-dir)"),
]

BY_NAME = {spec.name: spec for spec in REGISTRY}


def select(names) -> list[BenchmarkSpec]:
    if not names or names == ["all"]:
        return list(REGISTRY)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {unknown}. Available: {list(BY_NAME)}")
    return [BY_NAME[n] for n in names]
