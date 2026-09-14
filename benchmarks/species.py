"""Species classification — "is genomic identity distributed across the whole window".

There is no canonical auto-downloadable multi-species classification split (the InstaDeep
multi-species corpus is a dataset script that no longer loads under datasets>=3, and Genomic
Benchmarks' human-vs-worm covers only the 2-way 200bp case, which the genomic_benchmarks
benchmark already runs). So this reads reference FASTAs you place yourself:

    mkdir -p data/species
    for url in <ensembl toplevel dna fasta urls>; do curl -L "$url" | gunzip > data/species/<name>.fa; done
    uv run python scripts/benchmark.py ... --species-dir data/species

One class per .fa/.fasta file, windows sampled uniformly at random from each genome at the
requested context length, balanced across species. Unlike the fixed-length suites this IS
length-swept: species identity is exactly the kind of signal that should get easier with a
longer window, so the accuracy-vs-context curve is the interesting output.
"""
from pathlib import Path

import numpy as np

from benchmarks.fasta import Fasta, n_fraction
from benchmarks.harness import SkipBenchmark
from benchmarks.probe import linear_probe
from benchmarks.model_io import embed_sequences

MAX_N_FRACTION = 0.1


def _sample_windows(fasta: Fasta, n: int, ctx: int, rng) -> list[str]:
    chroms = [c for c in fasta.chroms if fasta.length(c) > 4 * ctx]
    if not chroms:
        raise SkipBenchmark(f"no contig longer than 4x ctx={ctx}")
    weights = np.array([fasta.length(c) for c in chroms], dtype=float)
    weights /= weights.sum()
    seqs, attempts = [], 0
    while len(seqs) < n and attempts < 20 * n:
        attempts += 1
        chrom = chroms[rng.choice(len(chroms), p=weights)]
        start = int(rng.integers(0, fasta.length(chrom) - ctx))
        seq = fasta.fetch(chrom, start, start + ctx)
        if len(seq) == ctx and n_fraction(seq) <= MAX_N_FRACTION:
            seqs.append(seq)
    if len(seqs) < n:
        raise SkipBenchmark(f"only found {len(seqs)}/{n} non-N windows")
    return seqs


def run(bctx) -> dict:
    if not bctx.args.species_dir:
        raise SkipBenchmark("no --species-dir given (see benchmarks/species.py for setup)")
    root = Path(bctx.args.species_dir)
    genomes = sorted(p for p in root.glob("*") if p.suffix in (".fa", ".fasta", ".fna"))
    if len(genomes) < 2:
        raise SkipBenchmark(f"need >=2 genome FASTAs in {root}, found {len(genomes)}")

    ctx = bctx.ctx
    per_class = bctx.args.species_per_class
    rng = np.random.default_rng(bctx.args.seed)

    seqs, labels = [], []
    for label, path in enumerate(genomes):
        fasta = Fasta(path)
        try:
            seqs.extend(_sample_windows(fasta, per_class, ctx, rng))
        finally:
            fasta.close()
        labels.extend([label] * per_class)

    x = embed_sequences(bctx.model, seqs, ctx=ctx, device=bctx.device,
                        batch_size=bctx.args.probe_batch_size)
    y = np.array(labels)
    # held-out split is by sample, not by genome — the classes are the genomes themselves
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    metrics = linear_probe(x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]])
    metrics.update(n_species=len(genomes), per_class=per_class,
                   chance=round(1 / len(genomes), 4),
                   species=",".join(p.stem for p in genomes))
    return metrics
