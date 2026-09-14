"""Random-access FASTA reader for the coordinate-based benchmarks (LRB, species).

Reuses the samtools-compatible .fai index builder/reader already written for the pretraining
pipeline (dataset/prepare.py) rather than adding a pyfaidx dependency, which also guarantees
benchmark windows are extracted byte-identically to pretraining windows (uppercased, N kept).
"""
from pathlib import Path

from dataset.prepare import _build_fai, _fai_lookup, _fetch_seq
from benchmarks.harness import SkipBenchmark


class Fasta:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise SkipBenchmark(f"fasta not found: {self.path}")
        fai_path = self.path.with_suffix(self.path.suffix + ".fai")
        if not fai_path.exists():
            _build_fai(self.path, fai_path)
        self.index = _fai_lookup(fai_path)
        self._handle = open(self.path, "rb")

    @property
    def chroms(self):
        return self.index

    def length(self, chrom: str) -> int:
        return self.index[chrom][0]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.index:
            return ""
        return _fetch_seq(self._handle, self.index[chrom], max(0, start), end)

    def window(self, chrom: str, centre: int, length: int) -> str:
        """Window of `length` bp centred on `centre`, clipped to the chromosome and then
        right-padded with N so every returned sequence has the requested length."""
        if chrom not in self.index:
            return ""
        half = length // 2
        seq = self.fetch(chrom, centre - half, centre - half + length)
        if not seq:
            return ""
        return seq if len(seq) >= length else seq + "N" * (length - len(seq))

    def close(self):
        self._handle.close()


def n_fraction(seq: str) -> float:
    return seq.count("N") / max(len(seq), 1)
