"""Build pretokenized uint16 shards from HyenaDNA's hg38 pretraining source.

Source files (same as HyenaDNA, for direct comparability — see CLAUDE.md §1):
    fasta: https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz
    bed:   https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed
           (chrom, start, end, split) intervals, reused from the Basenji/Enformer split.

Usage:
    uv run python -m dataset.prepare --data-dir data/hg38
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset.tokenizer import BOS_ID, encode

SHARD_TOKENS = 50_000_000  # tokens per .bin shard


def _build_fai(fa_path: Path, fai_path: Path):
    """Minimal samtools-compatible .fai index: name, length, offset, linebases, linewidth."""
    entries = []
    with open(fa_path, "rb") as f:
        offset = 0
        name = None
        seq_len = 0
        rec_offset = 0
        linebases = None
        linewidth = None
        for line in f:
            if line.startswith(b">"):
                if name is not None:
                    entries.append((name, seq_len, rec_offset, linebases, linewidth))
                name = line[1:].split()[0].decode()
                seq_len = 0
                rec_offset = offset + len(line)
                linebases = None
                linewidth = None
            else:
                stripped = line.rstrip(b"\n")
                if linebases is None:
                    linebases = len(stripped)
                    linewidth = len(line)
                seq_len += len(stripped)
            offset += len(line)
        if name is not None:
            entries.append((name, seq_len, rec_offset, linebases, linewidth))

    with open(fai_path, "w") as f:
        for name, seq_len, rec_offset, linebases, linewidth in entries:
            f.write(f"{name}\t{seq_len}\t{rec_offset}\t{linebases}\t{linewidth}\n")


def _fai_lookup(fai_path: Path) -> dict[str, tuple[int, int, int]]:
    fai = {}
    with open(fai_path) as f:
        for line in f:
            name, seq_len, rec_offset, linebases, linewidth = line.rstrip("\n").split("\t")
            fai[name] = (int(seq_len), int(rec_offset), int(linebases), int(linewidth))
    return fai


def _fetch_seq(fa_file, fai_entry, start: int, end: int) -> str:
    seq_len, rec_offset, linebases, linewidth = fai_entry
    end = min(end, seq_len)
    if start >= end:
        return ""
    start_line, start_col = divmod(start, linebases)
    byte_start = rec_offset + start_line * linewidth + start_col
    n_bases = end - start
    n_lines_span = (start_col + n_bases + linebases - 1) // linebases
    read_bytes = n_lines_span * linewidth
    fa_file.seek(byte_start)
    raw = fa_file.read(read_bytes)
    seq = raw.replace(b"\n", b"").decode("ascii")
    return seq[:n_bases].upper()


def _write_shards(token_chunks, out_dir: Path, split: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = []
    shard_idx = 0
    total = 0
    for chunk in token_chunks:
        buf.extend(chunk)
        while len(buf) >= SHARD_TOKENS:
            arr = np.array(buf[:SHARD_TOKENS], dtype=np.uint16)
            arr.tofile(out_dir / f"{split}_{shard_idx:05d}.bin")
            buf = buf[SHARD_TOKENS:]
            shard_idx += 1
            total += SHARD_TOKENS
    if buf:
        arr = np.array(buf, dtype=np.uint16)
        arr.tofile(out_dir / f"{split}_{shard_idx:05d}.bin")
        total += len(buf)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/hg38")
    parser.add_argument("--out-dir", default=None, help="defaults to <data-dir>")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    fa_path = data_dir / "hg38.ml.fa"
    bed_path = data_dir / "human-sequences.bed"
    fai_path = data_dir / "hg38.ml.fa.fai"

    if not fa_path.exists():
        raise FileNotFoundError(
            f"{fa_path} not found. Download with:\n"
            f"  curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz "
            f"| gunzip > {fa_path}"
        )
    if not bed_path.exists():
        raise FileNotFoundError(
            f"{bed_path} not found. Download with:\n"
            f"  curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed "
            f"-o {bed_path}"
        )

    if not fai_path.exists():
        print("Building fasta index...")
        _build_fai(fa_path, fai_path)
    fai = _fai_lookup(fai_path)

    intervals = {"train": [], "valid": [], "test": []}
    with open(bed_path) as f:
        for line in f:
            chrom, start, end, split = line.rstrip("\n").split("\t")
            if split not in intervals or chrom not in fai:
                continue
            intervals[split].append((chrom, int(start), int(end)))

    n_bases_total = 0
    n_bases_ambiguous = 0

    for split, ivs in intervals.items():
        print(f"[{split}] {len(ivs)} intervals")

        def token_gen(ivs=ivs, split=split):
            nonlocal n_bases_total, n_bases_ambiguous
            with open(fa_path, "rb") as fa_file:
                for chrom, start, end in tqdm(ivs, desc=split):
                    seq = _fetch_seq(fa_file, fai[chrom], start, end)
                    n_bases_total += len(seq)
                    n_bases_ambiguous += seq.count("N")
                    yield [BOS_ID] + encode(seq)

        n_tokens = _write_shards(token_gen(), out_dir / split, split)
        print(f"[{split}] wrote {n_tokens:,} tokens")

    frac_n = n_bases_ambiguous / max(n_bases_total, 1)
    print(f"\nTotal bases: {n_bases_total:,}  N fraction: {frac_n:.4%}")


if __name__ == "__main__":
    main()
