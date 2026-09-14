"""Download the evaluation data the benchmark suite skips when it is absent.

Three of the eight benchmarks in `scripts/benchmark.py` need files that the pretraining
pipeline does not leave on disk:

  lrb      needs data/hg38/hg38.ml.fa   -- dataset.prepare tokenizes it into .bin shards and
                                           the 3.1GB FASTA is usually deleted afterwards, but
                                           LRB cuts its coordinate windows from the FASTA
                                           itself, so it has to come back.
  species  needs --species-dir          -- one reference FASTA per class; there is no canonical
                                           packaged split (see benchmarks/species.py).
  gue      needs --gue-dir              -- the official DNABERT-2 distribution is a zip behind a
                                           Google Drive link that cannot be curl'd; this uses a
                                           community HuggingFace mirror instead (see GUE_REPO).

Usage:
    uv run python scripts/fetch_eval_data.py all              # everything (~7GB)
    uv run python scripts/fetch_eval_data.py hg38             # LRB's FASTA only (~3.1GB)
    uv run python scripts/fetch_eval_data.py species          # ~3.5GB
    uv run python scripts/fetch_eval_data.py gue              # ~150MB

Every step is resumable and skips work that is already on disk, so re-running is cheap.
"""
import argparse
import gzip
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HG38_URL = "https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"

# GUE mirror. The official release is a Google-Drive zip linked from the DNABERT-2 README and
# cannot be fetched non-interactively; this repo is a straight file-for-file copy of the same
# 37 task directories. Label it as a community mirror in the writeup, not as the official zip.
GUE_REPO = "leannmlindsey/GUE"

# Species classification classes. Ensembl release directories are listed over HTTPS and the
# exact filename is regex-matched, so an Ensembl version bump does not break this. Each genome
# is chosen to have chromosomes far longer than the 1M-token sweep ceiling -- benchmarks/
# species.py requires contigs > 4x the context length, which rules out e.g. yeast.
ENSEMBL_RELEASE = "112"
SPECIES = {
    "mouse": ("mus_musculus", "Mus_musculus"),
    "zebrafish": ("danio_rerio", "Danio_rerio"),
    "fly": ("drosophila_melanogaster", "Drosophila_melanogaster"),
    "worm": ("caenorhabditis_elegans", "Caenorhabditis_elegans"),
    "chicken": ("gallus_gallus", "Gallus_gallus"),
}


def _download_gunzip(url: str, dest: Path):
    """Stream a .gz straight to an uncompressed file; never materialize the archive."""
    if dest.exists():
        print(f"  have {dest} ({dest.stat().st_size / 1e9:.2f} GB) -- skipping")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {url}\n    -> {dest}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        with gzip.GzipFile(fileobj=resp) as gz:
            shutil.copyfileobj(gz, out, length=1 << 22)
    tmp.rename(dest)
    print(f"    done ({dest.stat().st_size / 1e9:.2f} GB)")


def fetch_hg38(data_dir: Path):
    """LRB reads windows from the same FASTA the pretraining shards were built from."""
    print("[hg38] FASTA for the lrb benchmark")
    data_dir.mkdir(parents=True, exist_ok=True)
    fa = data_dir / "hg38.ml.fa"
    _download_gunzip(HG38_URL, fa)
    # The .fai already on disk was built from this exact file; only build it if it is missing.
    fai = fa.with_suffix(".fa.fai")
    if not fai.exists():
        from dataset.prepare import _build_fai
        print("  building .fai index")
        _build_fai(fa, fai)
    return fa


def _ensembl_url(species_dir: str, prefix: str) -> str:
    """Prefer the primary assembly (no scaffolds/haplotypes); fall back to toplevel, which is
    what Ensembl ships for the compact genomes that have no scaffold-free build."""
    base = f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/fasta/{species_dir}/dna/"
    with urllib.request.urlopen(base) as resp:
        listing = resp.read().decode("utf-8", "replace")
    for kind in ("primary_assembly", "toplevel"):
        names = re.findall(rf"{prefix}\.[\w.-]*dna\.{kind}\.fa\.gz", listing)
        if names:
            return base + sorted(set(names))[0]
    raise SystemExit(f"no {prefix} dna.[primary_assembly|toplevel].fa.gz in {base} "
                     "-- Ensembl layout changed?")


def fetch_species(species_dir: Path, data_dir: Path, which: list[str]):
    """One .fa per class. Human is a symlink to the hg38 FASTA so the human class is byte-for-byte
    the pretraining assembly rather than a second copy on disk."""
    print("[species] reference genomes for the species benchmark")
    species_dir.mkdir(parents=True, exist_ok=True)

    if "human" in which:
        human = species_dir / "human.fa"
        hg38 = (data_dir / "hg38.ml.fa").resolve()
        if not hg38.exists():
            fetch_hg38(data_dir)
        if not human.exists():
            human.symlink_to(hg38)
        print(f"  human.fa -> {hg38}")

    for name in which:
        if name == "human":
            continue
        sub, pattern = SPECIES[name]
        _download_gunzip(_ensembl_url(sub, pattern), species_dir / f"{name}.fa")
    return species_dir


def fetch_gue(gue_dir: Path):
    """Snapshot the mirror, then normalise away its top-level GUE/ wrapper so the resulting
    directory matches the layout benchmarks/gue.py expects (<group>/<task>/train.csv)."""
    print(f"[gue] DNABERT-2 GUE tasks from the {GUE_REPO} mirror")
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download
    # .env stores the token as HUGGING_FACE_TOKEN; huggingface_hub only reads HF_TOKEN
    # (same bridge as scripts/benchmark.py:authenticate_hf).
    load_dotenv()
    if os.environ.get("HUGGING_FACE_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_TOKEN"]
    local = Path(snapshot_download(GUE_REPO, repo_type="dataset", max_workers=4))
    inner = local / "GUE"
    src = inner if inner.is_dir() else local
    gue_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for task in sorted(p.parent for p in src.rglob("train.csv")):
        dest = gue_dir / task.relative_to(src)
        dest.mkdir(parents=True, exist_ok=True)
        for csv in task.glob("*.csv"):
            target = dest / csv.name
            if not target.exists():
                shutil.copyfile(csv, target)
        n += 1
    print(f"  {n} task directories under {gue_dir}")
    return gue_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="+", choices=["all", "hg38", "species", "gue"])
    p.add_argument("--data-dir", default="data/hg38")
    p.add_argument("--species-dir", default="data/species")
    p.add_argument("--gue-dir", default="data/GUE")
    p.add_argument("--species", default="human,mouse,zebrafish,fly,worm",
                   help=f"comma-separated subset of: human,{','.join(SPECIES)}")
    args = p.parse_args()

    what = set(args.what)
    if "all" in what:
        what = {"hg38", "species", "gue"}
    data_dir = Path(args.data_dir)

    if "hg38" in what:
        fetch_hg38(data_dir)
    if "species" in what:
        fetch_species(Path(args.species_dir), data_dir,
                      [s.strip() for s in args.species.split(",") if s.strip()])
    if "gue" in what:
        fetch_gue(Path(args.gue_dir))
    print("\ndone.")


if __name__ == "__main__":
    main()
