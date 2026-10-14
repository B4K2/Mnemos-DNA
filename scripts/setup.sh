#!/usr/bin/env bash
# One-shot project setup: Python env, .env, hg38 download + tokenized shards.
# Idempotent — every step is skipped if its output already exists, so rerun freely.
#
#   bash scripts/setup.sh              # env + hg38 pretraining shards (~3.1GB FASTA)
#   bash scripts/setup.sh --eval       # + species genomes and GUE for the full benchmark suite
#   bash scripts/setup.sh --weights    # + pretrained checkpoints from B4K2xx/Mnemos-DNA
#   bash scripts/setup.sh --all        # everything
#
# The Caduceus baselines (c01/c02) additionally need mamba-ssm/causal-conv1d built for sm_120;
# see CLAUDE.md §9 — not automated here.
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL=0; WEIGHTS=0
for arg in "$@"; do
  case "$arg" in
    --eval) EVAL=1 ;;
    --weights) WEIGHTS=1 ;;
    --all) EVAL=1; WEIGHTS=1 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

DATA_DIR=data/hg38
BED_URL=https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed

step() { echo; echo "==> $*"; }

step "uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "Python environment (cu128 torch wheels, pinned in pyproject.toml)"
# --inexact: keep packages installed outside the lockfile (e.g. the sm_120 mamba-ssm build)
uv sync --inexact
uv run python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

step ".env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example — set HUGGING_FACE_TOKEN in it"
else
  echo ".env exists"
fi

step "hg38 FASTA + index (HyenaDNA/Basenji source)"
uv run python scripts/fetch_eval_data.py hg38 --data-dir "$DATA_DIR"

step "Basenji/Enformer train/valid/test split bed"
if [ ! -s "$DATA_DIR/human-sequences.bed" ]; then
  curl -fL "$BED_URL" -o "$DATA_DIR/human-sequences.bed.tmp"
  mv "$DATA_DIR/human-sequences.bed.tmp" "$DATA_DIR/human-sequences.bed"
else
  echo "bed exists"
fi

step "Tokenized uint16 shards"
if ls "$DATA_DIR"/train/train_*.bin >/dev/null 2>&1 && ls "$DATA_DIR"/test/test_*.bin >/dev/null 2>&1; then
  echo "shards exist in $DATA_DIR/{train,valid,test} — delete them to rebuild"
else
  uv run python -m dataset.prepare --data-dir "$DATA_DIR"
fi

if [ "$EVAL" = 1 ]; then
  step "Evaluation data: species genomes + GUE"
  uv run python scripts/fetch_eval_data.py species gue
fi

if [ "$WEIGHTS" = 1 ]; then
  step "Pretrained checkpoints -> checkpoints/Mnemos-DNA"
  uv run hf download B4K2xx/Mnemos-DNA --local-dir checkpoints/Mnemos-DNA
fi

step "Done"
echo "Next: uv run python scripts/train.py --config configs/d06_kmer_fastweight.yaml"
