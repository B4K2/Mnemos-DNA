# Mnemos-DNA

**Gated delta-rule linear attention with a k-mer fast-weight memory, for edge-scale DNA
language modeling.** ~2.3M-parameter character-level models pretrained on the human genome
(GRCh38, 4.5B tokens, 131,072 bp context) — every experiment trained and evaluated on a single
8 GB consumer GPU (RTX 5060 Laptop).

Paper: [`paper/main.pdf`](paper/main.pdf) ·
Pretrained weights: [B4K2xx/Mnemos-DNA on Hugging Face](https://huggingface.co/B4K2xx/Mnemos-DNA)

## What this is

Kimi Delta Attention (KDA) — gated delta-rule linear attention — has strong natural-language
results but no published genomic evaluation. This repo is a matched-token, matched-parameter
ablation study asking where its finite-size-state recall bound shows up on DNA, and whether it
can be repaired within an O(N) budget.

**Main findings**

1. **The recall bound is a wall at distance zero.** On a repeat-copy probe over real hg38
   background, pure KDA (32×32 state per head) shows no measurable exact recall at *any*
   distance — and a windowed local-attention hybrid is identical to it.
2. **A content-addressed k-mer fast-weight memory removes the wall.** Every layer keys a table
   of 4^8 = 65,536 slots by the exact last 8-mer; each slot is a 16×16 delta-rule fast-weight
   matrix. This restores 0.53–0.87 nats of copy advantage out to 64 kbp, is robust to SNP-level
   substitutions in the repeat, and stays O(N) and bitwise-exact under chunked inference.
   Costs: ~3% parameters, ~2× prefill throughput, 4096× recurrent inference state.
3. **Ultra-long fixed-context pretraining damages sub-kilobase grammar.** Under a matched
   fine-tuning protocol, a randomly initialized model beats its own 4.5B-token pretrained
   checkpoint by ~9 points on splice sites. The same control applied to the released
   HyenaDNA-1M checkpoint shows the same signature.
4. **Fine-tuned architecture comparisons are protocol-fragile.** Letting the pooling head,
   epoch budget or training-set caps drift between arms produces spurious cross-arm effects of
   up to 30 points — an order of magnitude larger than any real architecture effect measured
   here. Every downstream number is therefore produced by one frozen, seeded protocol.
5. **Deployment:** 1 Mbp exact inference in 5.6 GB VRAM, where released HyenaDNA/Caduceus
   checkpoints (~3× the parameters) stop at 256–512 K on the same card.

## Repo layout

```
models/           model arms; shared blocks + per-arm mixers (models/d06 = Mnemos-DNA)
models/external/  released HyenaDNA / Caduceus checkpoints wrapped into the same harness
dataset/          hg38 download + tokenization into uint16 shards (python -m dataset.prepare)
training/         trainer, config merging (configs/base.yaml + per-arm yaml), curriculum
configs/          one yaml per arm (d01/d02/d06/d07, external h01/h02/c01/c02, controls)
benchmarks/       evaluation suite: lm, needle (recall probe), scaling, species, lrb,
                  genomic_benchmarks, nt, gue — probe and full-fine-tune modes
scripts/          train.py, benchmark.py, fetch_eval_data.py, eval_all.sh,
                  run_paper_queue.py (the resumable job queue that produced every paper number)
analysis/         figure/table helpers
paper/            LaTeX source, make_figures.py (regenerates all figures from results/)
results/          per-arm result tables (JSON/CSV committed; checkpoints not)
```

## Setup

Requires a CUDA GPU (Blackwell/sm_120 needs the cu128 PyTorch wheels pinned in
`pyproject.toml`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # set HUGGING_FACE_TOKEN
uv run python -m dataset.prepare        # downloads + tokenizes hg38
```

## Reproduce

```bash
# pretrain one arm (resumable; checkpoints + MLflow run under results/<id>/)
uv run python scripts/train.py --config configs/d06_kmer_fastweight.yaml

# full evaluation suite for one arm
bash scripts/eval_all.sh configs/d06_kmer_fastweight.yaml results/d06/final.pt

# everything behind the paper, in order, resumably (Ctrl+C pauses; rerun resumes)
uv run python scripts/run_paper_queue.py --list
uv run python scripts/run_paper_queue.py

# regenerate paper figures/tables from results/, then build the PDF
uv run python paper/make_figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data notes

- Pretraining: GRCh38/hg38, HyenaDNA's held-out-chromosome split, single-nucleotide vocabulary.
- Downstream: Genomic Benchmarks, Nucleotide Transformer tasks, Genomics Long-Range Benchmark,
  GUE (via the `leannmlindsey/GUE` HuggingFace community mirror — cite as a mirror, not the
  official DNABERT-2 release). `scripts/fetch_eval_data.py all` fetches what the suite needs.
- External baselines are the released `LongSafari/hyenadna-*` and `kuleshov-group/caduceus-*`
  checkpoints, run through this harness with vocabulary remapped losslessly; comparisons to
  them are deployment-feasibility comparisons (different pretraining budgets), labelled as such
  everywhere.

## Author

Akshat Balyan
