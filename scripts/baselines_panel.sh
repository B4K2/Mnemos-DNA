#!/usr/bin/env bash
# One apples-to-apples panel: released HyenaDNA / Caduceus-Ph / Caduceus-PS checkpoints and our
# own arms, all pushed through the SAME harness — same hg38 windows, same seeds, same frozen
# linear probe, same 1000x bootstrap CIs.
#
#   bash scripts/baselines_panel.sh                 # external arms + d02, d06, d07
#   bash scripts/baselines_panel.sh h01_hyenadna_1m # just these configs
#
# Benchmarks: `species` (length-swept 8K -> 1M: the "does more context help" curve) and `lrb`
# (Genomics Long-Range Benchmark; capped at 64K by --lrb-max-ctx as always).
# Everything is a frozen-feature probe, so the external checkpoints are never fine-tuned and
# their pretraining budget is the only thing that differs — label that in the writeup.
set -uo pipefail

LENGTHS="${LENGTHS:-8K,16K,32K,65K,131K,262K,524K,1M}"
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  # d01/d01_new are omitted: their released checkpoints predate the current KDA module and no
  # longer load (see MEMORY: bf16 scalar gate freeze). Add them back once retrained.
  ARMS=(h01_hyenadna_1m c01_caduceus_ph c02_caduceus_ps d02_kda_local \
        d06_kmer_fastweight d07_kmer_local)
fi

for arm in "${ARMS[@]}"; do
  echo "=============================== $arm ==============================="
  # --offload (chunked prefill carrying the recurrent state) is exact for the KDA arms only.
  # HyenaDNA's implicit long convolution and Caduceus' bidirectional scan both read the whole
  # sequence at once, so chunking them would change their output, not just their memory: the
  # external arms run natively and their OOM ceiling is a real result (that ceiling IS the
  # §6.3 comparison), recorded as a SKIP: OOM row.
  OFFLOAD=--offload
  case "$arm" in h0*|c0*) OFFLOAD= ;; esac
  uv run python scripts/benchmark.py \
    --config "configs/${arm}.yaml" \
    --benchmarks species,lrb \
    --lengths "$LENGTHS" \
    --species-dir data/species \
    --hg38-fasta data/hg38/hg38.ml.fa \
    --lrb-tasks "${LRB_TASKS:-vep_eqtl,chromatin,enhancer,promoter}" \
    --downstream-mode probe \
    --bootstrap 1000 \
    --output-stem panel_cuda \
    $OFFLOAD
done

echo
echo "per-arm results: results/<model_id>/panel_cuda.{json,csv}"
