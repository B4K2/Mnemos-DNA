#!/usr/bin/env bash
# Full post-training evaluation for one arm: (1) frozen-probe suite with the offload fallback so
# the length sweeps reach 1M, then (2) full fine-tuning on the fixed-length downstream suites.
# Two output files, never overwriting each other:
#   results/<id>/benchmark_cuda.json           probe rows + lm/needle/scaling/lrb/species
#   results/<id>/benchmark_cuda_finetune.json  fine-tuned genomic_benchmarks / nt / gue rows
#
# Both stages resume: rerunning the same command after a Ctrl-C replays finished points (and,
# in the fine-tune stage, finished tasks) from results/<id>/benchmark_cuda[_finetune].partial.jsonl
# instead of re-measuring them. The ledgers survive a *successful* stage too, so relaunching this
# script because stage 2 was interrupted replays stage 1 in seconds rather than redoing it, and
# the .json/.csv are merged rather than overwritten. --no-resume forces a clean re-measurement.
#
#   bash scripts/eval_all.sh configs/d06_kmer_fastweight.yaml [results/d06/final.pt] [extra benchmark.py args...]
#   bash scripts/eval_all.sh configs/d06_kmer_fastweight.yaml results/d06/final.pt --skip-gue
set -euo pipefail
CONFIG=${1:?usage: eval_all.sh <config.yaml> [checkpoint] [extra args]}
CKPT=${2:-}
shift $(( $# >= 2 ? 2 : 1 ))
CKPT_ARG=(); [[ -n "$CKPT" ]] && CKPT_ARG=(--checkpoint "$CKPT")

# --skip-gue applies to the fine-tune stage only: 37 GUE tasks x a full fine-tune dominate
# that stage's runtime, while the probe stage's GUE rows are cheap. Strip it from the args
# forwarded to stage 1 so passing it here does not silently drop the probe rows too.
EXTRA=(); SKIP_GUE=()
for a in "$@"; do
  if [[ "$a" == "--skip-gue" ]]; then SKIP_GUE=(--skip-gue); else EXTRA+=("$a"); fi
done
set -- "${EXTRA[@]+"${EXTRA[@]}"}"

echo "=== [1/2] probe suite (all benchmarks, --offload) ==="
uv run python scripts/benchmark.py --config "$CONFIG" "${CKPT_ARG[@]}" --offload "$@"

echo "=== [2/2] fine-tune suite (genomic_benchmarks, nt, gue) ==="
uv run python scripts/benchmark.py --config "$CONFIG" "${CKPT_ARG[@]}" \
    --benchmarks genomic_benchmarks,nt,gue --downstream-mode finetune \
    "${SKIP_GUE[@]+"${SKIP_GUE[@]}"}" \
    --probe-train-limit 0 --probe-test-limit 0 "$@"   # full official splits, as HyenaDNA/NT do
