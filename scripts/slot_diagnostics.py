"""Fast-weight memory slot-utilisation diagnostics for the k-mer arms (D05/D06/D07).

Answers the questions §4.1 of the paper has to answer with measurements rather than with the
uniform-distribution pigeonhole bound:

  * what fraction of the 4^k slots is ever touched inside one window,
  * how concentrated the writes are (collisions per touched slot, entropy, effective slots),
  * whether that concentration is repeat-driven (the top-written k-mers are printed and
    flagged for low sequence complexity, e.g. poly-A runs),
  * and how much of each write actually lands, since the delta rule scales every update by the
    content-dependent gate beta_t: a slot hit 400 times at beta ~ 0.02 is not overwritten 400
    times. Both the raw write count and the beta-weighted "effective writes" are reported.

Genomic DNA is not uniform, so the uniform bound is a floor on collisions, not an estimate;
these numbers are what the writeup should quote.

    uv run python scripts/slot_diagnostics.py --config configs/d06_kmer_fastweight.yaml \
        --checkpoint results/d06/step_34178.pt --ctx 131072 --windows 8
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.lm_eval import _loader
from benchmarks.model_io import load_model, resolve_dtype
from dataset.tokenizer import TOKEN_TO_ID
from models.d05.kmer_memory import kmer_addresses, shift_write_addresses
from training.config import load_config

ID_TO_BASE = {TOKEN_TO_ID[b]: b for b in "ACGT"}


def decode_addr(addr: int, k: int) -> str:
    """Address -> k-mer string. `kmer_addresses` packs base-4 with the most recent base least
    significant, so digit i is the base i positions back from the end of the window."""
    return "".join("ACGT"[(addr >> (2 * i)) & 3] for i in range(k))[::-1]


def _low_complexity(kmer: str) -> bool:
    """True for the repeat-expansion k-mers the utilisation question is really about:
    homopolymers and short tandem repeats (period 1-4)."""
    return any(kmer == (kmer[:p] * (len(kmer) // p + 1))[:len(kmer)]
               for p in (1, 2, 3, 4) if p < len(kmer))


def _gate_hooks(model):
    """Capture beta_t = sigmoid(write_gate(h)) per k-mer memory without touching the model.

    The gate is read off the same hidden state the scan uses, so these are the actual write
    strengths of this forward pass, not a re-derivation.
    """
    captured, handles = {}, []
    for bi, blk in enumerate(model.blocks):
        attn = blk.attn
        if not hasattr(attn, "memories"):
            continue                       # D07 interleaves local-attention blocks; skip those
        for mem in attn.memories:
            def hook(module, inp, out, key=(bi, mem.k)):
                captured.setdefault(key, []).append(
                    torch.sigmoid(out.float().squeeze(-1)).detach().cpu())
            handles.append(mem.write_gate.register_forward_hook(hook))
    return captured, handles


def _stats(counts: np.ndarray, weights: np.ndarray, n_slots: int, k: int,
           n_writes: int, top_n: int) -> dict:
    touched = int((counts > 0).sum())
    hit = counts[counts > 0]
    p = hit / hit.sum() if hit.sum() else hit
    entropy = float(-(p * np.log(p)).sum()) if hit.size else 0.0
    # Expected distinct slots if the same number of writes were uniform over 4^k — the
    # pigeonhole baseline the paper quotes, evaluated exactly rather than asymptotically.
    uniform_touched = n_slots * (1.0 - math.exp(-n_writes / n_slots))
    order = np.argsort(-counts)[:top_n]
    return {
        "k": k,
        "n_slots": n_slots,
        "writes": n_writes,
        "slots_touched": touched,
        "coverage": round(touched / n_slots, 6),
        "coverage_if_uniform": round(uniform_touched / n_slots, 6),
        # Collisions = writes that landed on an already-occupied slot. This is the quantity
        # §4.1 bounds below by the pigeonhole argument.
        "collisions": n_writes - touched,
        "collision_rate": round((n_writes - touched) / max(n_writes, 1), 6),
        "writes_per_touched_mean": round(float(hit.mean()) if hit.size else 0.0, 3),
        "writes_per_touched_median": float(np.median(hit)) if hit.size else 0.0,
        "writes_per_touched_p99": float(np.percentile(hit, 99)) if hit.size else 0.0,
        "writes_per_touched_max": int(hit.max()) if hit.size else 0,
        # exp(H) — how many slots the write mass behaves like, if it were spread evenly over
        # that many. Far below `slots_touched` means capacity is concentrated on a few k-mers.
        "effective_slots": round(math.exp(entropy), 2),
        "write_entropy_nats": round(entropy, 4),
        "beta_mean": round(float(weights.sum() / max(n_writes, 1)), 4),
        # Sum of beta over each slot: the delta rule's actual overwrite pressure.
        "effective_writes_per_touched_max": round(float(weights.max()), 3),
        "top_kmers": [
            {"kmer": decode_addr(int(a), k), "writes": int(counts[a]),
             "effective_writes": round(float(weights[a]), 2),
             "low_complexity": _low_complexity(decode_addr(int(a), k))}
            for a in order if counts[a] > 0],
        "top_share": round(float(counts[order].sum() / max(n_writes, 1)), 4),
        "low_complexity_write_share": round(
            float(sum(counts[a] for a in order if _low_complexity(decode_addr(int(a), k)))
                  / max(n_writes, 1)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ctx", type=int, default=131072, help="window length in bp")
    ap.add_argument("--windows", type=int, default=8,
                    help="held-out windows to average over (each is reported separately too)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-n", type=int, default=20, help="most-written k-mers to list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="JSON output path (default results/<model_id>/slot_diagnostics.json)")
    args = ap.parse_args()

    config = load_config(args.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = resolve_dtype(args.device, config.get("precision", "bf16"))
    model, step = load_model(config, args.checkpoint, args.device, dtype)
    if not any(hasattr(b.attn, "memories") for b in model.blocks):
        print(f"{config['model_id']} has no k-mer memory — nothing to diagnose", file=sys.stderr)
        return 2

    loader = _loader(config, args.split, args.device)
    per_window = []
    for w in range(args.windows):
        x, _ = loader.get_batch(1, args.ctx)
        captured, handles = _gate_hooks(model)
        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()

        # Write addresses are a pure function of the token ids (same call KmerGPT makes), so
        # they are recomputed here rather than scraped out of the forward pass.
        rows = {}
        for (block, k), betas in sorted(captured.items()):
            wa = shift_write_addresses(kmer_addresses(x, k))[0].cpu().numpy()
            # One capture per forward call of that memory; the chunked path calls it once per
            # chunk, so concatenate along time to recover the full-sequence beta_t.
            beta = torch.cat(betas, dim=1)[0].numpy()
            valid = wa >= 0
            addr = wa[valid].astype(np.int64)
            n_slots = 4 ** k
            counts = np.bincount(addr, minlength=n_slots)
            weights = np.bincount(addr, weights=beta[valid].astype(np.float64),
                                  minlength=n_slots)
            rows[f"block{block}_k{k}"] = _stats(counts, weights, n_slots, k,
                                                int(valid.sum()), args.top_n)
        per_window.append(rows)
        print(f"[window {w + 1}/{args.windows}] "
              + "  ".join(f"{name}: cov={r['coverage']:.4f} "
                          f"coll={r['collision_rate']:.4f} eff={r['effective_slots']:.0f}"
                          for name, r in rows.items()), flush=True)

    # Average the scalar fields across windows; keep window 0's k-mer listing as the exemplar.
    summary = {}
    for name in per_window[0]:
        agg = {}
        for key, val in per_window[0][name].items():
            vals = [pw[name][key] for pw in per_window]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                agg[key] = round(float(np.mean(vals)), 6)
                agg[f"{key}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6)
            else:
                agg[key] = val
        summary[name] = agg

    out = {"model_id": config["model_id"], "checkpoint": args.checkpoint,
           "checkpoint_step": step, "ctx": args.ctx, "windows": args.windows,
           "split": args.split, "seed": args.seed,
           "summary": summary, "per_window": per_window}
    path = args.out or f"results/{config['model_id']}/slot_diagnostics.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{config['model_id']} @ step {step} — {args.windows} x {args.ctx}bp windows\n")
    for name, r in summary.items():
        print(f"  {name}: {r['n_slots']:,} slots | {r['writes']:,.0f} writes")
        print(f"    touched          {r['slots_touched']:,.0f} ({r['coverage']:.2%})"
              f"   [uniform would touch {r['coverage_if_uniform']:.2%}]")
        print(f"    collisions       {r['collisions']:,.0f} ({r['collision_rate']:.2%} of writes)")
        print(f"    writes/slot      mean {r['writes_per_touched_mean']:.2f}  "
              f"med {r['writes_per_touched_median']:.0f}  "
              f"p99 {r['writes_per_touched_p99']:.0f}  max {r['writes_per_touched_max']:.0f}")
        print(f"    effective slots  {r['effective_slots']:,.0f}  "
              f"(exp of write entropy; vs {r['slots_touched']:,.0f} touched)")
        print(f"    mean beta        {r['beta_mean']:.4f}")
        print(f"    top-{args.top_n} share    {r['top_share']:.2%}  "
              f"(low-complexity among them: {r['low_complexity_write_share']:.2%})")
        for e in r["top_kmers"][:8]:
            print(f"      {e['kmer']}  {e['writes']:>7,} writes  "
                  f"beta-sum {e['effective_writes']:>9,.1f}"
                  f"{'  <- low complexity' if e['low_complexity'] else ''}")
        print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
