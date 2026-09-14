"""Long-range exact-recall probe — the mechanistic centrepiece of §6.4 in CLAUDE.md.

Design (associative recall / "needle in a haystack", zero-shot, no fine-tuning):

    [ background ][ KEY(k bp) VALUE(1 bp) ][ d tokens of background ][ KEY ]
                                                                        ^
                                                model must predict VALUE here

The KEY is a random k-mer that (with k=12, 4^12 ~ 1.7e7) does not recur by chance in the
window; the VALUE is a single nucleotide drawn uniformly from ACGT. Success = argmax over the
four base logits at the final position equals VALUE, so chance is 25%.

`d` — the key->query distance — is swept INDEPENDENTLY of the context length. This matters:
an earlier version of this probe planted the needle in the first 5% of the window and queried
at the end, which made d ~= ctx by construction, so the shortest measurable distance was the
shortest context in the sweep (~8kbp). That left no near-distance regime where a model is
expected to succeed, i.e. no positive control, and a flat chance-level line was the only
possible outcome whether or not the model had any in-context copy ability at all. The short-d
rows are what make a long-d failure interpretable as the finite-state recall bound (Prop. 2)
rather than as "this model never had induction ability to begin with".

Read the output as a curve, not a number: d=64 is the control (if that is at chance, the model
has no in-context copy ability and the long-d rows say nothing about state capacity), and the d
at which accuracy falls to chance is the quantity Prop. 2 predicts differs across arms — pure
KDA (D01/D04) should break earliest, windowed hybrids (D02/D05) at roughly their window, and
the full-attention hybrids (D03/D06) and D07 latest.

The background is real held-out genome rather than random ACGT so that the difficulty comes
from genuine genomic statistics (tandem repeats, transposons) rather than from an artificial
i.i.d. distractor distribution.

Three modes (`--needle-mode`):

`copy` (default) — repeat-copy Δ-perplexity, zero-shot. Layout, built to exactly fill each
swept context length:

    repeat : [ background ][ segment A ][ d tokens of background ][ segment A ]
    control: [ background ][ (bg)      ][ d tokens of background ][ segment A ]
                                                                    ^^^^^^^^^
                                          per-token CE here, identical tokens in both windows

The metric is `copy_advantage` = ce(control) - ce(repeat), in nats/token: how much better the
model predicts segment A when it already saw it `d` tokens earlier. Genomes are full of
genuine repeats (transposons, tandem arrays), so pretrained DNA LMs have this copying ability
zero-shot. The pair shares the same background AND the same scored segment — the two windows
differ only in the earlier occurrence — so the diff isolates recall; `stderr` is the paired
standard error. (Scoring *different* segments in repeat vs control does not work: segments'
intrinsic entropy differences, std ~0.3 nats, swamp the ~0.1-nat recall effect.) This is the mode that operationalizes the Prop. 2 state-capacity bound:
to profit from the repeat, the model must still hold segment A in its finite state after `d`
intervening tokens.

`trained` — the MQAR protocol from the linear-attention recall literature (DeltaNet,
Gated DeltaNet, Kimi Linear, KATA all measure recall on models *trained* on the task): each arm
is first fine-tuned for a few hundred steps on the synthetic format at a short context
(`--needle-train-ctx`), with distances sampled log-uniformly up to what fits that context, loss
on the value position only. Evaluation then sweeps d far beyond the trained range, so long-d
degradation measures state capacity, not format unfamiliarity. Training is identical (same
hyperparameters, same seed, same data order) across arms, so the D01..D07 comparison stays
paired. The fine-tuned weights are cached for the whole sweep and the original checkpoint is
restored afterwards, so every other benchmark still sees pristine weights.

`zeroshot` — the raw probe with no task training. Empirically this is at chance even for
pretrained HyenaDNA at d=64: in genomic statistics an exact 12-mer match recurs constantly
across the corpus with uncorrelated continuations, so pretraining alone gives the model no
reason to copy — unlike text, where induction emerges as a byproduct. Kept for reference, but a
chance-level zeroshot row says the format is out-of-distribution, not that the model lacks
recall capacity.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from dataset.tokenizer import TOKEN_TO_ID, UNK_ID
from benchmarks.bootstrap import bootstrap_mean
from benchmarks.harness import SkipBenchmark, parse_lengths
from benchmarks.lm_eval import _loader

BASE_IDS = [TOKEN_TO_ID[b] for b in "ACGT"]

# Swept per context length; distances that do not fit the window are skipped, so the same
# default works at every ctx. d=64 is the positive control.
DEFAULT_DISTANCES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

# Background tokens kept before the needle, so the key/value pair never sits at position 0
# (a model can special-case the sequence start).
_LEAD_IN = 64


def _plant(seq, rng, base_ids, ctx: int, dist: int, key_len: int) -> int:
    """Overwrite `seq` in place with [UNK][key][value] ... [UNK][key] at distance `dist`.
    Returns the 0..3 index of the value base.

    The [UNK] marker (never present in the hg38 shards, which are pure ACGTN) delimits the
    needle. Without it, solving the task requires *discovering* a 12-token string-match
    circuit against genomic background — a 2M-param model showed zero learning over 3000
    fine-tune steps on the unmarked format. The marker makes the needle findable, so the
    probe isolates what §6.4 is actually after: whether the binding survives `dist` tokens of
    state decay, not whether the model can invent 12-gram matching from scratch."""
    q_start = ctx - key_len - 1                       # [UNK][key] flush with the end
    p_start = q_start - dist - key_len - 2            # [UNK][key][value]
    key = base_ids[torch.from_numpy(rng.integers(0, 4, key_len)).to(base_ids.device)]
    value = int(rng.integers(0, 4))
    seq[p_start] = UNK_ID
    seq[p_start + 1:p_start + 1 + key_len] = key
    seq[p_start + 1 + key_len] = base_ids[value]
    seq[q_start] = UNK_ID
    seq[q_start + 1:] = key
    return value


def _snapshot(model) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _plant_training_needles(seq, rng, base_ids, ctx, key_len, n_pairs, log_lo, log_hi):
    """Plant up to n_pairs [key][value] pairs plus their bare-[key] queries at random,
    non-overlapping positions. Returns (loss_positions, targets): the loss position is the
    last token of each query key, whose next-token prediction should be the value — the exact
    readout the eval uses. One needle per sequence gave one supervised token per ctx tokens,
    which was far too sparse to learn from; this is the multi-query MQAR training layout."""
    occupied = []

    def _free(a, b):
        return all(b <= s or a >= e for s, e in occupied)

    positions, targets = [], []
    for _ in range(n_pairs):
        for _attempt in range(20):
            dist = int(round(math.exp(rng.uniform(log_lo, log_hi))))
            lo = _LEAD_IN + dist + key_len + 2
            hi = ctx - key_len  # q_start + key_len + 1 <= ctx
            if lo >= hi:
                continue
            q_start = int(rng.integers(lo, hi))
            p_start = q_start - dist - key_len - 2
            if (_free(p_start, p_start + key_len + 2)
                    and _free(q_start, q_start + key_len + 1)):
                break
        else:
            continue
        key = base_ids[torch.from_numpy(rng.integers(0, 4, key_len)).to(base_ids.device)]
        value = int(rng.integers(0, 4))
        seq[p_start] = UNK_ID
        seq[p_start + 1:p_start + 1 + key_len] = key
        seq[p_start + 1 + key_len] = base_ids[value]
        seq[q_start] = UNK_ID
        seq[q_start + 1:q_start + 1 + key_len] = key
        occupied += [(p_start, p_start + key_len + 2), (q_start, q_start + key_len + 1)]
        positions.append(q_start + key_len)
        targets.append(value)
    return positions, targets


def _finetune(bctx, base_ids) -> None:
    """Fine-tune the model in place on the synthetic format (MQAR protocol). Deterministic
    given args.seed and identical across arms, so the comparison stays paired."""
    args = bctx.args
    key_len = args.needle_key_len
    ctx = args.needle_train_ctx
    max_d = ctx - _LEAD_IN - 2 * key_len - 3
    if getattr(args, "needle_train_max_d", 0):
        max_d = min(max_d, args.needle_train_max_d)
    if max_d < 64:
        raise SkipBenchmark(f"--needle-train-ctx {ctx} too short for a {key_len}bp key")
    loader = _loader(bctx.config, "train", bctx.device)
    if min(loader.shard_lengths) <= ctx + 1:
        raise SkipBenchmark(f"no train shard longer than needle-train-ctx={ctx}")
    rng = np.random.default_rng([args.seed, 0xF17])
    log_lo, log_hi = math.log(64), math.log(max_d)

    # fp32 master weights: in pure bf16 the AdamW updates (~lr in magnitude) sit at bf16's
    # relative-precision floor and mostly round away — the loss never moves. Compute still
    # runs in bf16 via autocast; weights are cast back to the benchmark dtype afterwards.
    bctx.model.float()
    opt = torch.optim.AdamW(bctx.model.parameters(), lr=args.needle_train_lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    bctx.model.train()
    try:
        with torch.enable_grad():
            for step in range(args.needle_train_steps):
                xs, pos_b, pos_t, targets = [], [], [], []
                for b in range(args.needle_train_batch):
                    background, _ = loader.get_batch(1, ctx)
                    seq = background[0].clone()
                    positions, tgts = _plant_training_needles(
                        seq, rng, base_ids, ctx, key_len, args.needle_train_pairs,
                        log_lo, log_hi)
                    xs.append(seq)
                    pos_b += [b] * len(positions)
                    pos_t += positions
                    targets += tgts
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bctx.is_cuda):
                    logits, _ = bctx.model(torch.stack(xs))
                # Loss only at the query positions, restricted to ACGT — teach "a repeated
                # key means copy" without disturbing the language model anywhere else.
                sel = logits[pos_b, pos_t][:, base_ids].float()
                loss = F.cross_entropy(sel, torch.tensor(targets, device=bctx.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 50 == 0 or step == args.needle_train_steps - 1:
                    print(f"  .. needle fine-tune step {step:4d}/{args.needle_train_steps}"
                          f"  loss {loss.item():.4f}  ({len(targets)} queries/step, "
                          f"chance {math.log(4):.4f})", flush=True)
    finally:
        bctx.model.to(dtype=bctx.dtype)
        bctx.model.eval()
        opt.zero_grad(set_to_none=True)
        del opt


def _distances(bctx) -> list[int]:
    raw = getattr(bctx.args, "needle_distances", None)
    if not raw:
        return DEFAULT_DISTANCES
    if isinstance(raw, str):
        return parse_lengths(raw)  # same `8K,1M` suffix grammar as --lengths
    return sorted(set(raw))


def _mutate(seg: torch.Tensor, rng, base_ids: torch.Tensor, n_mut: int) -> torch.Tensor:
    """Return a copy of `seg` with exactly `n_mut` point substitutions at distinct positions.

    Each drawn position is reassigned to one of the three *other* bases, so the mutation count
    is exact rather than a binomial draw that sometimes resamples the original base. Positions
    that are not ACGT in the background (an `N` run) are left alone and redrawn, so a window
    that happens to land in a centromere gap still gets `n_mut` real substitutions.
    """
    out = seg.clone()
    if n_mut <= 0:
        return out
    lut = torch.full((int(base_ids.max().item()) + 1,), -1, dtype=torch.long, device=seg.device)
    lut[base_ids] = torch.arange(4, device=seg.device)
    code = lut[seg.long()]
    eligible = (code >= 0).nonzero(as_tuple=True)[0]
    if eligible.numel() < n_mut:
        raise SkipBenchmark(f"segment has only {eligible.numel()} ACGT bases, need {n_mut}")
    pick = eligible[torch.from_numpy(
        rng.choice(eligible.numel(), size=n_mut, replace=False)).to(seg.device)]
    # +1..+3 mod 4 is uniform over the three alternatives and can never land on the original.
    shift = torch.from_numpy(rng.integers(1, 4, n_mut)).to(seg.device)
    out[pick] = base_ids[(code[pick] + shift) % 4]
    return out


@torch.no_grad()
def _run_copy_distance(bctx, loader, dist: int, seg_len: int, n_trials: int,
                       n_mut: int = 0, base_ids: torch.Tensor | None = None) -> dict:
    """One (ctx, distance) point of the repeat-copy probe. Repeat and control share the same
    background and first segment — a paired design, so the advantage is a per-trial diff.

    `n_mut` substitutions are applied to the *earlier* occurrence only. The scored tokens are
    therefore byte-identical across every mutation level and across repeat/control, so
    copy_advantage stays on one scale and the n_mut sweep reads as a degradation curve: how
    much of the copy benefit survives a degenerate rather than exact match. Mutating the
    scored copy instead would change the target tokens and confound recall with the intrinsic
    entropy of the mutated bases."""
    ctx = bctx.ctx
    lead = ctx - 2 * seg_len - dist
    if lead < _LEAD_IN:
        raise SkipBenchmark(f"distance {dist} does not fit ctx={ctx} with seg={seg_len}")

    diffs, ces_r, ces_c = [], [], []
    for _ in range(n_trials):
        bg, _ = loader.get_batch(1, ctx)
        seg_a, _ = loader.get_batch(1, seg_len)
        # Both sequences end in the SAME segment A and share the same background — they differ
        # only in whether A also occurred `dist` tokens earlier. Scoring identical final tokens
        # is what isolates recall: an earlier version scored *different* segments in the repeat
        # and control windows, and the segments' intrinsic entropy difference (std ~0.3 nats)
        # swamped the recall effect entirely.
        seq_r = bg[0].clone()
        first = seg_a[0] if n_mut == 0 else _mutate(seg_a[0], bctx._needle_rng, base_ids, n_mut)
        seq_r[lead:lead + seg_len] = first         # first occurrence (mutated when n_mut > 0)
        seq_r[ctx - seg_len:] = seg_a[0]           # repeat at the end
        seq_c = bg[0].clone()
        seq_c[ctx - seg_len:] = seg_a[0]           # same ending, no earlier occurrence
        x = torch.stack([seq_r, seq_c])
        logits, _ = bctx.model(x)
        ce = F.cross_entropy(logits[:, -seg_len - 1:-1].float().transpose(1, 2),
                             x[:, -seg_len:], reduction="none").mean(dim=1)
        ces_r.append(ce[0].item())
        ces_c.append(ce[1].item())
        diffs.append(ce[1].item() - ce[0].item())

    diffs = np.asarray(diffs)
    # The mutation level is part of the row identity: with it in `_task`, one output stem can
    # hold the whole n_mut sweep and the resume ledger will not treat m=1 as a redo of m=0.
    return {
        "_task": f"d={dist}" + (f" m={n_mut}" if n_mut else ""),
        # nats/token the repeat is easier than the paired novel segment; 0 = no recall.
        "copy_advantage": round(float(diffs.mean()), 4),
        **bootstrap_mean(diffs, "copy_advantage"),   # paired per-trial diffs are the resample unit
        "stderr": round(float(diffs.std(ddof=1) / math.sqrt(len(diffs))), 4),
        "ce_repeat": round(float(np.mean(ces_r)), 4),
        "ce_control": round(float(np.mean(ces_c)), 4),
        "distance_bp": dist,
        "seg_len": seg_len,
        "n_mutations": n_mut,
        "mut_identity": round(1.0 - n_mut / seg_len, 4),
        "trials": n_trials,
    }


@torch.no_grad()
def _run_one_distance(bctx, loader, base_ids, dist: int, key_len: int, n_trials: int) -> dict:
    """One (ctx, distance) measurement point. Returns a metrics dict for a single row."""
    ctx = bctx.ctx

    # Layout: [_LEAD_IN bg][UNK][key][value][dist bg][UNK][key] -> predict value
    if ctx - dist - 2 * key_len - 3 < _LEAD_IN:
        raise SkipBenchmark(f"distance {dist} does not fit ctx={ctx}")

    correct = 0
    correct_flags, probs_correct = [], []
    for _ in range(n_trials):
        background, _ = loader.get_batch(1, ctx)
        seq = background[0].clone()
        target = _plant(seq, bctx._needle_rng, base_ids, ctx, dist, key_len)

        logits, _ = bctx.model(seq.unsqueeze(0))
        # Restrict to ACGT: PAD/BOS/UNK/N are not admissible answers, and letting them win the
        # argmax would understate recall for reasons that have nothing to do with recall.
        base_logits = logits[0, -1, base_ids].float()
        pred = base_logits.argmax().item()
        correct += int(pred == target)
        correct_flags.append(int(pred == target))
        probs_correct.append(torch.softmax(base_logits, dim=-1)[target].item())

    acc = correct / n_trials
    # Binomial standard error, so a reader can see whether a deviation from 0.25 is signal.
    stderr = math.sqrt(max(acc * (1.0 - acc), 1e-12) / n_trials)
    return {
        "_task": f"d={dist}",
        "recall_acc": round(acc, 4),
        **bootstrap_mean(np.asarray(correct_flags, dtype=float), "recall_acc"),
        "stderr": round(stderr, 4),
        "chance": 0.25,
        # Mean probability mass on the correct base — a continuous signal that moves before
        # argmax accuracy does, so partial recall is visible instead of being rounded to chance.
        "p_correct": round(float(np.mean(probs_correct)), 4),
        "distance_bp": dist,
        "key_len": key_len,
        "trials": n_trials,
    }


@torch.no_grad()
def run(bctx) -> list[dict]:
    ctx = bctx.ctx
    key_len = bctx.args.needle_key_len
    n_trials = bctx.args.needle_trials
    if ctx < 4 * key_len + 64:
        raise SkipBenchmark(f"ctx={ctx} too short for a {key_len}bp key probe")

    loader = _loader(bctx.config, "test", bctx.device)
    if min(loader.shard_lengths) <= ctx + 1:
        raise SkipBenchmark(f"no test shard longer than ctx={ctx}")

    # Seeded from (seed, ctx): identical across arms at the same ctx, so the D01..D07
    # comparison stays paired, but different across context lengths. Seeding from `seed`
    # alone made every ctx row at a given distance reuse the same key/value draws, so rows
    # that look like independent repeats are not — pooling them across ctx then understates
    # the standard error by ~sqrt(#ctx) and manufactures significance that is not there.
    bctx._needle_rng = np.random.default_rng([bctx.args.seed, ctx])
    base_ids = torch.tensor(BASE_IDS, device=bctx.device)

    mode = getattr(bctx.args, "needle_mode", "copy")
    if mode == "copy":
        seg_len = getattr(bctx.args, "needle_seg_len", 256)
        if ctx < 2 * seg_len + _LEAD_IN + 64:
            raise SkipBenchmark(f"ctx={ctx} too short for two {seg_len}bp segments")
        n_mut = int(getattr(bctx.args, "needle_copy_mutations", 0) or 0)
        if n_mut >= seg_len:
            raise SkipBenchmark(f"--needle-copy-mutations {n_mut} >= seg_len {seg_len}")
        rows = []
        for dist in _distances(bctx):
            try:
                row = _run_copy_distance(bctx, loader, dist, seg_len,
                                         bctx.args.needle_trials,
                                         n_mut=n_mut, base_ids=base_ids)
                row["mode"] = mode
                rows.append(row)
            except SkipBenchmark as exc:
                rows.append({"_task": f"d={dist}", "_status": "SKIP: data unavailable",
                             "_note": str(exc)})
        return rows

    orig_state = _snapshot(bctx.model) if mode == "trained" else None
    try:
        if mode == "trained":
            # Fine-tune once per suite run; the state is cached on the args namespace so
            # every context length in the sweep reuses the same trained weights.
            ft_state = getattr(bctx.args, "_needle_ft_state", None)
            if ft_state is None:
                print(f"  .. fine-tuning on the synthetic format "
                      f"({bctx.args.needle_train_steps} steps @ ctx="
                      f"{bctx.args.needle_train_ctx}; cached for the rest of the sweep)",
                      flush=True)
                _finetune(bctx, base_ids)
                ft_state = _snapshot(bctx.model)
                bctx.args._needle_ft_state = ft_state
            else:
                bctx.model.load_state_dict(ft_state)

        rows = []
        for dist in _distances(bctx):
            try:
                row = _run_one_distance(bctx, loader, base_ids, dist, key_len, n_trials)
                row["mode"] = mode
                rows.append(row)
            except SkipBenchmark as exc:
                rows.append({"_task": f"d={dist}", "_status": "SKIP: data unavailable",
                             "_note": str(exc)})
        return rows
    finally:
        if orig_state is not None:
            # Every other benchmark must see the pristine checkpoint.
            bctx.model.load_state_dict(orig_state)
