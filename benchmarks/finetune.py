"""Full fine-tuning for the fixed-length downstream suites (Genomic Benchmarks, NT, GUE).

This is the `--downstream-mode finetune` counterpart of benchmarks/probe.py. The probe keeps
the backbone frozen and is a *lower bound*; HyenaDNA / NT / DNABERT-2 all report numbers
after fine-tuning the whole model, so a table meant to sit next to theirs has to come from
here. Both modes share the same data loading, split handling and subsampling — only the
classifier differs — so `probe` and `finetune` rows are directly comparable to each other.

Protocol (identical across arms, so the architecture comparison stays controlled):
  - classifier = backbone (all parameters trainable) + a linear head on, per strand,
    [masked mean | last real token | `--ft-bins` positional-bin means] of the final hidden
    state. With `--ft-rc` (default on) every sequence is encoded twice, forward and as its
    reverse complement, and the two feature vectors are concatenated: 2 * (2 + bins) * d_model.
    Rationale (2026-08-29 diagnosis): with the old [mean | last] head every arm scored the
    bag-of-k-mers ceiling on NT promoter (0.85, random init == pretrained) and *below* a
    linear PWM on splice sites (0.62 vs 0.83) — a causal model reading out 300bp downstream
    of the annotated locus through a decaying state, plus a mean that dilutes the locus
    150x, loses "where in the window". The bins restore position, the reverse-complement
    pass gives the head a view in which the locus is close to the read-out end.
  - AdamW, separate LRs for backbone (`--ft-lr`) and head (`--ft-head-lr`), 5% linear
    warmup then cosine to 10%, grad-clip 1.0, bf16 autocast with fp32 master weights
    (pure-bf16 parameters silently freeze small-gradient weights — see scripts/train.py).
  - a stratified 10% slice of the *training* split is the dev set; the epoch with the best dev
    accuracy is the one scored on test. The test split is never touched during training.
  - the pretrained weights are restored after every task, so later tasks (and later benchmarks
    in the same run) always start from the same checkpoint.

Sequences are right-padded; every arm here is causal and the pooling is masked, so padding
never leaks into the features.

Memory: `--ft-batch-size` is the *effective* batch (optimizer step). It is processed as
micro-batches of `--ft-micro-batch` sequences with gradient accumulation, and the micro-batch
is halved automatically on a CUDA OOM (down to 1) — necessary because the k-mer-memory arms
(D05/D06) carry a per-sequence table (~0.6 GB per sequence for D06 in training mode), so
their feasible micro-batch on 8 GB is ~8 regardless of how short the sequences are.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from benchmarks.bootstrap import bootstrap_classification
from benchmarks.probe import _f1
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from benchmarks.model_io import reverse_complement, tokenize_batch


class _Head(nn.Module):
    """Linear head on [mean | last | n_bins positional-bin means] per strand."""

    def __init__(self, d_model: int, n_classes: int, n_bins: int = 8, rc: bool = True):
        super().__init__()
        self.n_bins, self.rc = n_bins, rc
        self.proj = nn.Linear((2 if rc else 1) * (2 + n_bins) * d_model, n_classes)

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = h.float()
        m = mask.unsqueeze(-1).float()
        lengths = mask.sum(1).clamp(min=1)
        mean = (h * m).sum(1) / lengths.unsqueeze(-1).float()
        last = h[torch.arange(h.shape[0], device=h.device), lengths - 1]
        feats = [mean, last]
        if self.n_bins:
            B, T, d = h.shape
            pos = torch.arange(T, device=h.device).unsqueeze(0)
            bin_idx = ((pos * self.n_bins) // lengths.unsqueeze(1)).clamp(max=self.n_bins - 1)
            onehot = F.one_hot(bin_idx, self.n_bins).float() * m
            bins = torch.einsum("btk,btd->bkd", onehot, h) / onehot.sum(1).clamp(min=1.0).unsqueeze(-1)
            feats.append(bins.reshape(B, -1))
        return torch.cat(feats, dim=1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`h` is (2B, T, d) when rc: forward views first, reverse-complement views second."""
        if self.rc:
            B = mask.shape[0]
            f = torch.cat([self._pool(h[:B], mask), self._pool(h[B:], mask)], dim=1)
        else:
            f = self._pool(h, mask)
        return self.proj(f)


def _encode_views(model, head, ids, mask):
    """Run the backbone on the forward strand and, if the head wants it, the reverse complement
    (stacked along the batch dim so the OOM micro-batch halving still bounds memory)."""
    if head.rc:
        ids = torch.cat([ids, reverse_complement(ids, mask)], dim=0)
    return model.encode(ids)


def _batches(seqs, labels, batch_size, ctx, device, shuffle, rng):
    order = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(order)
    else:  # length-sort for eval so batches carry little padding
        order = np.array(sorted(order, key=lambda i: len(seqs[i])))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        ids, mask = tokenize_batch([seqs[i] for i in idx], ctx, device)
        yield idx, ids, mask, torch.as_tensor(labels[idx], device=device, dtype=torch.long)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


@torch.no_grad()
def _predict(model, head, seqs, labels, micro, ctx, device, use_autocast):
    model.eval()
    logits = np.zeros((len(seqs), head.proj.out_features), dtype=np.float32)
    for idx, ids, mask, _ in _batches(seqs, labels, micro, ctx, device, False, None):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            h = _encode_views(model, head, ids, mask)
        logits[idx] = head(h, mask).float().cpu().numpy()
    return logits


def _step_loss(model, head, seqs, labels, idx, micro, ctx, device, use_autocast):
    """Accumulate gradients over one effective batch (`idx`) in micro-batches of `micro`.
    Returns the mean loss. Raises the OOM through so the caller can shrink `micro`."""
    total, n = 0.0, len(idx)
    for start in range(0, n, micro):
        sub = idx[start:start + micro]
        ids, mask = tokenize_batch([seqs[i] for i in sub], ctx, device)
        y = torch.as_tensor(labels[sub], device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            h = _encode_views(model, head, ids, mask)
        loss = F.cross_entropy(head(h, mask), y)
        (loss * (len(sub) / n)).backward()
        total += loss.item() * len(sub)
    return total / n


def _score(logits, y) -> dict:
    pred = logits.argmax(1)
    out = {"acc": round(float(accuracy_score(y, pred)), 4),
           "mcc": round(float(matthews_corrcoef(y, pred)), 4),
           "f1": _f1(y, pred)}
    prob = torch.softmax(torch.from_numpy(np.asarray(logits, dtype=np.float32)), dim=1).numpy()
    try:
        if logits.shape[1] == 2:
            out["auroc"] = round(float(roc_auc_score(y, prob[:, 1])), 4)
        elif len(np.unique(y)) == logits.shape[1]:
            out["auroc"] = round(float(roc_auc_score(y, prob, multi_class="ovr")), 4)
    except ValueError:
        pass
    score = prob[:, 1] if (logits.ndim == 2 and logits.shape[1] == 2) else logits
    out.update(bootstrap_classification(y, pred, score=score))
    # Collapse diagnostics. A multi-class fine-tune that diverges does not degrade gracefully —
    # it predicts one class for everything, which acc alone cannot distinguish from a model
    # that learned the majority class. `pred_entropy` 0 with `pred_top_share` 1.0 IS collapse;
    # comparing acc to `majority_acc` says whether the run beat the label prior at all.
    counts = np.bincount(pred, minlength=logits.shape[1]).astype(np.float64)
    q = counts[counts > 0] / counts.sum()
    out["pred_top_share"] = round(float(counts.max() / counts.sum()), 4)
    out["pred_classes_used"] = int((counts > 0).sum())
    out["pred_entropy"] = round(float(-(q * np.log(q)).sum()), 4)
    out["majority_acc"] = round(float(np.bincount(np.asarray(y)).max() / len(y)), 4)
    out["collapsed"] = bool(out["pred_classes_used"] == 1)
    return out


def finetune_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx: int,
                  batch_size: int = 32, label: str = "") -> dict:
    """Fine-tune the whole model on one task and score the test split with the best-dev epoch.
    Restores the pretrained weights (and the benchmark dtype) before returning."""
    args, model, device = bctx.args, bctx.model, bctx.device
    train_y, test_y = np.asarray(train_y), np.asarray(test_y)
    n_classes = int(max(train_y.max(), test_y.max())) + 1
    use_autocast = bctx.is_cuda

    # --- pristine weights: every task starts from the same checkpoint --------------------
    orig_dtype = next(model.parameters()).dtype
    snapshot = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    was_ckpt = getattr(model, "gradient_checkpointing", False)
    req_grad = {n: p.requires_grad for n, p in model.named_parameters()}

    # --- dev split from TRAIN only ---------------------------------------------------------
    n = len(train_y)
    try:
        tr, dev = train_test_split(np.arange(n), test_size=0.1, random_state=args.seed,
                                   stratify=train_y)
    except ValueError:
        tr, dev = train_test_split(np.arange(n), test_size=0.1, random_state=args.seed)
    tr_seqs, tr_y = [train_seqs[i] for i in tr], train_y[tr]
    dev_seqs, dev_y = [train_seqs[i] for i in dev], train_y[dev]

    # --- model in fp32 master weights, trainable ------------------------------------------
    model.float()
    model.gradient_checkpointing = bool(args.ft_grad_checkpoint)
    for p in model.parameters():
        p.requires_grad_(True)
    head = _Head(model.config["d_model"], n_classes, n_bins=int(getattr(args, "ft_bins", 8)),
                 rc=bool(getattr(args, "ft_rc", True))).to(device)

    steps_per_epoch = math.ceil(len(tr_seqs) / batch_size)
    total_steps = steps_per_epoch * args.ft_epochs
    if args.ft_max_steps and total_steps > args.ft_max_steps:
        total_steps = args.ft_max_steps
    epochs = math.ceil(total_steps / steps_per_epoch)
    warmup = max(1, int(0.05 * total_steps))

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if getattr(p, "_no_weight_decay", False) or p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.ft_lr, "weight_decay": args.ft_weight_decay},
        {"params": no_decay, "lr": args.ft_lr, "weight_decay": 0.0},
        {"params": head.parameters(), "lr": args.ft_head_lr, "weight_decay": 0.0},
    ])

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    rng = np.random.default_rng([args.seed, len(train_seqs), ctx])
    best = {"dev_acc": -1.0, "epoch": 0, "model": None, "head": None}
    micro = max(1, min(batch_size, int(args.ft_micro_batch or batch_size)))
    step, t0 = 0, time.time()
    pbar = tqdm(total=total_steps, desc=f"  ft {label}".rstrip(), unit="step",
                dynamic_ncols=True, leave=False)
    try:
        for epoch in range(epochs):
            model.train()
            head.train()
            order = np.arange(len(tr_seqs))
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                if step >= total_steps:
                    break
                idx = order[start:start + batch_size]
                while True:
                    opt.zero_grad(set_to_none=True)
                    try:
                        loss_val = _step_loss(model, head, tr_seqs, tr_y, idx, micro, ctx,
                                              device, use_autocast)
                        break
                    except Exception as exc:  # noqa: BLE001
                        if not _is_oom(exc) or micro == 1:
                            raise
                        micro = max(1, micro // 2)
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        pbar.write(f"    OOM -> micro-batch {micro} (effective batch stays {batch_size})")
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
                opt.step()
                sched.step()
                step += 1
                pbar.set_postfix(loss=f"{loss_val:.3f}", ep=epoch + 1, mb=micro)
                pbar.update(1)

            dev_logits = _predict(model, head, dev_seqs, dev_y, micro, ctx, device, use_autocast)
            dev_acc = float(accuracy_score(dev_y, dev_logits.argmax(1)))
            pbar.write(f"    epoch {epoch + 1}/{epochs}  dev_acc {dev_acc:.4f}  "
                       f"[{time.time() - t0:.0f}s]")
            if dev_acc > best["dev_acc"]:
                best = {"dev_acc": dev_acc, "epoch": epoch + 1,
                        "model": copy.deepcopy(model.state_dict()),
                        "head": copy.deepcopy(head.state_dict())}
            if step >= total_steps:
                break

        model.load_state_dict(best["model"])
        head.load_state_dict(best["head"])
        test_logits = _predict(model, head, test_seqs, test_y, micro, ctx, device, use_autocast)
        metrics = _score(test_logits, test_y)
        metrics.update({
            "ft_dev_acc": round(best["dev_acc"], 4),
            "ft_best_epoch": best["epoch"],
            "ft_epochs": epochs,
            "ft_steps": step,
            "ft_batch": batch_size,
            "ft_micro_batch": micro,
            "ft_seconds": round(time.time() - t0, 1),
            "n_train": int(len(train_y)),
            "n_test": int(len(test_y)),
            "seq_len": ctx,
            "ft_head": f"mean+last+{head.n_bins}bins" + ("+rc" if head.rc else ""),
        })
    finally:
        pbar.close()
        # restore the pretrained checkpoint exactly, in the benchmark's dtype
        model.load_state_dict({k: v.to(device) for k, v in snapshot.items()})
        model.to(dtype=orig_dtype)
        model.gradient_checkpointing = was_ckpt
        for n, p in model.named_parameters():
            p.requires_grad_(req_grad[n])
        model.eval()
        del head, opt, best
        if bctx.is_cuda:
            torch.cuda.empty_cache()
    return metrics
