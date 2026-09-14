"""Dispatch for the fixed-length classification suites: frozen linear probe (default) or full
fine-tuning, selected by `--downstream-mode`. The three task modules call `classify_task`
so they never need to know which one is active; the mode is recorded in every row."""
from benchmarks.probe import probe_task
from benchmarks.finetune import finetune_task


def classify_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx: int, label: str = "") -> dict:
    mode = getattr(bctx.args, "downstream_mode", "probe")
    if mode == "finetune":
        metrics = finetune_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx,
                                batch_size=bctx.args.ft_batch_size, label=label)
    elif mode == "probe":
        metrics = probe_task(bctx, train_seqs, train_y, test_seqs, test_y, ctx,
                             batch_size=bctx.args.probe_batch_size)
    else:
        raise ValueError(f"unknown --downstream-mode {mode!r}")
    metrics["mode"] = mode
    return metrics
