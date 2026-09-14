"""HyenaDNA-style embedding scatter: t-SNE of frozen pooled embeddings, coloured by class.

Mirrors HyenaDNA Fig. 4.3 (pretrained embeddings, mean-pooled, t-SNE, one colour per class),
but through *our* harness: same tokenizer, same `embed_sequences` readout, same seeds as
benchmarks/, so panels for different arms are directly comparable.

    uv run python analysis/embedding_viz.py --config configs/d06_kmer_fastweight.yaml \
        --checkpoint results/d06/step_34177.pt --source species --ctx 8192

    # one panel per arm, same points, same layout
    uv run python analysis/embedding_viz.py --source gb:human_vs_worm \
        --arm configs/d01_kda.yaml:results/d01/final.pt \
        --arm configs/d06_kmer_fastweight.yaml:results/d06/step_34177.pt \
        --arm configs/d07_kmer_local.yaml:results/d07/step_34000.pt

Sources: `species` (windows from data/species/*.fa, length-swept) or `gb:<task>` / `nt:<task>`
(task-native length). Pooling defaults to HyenaDNA's readout (mean over the final layer);
`--pool rich` uses the probe's multi-layer readout instead.
"""
import argparse
from pathlib import Path

import numpy as np

from benchmarks.model_io import embed_sequences, load_model, resolve_dtype
from training.config import load_config

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


# ---------------------------------------------------------------- data sources
def load_species(ctx, per_class, seed, species_dir="data/species"):
    from benchmarks.fasta import Fasta
    from benchmarks.species import _sample_windows

    rng = np.random.default_rng(seed)
    paths = sorted(p for p in Path(species_dir).glob("*") if p.suffix in (".fa", ".fasta", ".fna"))
    if len(paths) < 2:
        raise SystemExit(f"need >=2 genome FASTAs in {species_dir}, found {len(paths)}")
    seqs, labels, names = [], [], []
    for label, path in enumerate(paths):
        fasta = Fasta(path)
        try:
            seqs.extend(_sample_windows(fasta, per_class, ctx, rng))
        finally:
            fasta.close()
        labels.extend([label] * per_class)
        names.append(path.stem)
    return seqs, np.array(labels), names, ctx


def load_hf_task(source, n, seed):
    """gb:<task> / nt:<task> -> (seqs, labels, class names, native length)."""
    from datasets import load_dataset
    kind, _, task = source.partition(":")
    if kind == "gb":
        from benchmarks.genomic_benchmarks import REPO, TASKS
        match = [t for t in TASKS if t[0] == task]
        if not match:
            raise SystemExit(f"unknown gb task {task!r}; have {[t[0] for t in TASKS]}")
        short, hf_name, native = match[0]
        ds = load_dataset(f"{REPO}/{hf_name}")["test"]
    else:
        from benchmarks.nt_bench import ALL_TASKS, _load
        if task not in ALL_TASKS:
            raise SystemExit(f"unknown nt task {task!r}; have {ALL_TASKS}")
        ds = _load(task)["test"]
        native = 600  # NT tasks are 200-600bp

    seq_key = next(c for c in ("sequence", "seq", "dna") if c in ds.column_names)
    lab_key = next(c for c in ("label", "labels", "target") if c in ds.column_names)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n]
    rows = ds.select(idx)
    labels = np.asarray(rows[lab_key])
    names = [f"class {c}" for c in sorted(set(labels.tolist()))]
    return list(rows[seq_key]), labels, names, native


# ---------------------------------------------------------------- projection
def project(x, method, seed, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 50:                      # standard t-SNE preprocessing
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    from sklearn.manifold import TSNE
    perp = min(perplexity, max(5.0, (len(x) - 1) / 3))
    return TSNE(n_components=2, random_state=seed, perplexity=perp,
                init="pca", learning_rate="auto").fit_transform(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config"), ap.add_argument("--checkpoint")
    ap.add_argument("--arm", action="append", default=[],
                    help="config.yaml:checkpoint.pt — repeatable, one panel each")
    ap.add_argument("--source", default="species", help="species | gb:<task> | nt:<task>")
    ap.add_argument("--ctx", type=int, default=8192, help="species source only")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=1500, help="gb/nt sources")
    ap.add_argument("--pool", default="mean", choices=["mean", "rich", "last"])
    ap.add_argument("--method", default="tsne", choices=["tsne", "pca"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda"), ap.add_argument("--precision", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    arms = [tuple(a.split(":", 1)) for a in args.arm]
    if args.config:
        arms.insert(0, (args.config, args.checkpoint))
    if not arms:
        raise SystemExit("give --config/--checkpoint or one or more --arm cfg:ckpt")

    if args.source == "species":
        seqs, labels, names, ctx = load_species(args.ctx, args.n_per_class, args.seed)
    else:
        seqs, labels, names, ctx = load_hf_task(args.source, args.n_samples, args.seed)
    print(f"{len(seqs)} sequences, {len(names)} classes, ctx={ctx}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms) + 1.6, 4.4), squeeze=False)
    for ax, (cfg_path, ckpt) in zip(axes[0], arms):
        config = load_config(cfg_path)
        dtype = resolve_dtype(args.device, args.precision)
        model, step = load_model(config, ckpt, args.device, dtype)
        x = embed_sequences(model, seqs, ctx=ctx, device=args.device,
                            batch_size=args.batch_size, pool=args.pool,
                            layers=None if args.pool == "mean" else (-3, -2, -1))
        del model
        import torch; torch.cuda.empty_cache()

        xy = project(x, args.method, args.seed, args.perplexity)
        for label, name in enumerate(names):
            sel = labels == label
            ax.scatter(xy[sel, 0], xy[sel, 1], s=9, alpha=0.75, linewidths=0,
                       color=PALETTE[label % len(PALETTE)], label=name)
        title = config.get("run_name", config["model_id"])
        ax.set_title(f"{title}" + (f"  (step {step})" if step else "  (random init)"), fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                       markerscale=2.0, fontsize=9)
    fig.suptitle(f"{args.method.upper()} of {args.pool}-pooled embeddings — "
                 f"{args.source} @ {ctx}bp", fontsize=11)
    fig.tight_layout()

    out = args.out or f"paper/figures/embed_{args.method}_{args.source.replace(':', '_')}_{ctx}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
