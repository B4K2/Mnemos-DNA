"""Generate the figures for the DNA-KDA paper (paper/figures/*.svg).

Reads only existing result files (results/*/benchmark_cuda*.json, panel_cuda.json,
benchmark_cuda_finetune.json) and mlflow.db. Writes SVGs to paper/figures/.
New file added for the paper — does not modify any experiment code.

Usage: uv run python paper/make_figures.py
"""
import json
import math
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette (validated, dataviz skill): entity -> hue, fixed
C = {
    "d06": "#2a78d6",
    "d07": "#eb6834",
    "d01": "#8a63d2",
    "d02": "#6B6A64",
    "hyenadna": "#1baf7a",
    "caduceus": "#e87ba4",
    "ink": "#1C1B18",
    "muted": "#6B6A64",
    "rule": "#E4E2DB",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": C["muted"],
    "axes.labelcolor": C["ink"],
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": C["muted"],
    "ytick.color": C["muted"],
    "xtick.labelcolor": C["ink"],
    "ytick.labelcolor": C["ink"],
    "grid.color": C["rule"],
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})


def rows(path, bench):
    d = json.load(open(ROOT / path))
    return [r for r in d["rows"] if r["benchmark"] == bench]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    fig.savefig((OUT / name).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def kfmt(n):
    return f"{n//1024}K" if n >= 1024 else str(n)


# ---------------------------------------------------------------- fig: training curves
def fig_curves():
    con = sqlite3.connect(ROOT / "mlflow.db")
    rid = {m: open(ROOT / f"results/{m}/mlflow_run_id.txt").read().strip() for m in ("d06", "d07")}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    series = [
        ("d06", "val_loss", C["d06"], "-", "Mnemos-DNA val (131K ctx)"),
        ("d07", "val_loss", C["d07"], "-", "Mnemos-DNA-C val (131K ctx)"),
        ("d07", "val_loss_short", C["d07"], (0, (2, 2)), "Mnemos-DNA-C val (1K ctx)"),
    ]
    for m, key, col, ls, lab in series:
        pts = con.execute(
            "select step,value from metrics where run_uuid=? and key=? order by step",
            (rid[m], key)).fetchall()
        pts = [(s, v) for s, v in pts if s >= 400]
        ax.plot([s / 1000 for s, _ in pts], [v for _, v in pts],
                color=col, ls=ls, lw=1.6, label=lab)
        ax.annotate(f"{pts[-1][1]:.3f}", (pts[-1][0] / 1000, pts[-1][1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=col, fontsize=8, va="center")
    ax.set_xlabel("training step (thousands; 131,072 tokens per step)")
    ax.set_ylabel("validation loss (nats/base)")
    ax.set_xlim(0, 40)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    save(fig, "fig_training_curves.svg")


# ---------------------------------------------------------------- fig: needle
def fig_needle():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    arms = [
        ("d01", "results/d01_new/benchmark_cuda.json", C["d01"], "Pure-KDA (D01) · matched control"),
        ("d02", "results/d02/needle_copy.json", C["d02"], "D02 · KDA + local window"),
        ("d06", "results/d06/benchmark_cuda_recovered_step34177.json", C["d06"], "Mnemos-DNA · KDA + k-mer fast-weight memory"),
        ("d07", "results/d07/benchmark_cuda.json", C["d07"], "Mnemos-DNA-C · + local window + curriculum"),
    ]
    for i, (mid, path, col, lab) in enumerate(arms):
        rs = [r for r in rows(path, "needle")
              if r["status"].startswith("ok") and r["ctx"] == 131072]
        rs = sorted(rs, key=lambda r: r["metrics"]["distance_bp"])
        xs = [r["metrics"]["distance_bp"] for r in rs]
        ys = [r["metrics"]["copy_advantage"] for r in rs]
        ax.plot(xs, ys, "o-", color=col, lw=1.8, ms=4.5, label=lab)
        # d01/d02 both sit at ~0 at 64K; stagger their end labels vertically
        dy = {"d01": 8, "d02": -8}.get(mid, 0)
        ax.annotate(lab.split(" · ")[0], (xs[-1], ys[-1]), xytext=(6, dy),
                    textcoords="offset points", color=col, fontsize=8.5, va="center")
    ax.axhline(0, color=C["muted"], lw=0.8, ls=":")
    ax.text(70, 0.015, "no recall", color=C["muted"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([64, 256, 1024, 4096, 16384, 65536])
    ax.xaxis.set_major_locator(FixedLocator([64, 256, 1024, 4096, 16384, 65536]))
    ax.set_xticklabels(["64", "256", "1K", "4K", "16K", "64K"])
    ax.set_xlabel("needle→query distance (bp), context 131,072")
    ax.set_ylabel("copy advantage (Δ cross-entropy, nats)")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(axis="y")
    ax.legend(loc="center left", fontsize=8)
    save(fig, "fig_needle.svg")


# ---------------------------------------------------------------- fig: vram scaling
def fig_scaling():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    d07 = [r for r in rows("results/d07/benchmark_cuda.json", "scaling") if r["status"] == "ok"]
    xs = [r["ctx"] for r in d07]
    inf = [r["metrics"]["infer_peak_gb"] for r in d07]
    trn = [(r["ctx"], r["metrics"].get("train_peak_gb")) for r in d07 if r["metrics"].get("train_peak_gb")]
    ax.plot(xs, inf, "o-", color=C["d07"], lw=1.8, ms=4.5, label="Mnemos-DNA(-C) inference")
    ax.plot([c for c, _ in trn], [v for _, v in trn], "s--", color=C["d07"],
            lw=1.4, ms=4, alpha=0.7, label="Mnemos-DNA(-C) training step")
    ax.annotate("Mnemos-DNA(-C) prefill\nreaches 1M bp (5.6 GB)", (xs[-1], inf[-1]), xytext=(-8, 16),
                textcoords="offset points", ha="right", color=C["d07"], fontsize=8.5,
                linespacing=1.35)
    # Released checkpoints, batch-1 inference (scaling_b1): their real ceilings.
    externals = [
        ("results/hyenadna_1m/scaling_b1.json", C["hyenadna"], "HyenaDNA-1M ckpt (OOM at 512K)"),
        ("results/caduceus_ps/scaling_b1.json", C["caduceus"], "Caduceus-PS ckpt (OOM at 1M)"),
    ]
    for path, col, lab in externals:
        ext = [r for r in rows(path, "scaling") if r["status"] == "ok"]
        exs = [r["ctx"] for r in ext]
        eys = [r["metrics"]["infer_peak_gb"] for r in ext]
        ax.plot(exs, eys, "^-", color=col, lw=1.5, ms=4, alpha=0.85, label=lab)
        ax.plot([exs[-1]], [eys[-1]], "x", color=col, ms=9, mew=2)
    ax.axhline(8, color=C["ink"], lw=1.0, ls=":")
    ax.text(xs[-1], 8.15, "RTX 5060 \u00b7 8 GB", color=C["ink"], fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.set_xticklabels([kfmt(x) for x in xs])
    ax.set_ylim(0, 9.2)
    ax.set_xlabel("sequence length (bp)")
    ax.set_ylabel("peak VRAM (GB)")
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.88))
    save(fig, "fig_scaling.svg")


# ---------------------------------------------------------------- fig: downstream dumbbells
NT_GB = [
    ("nt", "splice_sites_donors", "NT splice sites (donors)"),
    ("nt", "splice_sites_all", "NT splice sites (all)"),
    ("nt", "promoter_all", "NT promoter (all)"),
    ("nt", "promoter_no_tata", "NT promoter (no TATA)"),
    ("nt", "enhancers", "NT enhancers"),
    ("nt", "H3K4me3", "NT H3K4me3"),
    ("genomic_benchmarks", "mouse_enhancers", "GB mouse enhancers"),
    ("genomic_benchmarks", "human_ocr_ensembl", "GB human OCR (Ensembl)"),
    ("genomic_benchmarks", "human_regulatory", "GB human regulatory"),
    ("genomic_benchmarks", "human_enhancers_ensembl", "GB human enhancers (Ensembl)"),
    ("genomic_benchmarks", "human_nontata_promoters", "GB non-TATA promoters"),
    ("genomic_benchmarks", "human_vs_worm", "GB human vs worm"),
    ("genomic_benchmarks", "coding_vs_intergenomic", "GB coding vs intergenomic"),
    ("genomic_benchmarks", "human_enhancers_cohn", "GB human enhancers (Cohn)"),
]


def ft_table(mid):
    d = json.load(open(ROOT / f"results/{mid}/benchmark_cuda_finetune.json"))
    return {(r["benchmark"], r["task"]): r["metrics"] for r in d["rows"] if r["status"] == "ok"}


def ft_seeds(mid, stems=("ft_seed0", "ft_seed1", "ft_seed2")):
    """Matched-protocol fine-tune accuracies across seeds: (bench, task) -> list of acc.

    d06's seed-0 GB rows were archived by the fingerprint guard, so some tasks have 2 seeds
    and some 3; the per-task n is carried into the table."""
    per = {}
    for stem in stems:
        p = ROOT / f"results/{mid}/{stem}.json"
        if not p.exists():
            continue
        for r in json.load(open(p))["rows"]:
            if r.get("status") == "ok" and r["metrics"].get("acc") is not None:
                per.setdefault((r["benchmark"], r["task"]), []).append(r["metrics"]["acc"])
    return per


def mean(v):
    return sum(v) / len(v)


def std(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def fig_downstream():
    a, b = ft_seeds("d06"), ft_seeds("d07")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    items = NT_GB[::-1]
    for i, (bn, task, lab) in enumerate(items):
        x0, x1 = mean(a[(bn, task)]), mean(b[(bn, task)])
        ax.plot([x0, x1], [i, i], color=C["rule"], lw=2, zorder=1)
        for x, vals, col in ((x0, a[(bn, task)], C["d06"]), (x1, b[(bn, task)], C["d07"])):
            s = std(vals)
            if s == s:  # not nan
                ax.plot([x - s, x + s], [i, i], color=col, lw=1.1, alpha=0.55, zorder=2)
        ax.plot([x0], [i], "o", color=C["d06"], ms=6, zorder=3)
        ax.plot([x1], [i], "o", color=C["d07"], ms=6, zorder=3)
        ax.annotate(f"{x1 - x0:+.2f}".replace("+0.00", "±0.00"), (max(x0, x1), i),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=C["ink"] if abs(x1 - x0) > .02 else C["muted"])
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab for _, _, lab in items], fontsize=8.5)
    ax.set_xlim(0.55, 1.02)
    ax.set_xlabel("fine-tune test accuracy")
    ax.grid(axis="x")
    ax.plot([], [], "o", color=C["d06"], label="Mnemos-DNA")
    ax.plot([], [], "o", color=C["d07"], label="Mnemos-DNA-C (+local window, +curriculum)")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "fig_downstream.svg")


# ---------------------------------------------------------------- fig: GUE deltas
def fig_gue():
    # Matched protocol: d06 from the ft_seed0_gue rerun; d07's main GUE run already used it.
    a = {(r["benchmark"], r["task"]): r["metrics"]
         for r in json.load(open(ROOT / "results/d06/ft_seed0_gue.json"))["rows"]
         if r["status"] == "ok"}
    b = ft_table("d07")
    keys = sorted([k for k in a if k[0] == "gue" and k in b],
                  key=lambda k: b[k]["acc"] - a[k]["acc"])
    labs = [k[1].replace("GUE/", "") for k in keys]
    dels = [b[k]["acc"] - a[k]["acc"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cols = [C["d07"] if d > 0 else C["d06"] for d in dels]
    ax.barh(range(len(keys)), dels, color=cols, height=0.62)
    ax.axvline(0, color=C["muted"], lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlabel("accuracy change, Mnemos-DNA-C − Mnemos-DNA (GUE fine-tune)")
    ax.grid(axis="x")
    save(fig, "fig_gue_delta.svg")


# ---------------------------------------------------------------- fig: species vs ctx
def fig_species():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # batch-1 embedding (species_b1): the ceilings are single-sequence, not probe-batch-8
    arms = [
        ("results/d06/species_b1.json", C["d06"], "Mnemos-DNA"),
        ("results/d07/species_b1.json", C["d07"], "Mnemos-DNA-C"),
        ("results/hyenadna_1m/species_b1.json", C["hyenadna"], "HyenaDNA-1M (pretrained ckpt)"),
        ("results/caduceus_ph/species_b1.json", C["caduceus"], "Caduceus-PH (pretrained ckpt)"),
    ]
    for path, col, lab in arms:
        rs = [r for r in rows(path, "species") if r["status"].startswith("ok")]
        xs = [r["ctx"] for r in rs]
        ys = [r["metrics"]["acc"] for r in rs]
        lo = [r["metrics"]["acc_lo"] for r in rs]
        hi = [r["metrics"]["acc_hi"] for r in rs]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(xs, ys, "o-", color=col, lw=1.7, ms=4, label=lab)
        oom = [r["ctx"] for r in rows(path, "species") if r["status"] == "SKIP: OOM"]
        if oom:
            ax.plot([xs[-1]], [ys[-1]], "x", color=col, ms=9, mew=2)
            ax.annotate("OOM →", (xs[-1], ys[-1]), xytext=(8, -11),
                        textcoords="offset points", color=col, fontsize=8)
    ax.set_xscale("log")
    ticks = [8192, 16384, 32768, 66560, 134144, 268288, 536576, 1073152]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(["8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"])
    ax.set_ylim(0.9, 1.005)
    ax.set_xlabel("input length (bp)")
    ax.set_ylabel("species classification accuracy\n(frozen linear probe)")
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_species.svg")


# ---------------------------------------------------------------- fig: lm loss vs eval context
LM_ARMS = [
    ("results/d01_new/benchmark_cuda.json", C["d01"], "Pure-KDA (D01) · matched control"),
    ("results/d02/benchmark_cuda.json", "#6B6A64", "D02 · KDA + local window"),
    ("results/d06/benchmark_cuda_recovered_step34177.json", C["d06"], "Mnemos-DNA · + k-mer fast weights"),
    ("results/d07/benchmark_cuda.json", C["d07"], "Mnemos-DNA-C · + local window + curriculum"),
]


def fig_lm_ctx():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.0), sharex=True)
    for path, col, lab in LM_ARMS:
        rs = [r for r in rows(path, "lm") if r["status"].startswith("ok")]
        xs = [r["ctx"] for r in rs]
        for a, key in ((ax, "ce"), (ax2, "acc")):
            ys = [r["metrics"][key] for r in rs]
            lo = [r["metrics"].get(key + "_lo") for r in rs]
            hi = [r["metrics"].get(key + "_hi") for r in rs]
            if all(v is not None for v in lo + hi):
                a.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
            a.plot(xs, ys, "o-", color=col, lw=1.6, ms=3.8, label=lab)
    ticks = [8192, 32768, 131072, 524288, 1048576]
    for a in (ax, ax2):
        a.set_xscale("log")
        a.xaxis.set_major_locator(FixedLocator(ticks))
        a.set_xticklabels(["8K", "32K", "128K", "512K", "1M"])
        a.set_xlabel("evaluation context (bp)")
        a.grid(axis="y")
    ax.set_ylabel("held-out cross-entropy\n(nats/base)")
    ax.set_ylim(0.98, 1.16)
    ax2.set_ylabel("next-nucleotide accuracy")
    ax2.set_ylim(0.46, 0.55)
    ax2.axhline(0.25, color=C["rule"], lw=0.8)
    ax.legend(loc="upper left", fontsize=7.2, ncol=1)
    fig.tight_layout()
    save(fig, "fig_lm_ctx.svg")


# ---------------------------------------------------------------- fig: throughput
def fig_throughput():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    d07 = [r for r in rows("results/d07/benchmark_cuda.json", "scaling") if r["status"] == "ok"]
    d01 = [r for r in rows("results/d01_new/benchmark_cuda.json", "scaling") if r["status"] == "ok"]
    xs = [r["ctx"] for r in d07]
    ax.plot(xs, [r["metrics"]["prefill_tok_per_s"] / 1e3 for r in d07], "o-",
            color=C["d07"], lw=1.8, ms=4.5, label="Mnemos-DNA(-C) prefill")
    ax.plot([r["ctx"] for r in d01], [r["metrics"]["prefill_tok_per_s"] / 1e3 for r in d01], "o-",
            color=C["d01"], lw=1.5, ms=4, alpha=0.85, label="Pure-KDA (D01) prefill")
    tr = [(r["ctx"], r["metrics"]["train_tok_per_s"] / 1e3) for r in d07
          if r["metrics"].get("train_tok_per_s")]
    ax.plot([c for c, _ in tr], [v for _, v in tr], "s--", color=C["d07"], lw=1.4, ms=4,
            alpha=0.7, label="Mnemos-DNA(-C) training step")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.set_xticklabels([kfmt(x) for x in xs])
    ax.set_ylim(0)
    ax.set_xlabel("sequence length (bp)")
    ax.set_ylabel("throughput (K tokens/s)")
    ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig_throughput.svg")


# ---------------------------------------------------------------- absolute downstream table
TAB_TMPL = r"""\begin{table}[!tbp]
\centering\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{lcccc}
\toprule
Task (full fine-tune, test accuracy) & Pure-KDA (D01) & Mnemos-DNA & Mnemos-DNA-C & random init \\
\midrule
%s \\
\bottomrule
\end{tabular}}
\caption{Absolute per-task fine-tuned accuracy on the Nucleotide Transformer subset and all eight
Genomic Benchmarks tasks under one matched protocol for every column (6 epochs / 6{,}000 steps,
4{,}000/2{,}000 train/test caps, $[\text{mean}|\text{last}|\text{8 bins}]{+}$RC head).
Mnemos-DNA(-C) cells are mean\,$\pm$\,sample std over fine-tuning seeds (per-cell $n$ in
superscript; Mnemos-DNA's seed-0 Genomic Benchmarks rows were archived by the harness's
fingerprint guard, leaving $n{=}2$ there). Pure-KDA (D01) is a single seed. The random-init
column fine-tunes the Mnemos-DNA architecture from scratch under the same head (2 seeds,
Nucleotide Transformer only). All our arms are $\sim$2.3M-parameter models pretrained for 4.5B
tokens on a single 8\,GB consumer GPU; published HyenaDNA / Nucleotide Transformer / DNABERT-2
numbers on these suites come from larger models and far larger pretraining budgets, so any
comparison to them is a comparison across compute budgets, not architectures.}
\label{tab:downstream-abs}
\end{table}
"""


def tab_downstream():
    d01 = ft_seeds("d01_new", stems=("ft_seed0",))
    a, b = ft_seeds("d06"), ft_seeds("d07")
    ri = ft_seeds("d06", stems=("ft_randinit_seed0", "ft_randinit_seed1"))
    lines = []

    def cell(per, bn, task, bold=False):
        vals = per.get((bn, task))
        if not vals:
            return "---"
        body = f"{mean(vals):.3f}"
        s = std(vals)
        if s == s:
            body += f"\\,{{\\scriptsize$\\pm${s:.3f}}}$^{{({len(vals)})}}$"
        return f"\\best{{{body}}}" if bold else body

    for bn, task, lab in NT_GB:
        if bn == "genomic_benchmarks" and not any(x.startswith("GB") for x in
                                                 [l.split(" &")[0] for l in lines]):
            lines.append("\\midrule")
        cands = {k: mean(v[(bn, task)]) for k, v in (("d01", d01), ("a", a), ("b", b))
                 if v.get((bn, task))}
        top = max(cands, key=cands.get)
        lines.append(f"{lab} & {cell(d01, bn, task, top == 'd01')} & {cell(a, bn, task, top == 'a')}"
                     f" & {cell(b, bn, task, top == 'b')} & {cell(ri, bn, task)}")
    body = " \\\\\n".join(lines).replace("\\midrule \\\\", "\\midrule")
    out = OUT.parent / "tab_downstream_abs.tex"
    out.write_text(TAB_TMPL % body)
    print("wrote", out.name)


# ---------------------------------------------------------------- fig: needle mutations
def fig_needle_mut():
    """Copy advantage vs distance at ctx 131,072 for 0-3 substitutions in the planted
    256bp segment, for Mnemos-DNA (d06) and Mnemos-DNA-C (d07). m=0 from the main suite
    (d06 step 34177, d07 step 33900); m=1/2/3 from the needle_mut sweep, rerun at the same
    checkpoints (d06 needle_mut{m}_34177.json)."""
    arms = [("d06", "Mnemos-DNA", "results/d06/benchmark_cuda_recovered_step34177.json"),
            ("d07", "Mnemos-DNA-C", "results/d07/benchmark_cuda.json")]
    shades = {0: None, 1: "#5a94dd", 2: "#9db9e0", 3: "#c8d8ef"}
    styles = {0: "-", 1: "--", 2: ":", 3: "-."}
    labels = {0: "m=0 · exact repeat",
              1: "m=1 · 1 substitution (99.6% id)",
              2: "m=2 · 2 substitutions (99.2% id)",
              3: "m=3 · 3 substitutions (98.8% id)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (arm, name, base) in zip(axes, arms):
        for m in (0, 1, 2, 3):
            if m == 0:
                rs = [r for r in rows(base, "needle")
                      if r["status"].startswith("ok") and r["ctx"] == 131072]
                col = C[arm]
            else:
                stem = f"needle_mut{m}_34177" if arm == "d06" else f"needle_mut{m}"
                rs = json.load(open(ROOT / f"results/{arm}/{stem}.json"))["rows"]
                rs = [r for r in rs if r["status"] == "ok" and r["ctx"] == 131072]
                col = shades[m]
            rs = sorted(rs, key=lambda r: r["metrics"]["distance_bp"])
            xs = [r["metrics"]["distance_bp"] for r in rs]
            ys = [r["metrics"]["copy_advantage"] for r in rs]
            lo = [r["metrics"].get("copy_advantage_lo",
                  r["metrics"]["copy_advantage"] - 1.96 * r["metrics"].get("stderr", 0)) for r in rs]
            hi = [r["metrics"].get("copy_advantage_hi",
                  r["metrics"]["copy_advantage"] + 1.96 * r["metrics"].get("stderr", 0)) for r in rs]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(xs, ys, "o", color=col, lw=1.8, ms=4.5, ls=styles[m], label=labels[m])
        ax.axhline(0, color=C["muted"], lw=0.8, ls=":")
        ax.set_xscale("log")
        ticks = [64, 256, 1024, 4096, 16384, 65536]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels(["64", "256", "1K", "4K", "16K", "64K"])
        ax.set_xlabel("needle\u2192query distance (bp), context 131,072")
        ax.set_ylim(-0.05, 1.0)
        ax.grid(axis="y")
        ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("copy advantage (\u0394 cross-entropy, nats)")
    axes[0].legend(loc="lower left", fontsize=8,
                   title="mutations in planted 256 bp segment", title_fontsize=8)
    save(fig, "fig_needle_mut.svg")


if __name__ == "__main__":
    fig_curves()
    fig_needle()
    fig_needle_mut()
    fig_scaling()
    fig_downstream()
    fig_gue()
    fig_species()
    fig_lm_ctx()
    fig_throughput()
    tab_downstream()
