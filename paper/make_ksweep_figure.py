"""k sweep figure for main_v2.tex: copy advantage vs distance for k = 4, 6, 7, 8.

Reuses the palette/rcParams of make_figures.py. Writes figures/fig_ksweep.{svg,pdf}.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from make_figures import C, ROOT, save

# sequential ramp on the Mnemos-DNA hue: larger address space = darker
ARMS = [
    (4, "results/d06_k4/review_needle.json", "#a9c8ee"),
    (6, "results/d06_k6/review_needle.json", "#6fa3e3"),
    (7, "results/d06_k7/review_needle.json", "#3f86da"),
    (8, "results/d06/review_needle.json", "#1b4f8f"),
]
TICKS = [64, 256, 1024, 4096, 16384, 65536]


def load(path):
    rs = [r for r in json.load(open(ROOT / path))["rows"]
          if r["benchmark"] == "needle" and r["status"].startswith("ok")]
    rs.sort(key=lambda r: r["metrics"]["distance_bp"])
    m = [r["metrics"] for r in rs]
    return ([x["distance_bp"] for x in m], [x["copy_advantage"] for x in m],
            [x["copy_advantage_lo"] for x in m], [x["copy_advantage_hi"] for x in m])


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    xs, ys, _, _ = load("results/d01_new/review_needle.json")
    a.plot(xs, ys, "o-", color=C["d01"], lw=1.4, ms=3.5, label="Pure-KDA (D01)")
    for k, path, col in ARMS:
        xs, ys, lo, hi = load(path)
        lab = f"k={k} ({4**k:,} slots)"
        a.plot(xs, ys, "o-", color=col, lw=1.8, ms=4, label=lab)
        a.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
        a.axvline(4**k, color=col, lw=0.8, ls=":") if 4**k >= 64 else None
        b.plot([x / 4**k for x in xs], [y / ys[0] for y in ys], "o-", color=col,
               lw=1.8, ms=4, label=lab)
    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator(TICKS))
    a.set_xticklabels(["64", "256", "1K", "4K", "16K", "64K"])
    a.set_xlabel("needle→query distance d (bp), context 131,072")
    a.set_ylabel("copy advantage (nats)")
    a.set_ylim(-0.05, 1.0)
    a.grid(axis="y")
    a.set_title("(a) recall vs distance; dotted line at d = 4^k")

    b.set_xscale("log")
    b.axvline(1, color=C["muted"], lw=0.8, ls=":")
    b.set_xlabel("d / 4^k  (expected uniform writes to one slot in between)")
    b.set_ylabel("copy advantage / value at d=64")
    b.set_ylim(-0.05, 1.1)
    b.grid(axis="y")
    b.legend(loc="lower left", fontsize=7.5)
    # D01 is only in (a): label it in place rather than in (b)'s legend
    a.text(1100, 0.035, "Pure-KDA (D01)", color=C["d01"], fontsize=8)
    b.set_title("(b) same curves, distance in units of 4^k")
    save(fig, "fig_ksweep.svg")


if __name__ == "__main__":
    main()
