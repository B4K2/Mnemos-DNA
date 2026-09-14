"""Draws paper/fig_arch.svg and paper/fig_mixer.svg (plus figures/*.pdf via headless Chrome).

fig_arch  — vertical, transformer-convention (input at the bottom): (a) the 8-block stack,
            (b) one block's pre-norm residual datapath with the four sigmoid gates as chips.
fig_mixer — (a) the two memories inside the mixer (KDA state + per-8-mer fast weights),
            (b) the Mnemos-DNA-C training-length curriculum.

Hand-laid-out schematics; no experiment data is read. Sizes are chosen so the smallest
label is >=6pt when each SVG is included at its \\includegraphics width in main.tex.
Usage: uv run python paper/make_arch_figure.py
"""
import subprocess, tempfile, pathlib

INK, MUT, RULE, BLU, ORA = "#1C1B18", "#6B6A64", "#E4E2DB", "#2a78d6", "#eb6834"
o = []
def a(s): o.append(s)

def txt(x, y, s, size=11, anchor="start", fill=INK, weight="normal", style="normal"):
    a(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
      f'font-weight="{weight}" font-style="{style}">{s}</text>')

def eqn(x, y, parts, size=10, color=INK):
    """parts: (text, kind); kind n=normal s=sub u=super i=italic-var. One <text> with tspans
    so the renderer does the advances (see git history for the drift a width estimate caused)."""
    shift, out = 0.0, []
    for t, k in parts:
        target = {"s": 0.30, "u": -0.42}.get(k, 0.0)
        attrs = ""
        if k in ("s", "u"):
            attrs += f' font-size="{size * 0.72:.1f}"'
        elif k == "i":
            attrs += ' font-style="italic" font-weight="600"'
        if abs(target - shift) > 1e-9:
            attrs += f' dy="{(target - shift) * size:.2f}"'
            shift = target
        out.append(f'<tspan{attrs}>{t}</tspan>')
    if abs(shift) > 1e-9:
        out.append(f'<tspan dy="{-shift * size:.2f}"></tspan>')
    a(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" xml:space="preserve">'
      + "".join(out) + '</text>')

def box(x, y, w, h, stroke=INK, fill="none", sw=1.1, rx=3, op=1):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
      f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{sw}"/>')

def arrow(x1, y1, x2, y2, color=INK, sw=1.1, marker="arr"):
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
      f'stroke-width="{sw}" marker-end="url(#{marker})"/>')

def grid(x, y, cols, rowsn, c=11, shade=None, fill="#ffffff"):
    a('<g stroke="#ffffff" stroke-width="0.6">')
    for j in range(rowsn):
        for i in range(cols):
            a(f'<rect x="{x+i*c}" y="{y+j*c}" width="{c}" height="{c}" '
              f'fill="{shade(i,j) if shade else fill}"/>')
    a('</g>')
    a(f'<rect x="{x}" y="{y}" width="{cols*c}" height="{rowsn*c}" fill="none" '
      f'stroke="{MUT}" stroke-width="0.7"/>')

def heat(base, vals):
    return lambda i, j: f'{base}{int(vals[j][i]*255):02x}'

SV = [[.55,.20,.10,.35,.05,.28,.12,.40],[.18,.62,.24,.08,.30,.14,.44,.10],
      [.09,.26,.50,.16,.12,.38,.20,.06],[.31,.10,.19,.58,.22,.09,.15,.33],
      [.06,.34,.13,.21,.47,.18,.10,.25],[.27,.12,.36,.10,.17,.53,.29,.11],
      [.14,.41,.08,.30,.23,.12,.49,.19],[.37,.07,.28,.13,.42,.24,.11,.56]]
WV = [[.50,.12,.30,.08,.44,.18,.26,.10],[.15,.55,.09,.33,.11,.40,.14,.28],
      [.28,.10,.48,.19,.25,.07,.36,.16],[.07,.31,.13,.52,.17,.29,.09,.38],
      [.41,.16,.22,.10,.47,.13,.31,.06],[.11,.37,.08,.27,.15,.51,.20,.34],
      [.24,.09,.39,.14,.32,.11,.45,.17],[.13,.29,.18,.42,.06,.35,.12,.49]]
COL = [.40,.15,.60,.25,.50,.10,.35,.55]
OUTV = [.50,.20,.45,.30,.55,.15,.40,.25]

DEFS = ('<defs>' + "".join(
    f'<marker id="{i}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
    f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{c}"/></marker>'
    for i, c in (("arr", INK), ("arrB", BLU), ("arrO", ORA), ("arrM", MUT))) + '</defs>')

root = pathlib.Path(__file__).resolve().parent

def render(name, W, H, aria):
    body = "\n".join(o) + "\n"
    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="{aria}" '
           f'style="font-family:\'IBM Plex Sans\',system-ui,sans-serif" fill="{INK}">\n'
           + DEFS + "\n" + body + '</svg>')
    (root / f"{name}.svg").write_text(svg)
    with tempfile.TemporaryDirectory() as td:
        html = pathlib.Path(td) / "a.html"
        html.write_text('<html><head><style>@page{size:%.3fin %.3fin;margin:0}'
                        'html,body{margin:0;padding:0}svg{width:100%%;height:100%%;display:block}'
                        '</style></head><body>%s</body></html>' % (W / 100, H / 100, svg))
        subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={root/'figures'/(name + '.pdf')}", str(html)],
                       check=True, capture_output=True)
    o.clear()
    print(f"wrote {name}.svg + figures/{name}.pdf")

# ================================================================ fig_arch (vertical)
# ---------------------------------------------------------------- a: the stack, bottom-up
txt(8, 20, "a &#183; the stack &#8212; 8 blocks, d = 128,", 12.5, "start", INK, "600")
txt(8, 36, "4 heads, no positional encoding", 12.5, "start", INK, "600")

ACX, ABW = 115, 150          # column centre / box width
def vbox(cy, h, label, hl=False, size=10):
    x = ACX - ABW / 2
    box(x, cy - h / 2, ABW, h, ORA if hl else INK, ORA if hl else "none",
        1.7 if hl else 1.1, 3, 0.12)
    txt(ACX, cy + 4, label, size, "middle", ORA if hl else INK, "600" if hl else "normal")

def vup(y1, y2, cx=ACX):     # upward arrow between elements
    arrow(cx, y1, cx, y2)

# bottom-up: y decreases upward
txt(ACX, 452, "&#8230; A C G T G G N &#8230;", 11, "middle")
txt(ACX, 466, "1 token = 1 base", 9, "middle", MUT)
vup(442, 428)
vbox(414, 26, "embed &#183; d = 128");  vup(401, 391)
vbox(378, 24, "block 0");               vup(366, 356)
vbox(344, 24, "block 1");               vup(332, 326)
txt(ACX, 318, "&#8942;", 12, "middle", MUT); txt(ACX + 88, 318, "blocks 2&#8211;3", 8.5, "start", MUT)
vup(308, 298)
vbox(283, 28, "block 4", True, 10.5)
vup(268, 258)
txt(ACX, 250, "&#8942;", 12, "middle", MUT); txt(ACX + 88, 250, "blocks 5&#8211;6", 8.5, "start", MUT)
vup(240, 230)
vbox(217, 24, "block 7");               vup(205, 195)
vbox(182, 24, "norm &#183; LM head");   vup(170, 158)
txt(ACX, 150, "p(next base)", 10, "middle", MUT)
# block-4 annotation
a(f'<line x1="{ACX+ABW/2}" y1="283" x2="{ACX+ABW/2+14}" y2="283" stroke="{ORA}" stroke-width="1.1"/>')
txt(ACX + ABW / 2 + 18, 266, "Mnemos-DNA-C only:", 8.5, "start", ORA, "600")
txt(ACX + ABW / 2 + 18, 278, "causal local attention, window 256", 8.5, "start", ORA)
txt(ACX + ABW / 2 + 18, 290, "(in Mnemos-DNA this is a KDA", 8.5, "start", MUT)
txt(ACX + ABW / 2 + 18, 301, "block like all the others)", 8.5, "start", MUT)

# ---------------------------------------------------------------- b: one block, vertical
txt(400, 20, "b &#183; one block &#8212; pre-norm residual datapath,", 12.5, "start", INK, "600")
txt(400, 36, "and the four gates that steer the mixer", 12.5, "start", INK, "600")

BCX, BBW = 460, 150
def bvbox(cy, h, label, sub=None, bold=False):
    x = BCX - BBW / 2
    box(x, cy - h / 2, BBW, h, INK, "none", 1.3 if bold else 1.1, 3)
    if sub:
        txt(BCX, cy - 1, label, 10, "middle", INK, "600")
        txt(BCX, cy + 11, sub, 8, "middle", MUT)
    else:
        txt(BCX, cy + 4, label, 9.5, "middle")

def plus(cy):
    a(f'<circle cx="{BCX}" cy="{cy}" r="10" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>')
    txt(BCX, cy + 4, "+", 12, "middle")

# bottom-up datapath
txt(BCX, 452, "x", 11, "middle", INK, "600", "italic")
a(f'<text x="{BCX+7}" y="455" font-size="8" fill="{INK}">&#8467;</text>')
arrow(BCX, 442, BCX, 432)
bvbox(419, 24, "RMSNorm")
arrow(BCX, 407, BCX, 397)
bvbox(376, 42, "sequence mixer", "KDA state + k-mer memory (Fig. 2)", True)
arrow(BCX, 355, BCX, 341)
plus(331)
arrow(BCX, 321, BCX, 311)
bvbox(298, 24, "RMSNorm")
arrow(BCX, 286, BCX, 276)
bvbox(263, 24, "SwiGLU FFN")
arrow(BCX, 251, BCX, 237)
plus(227)
arrow(BCX, 217, BCX, 207)
txt(BCX, 198, "x", 11, "middle", INK, "600", "italic")
a(f'<text x="{BCX+7}" y="201" font-size="8" fill="{INK}">&#8467;+1</text>')
# residual skips (left side)
a(f'<path d="M{BCX} 437 C {BCX-90} 430, {BCX-90} 338, {BCX-11} 331" fill="none" '
  f'stroke="{MUT}" stroke-width="1.1" marker-end="url(#arrM)"/>')
txt(BCX - 96, 388, "residual", 8.5, "middle", MUT)
a(f'<path d="M{BCX} 316 C {BCX-90} 310, {BCX-90} 233, {BCX-11} 227" fill="none" '
  f'stroke="{MUT}" stroke-width="1.1" marker-end="url(#arrM)"/>')
txt(BCX - 96, 274, "residual", 8.5, "middle", MUT)

# the four gate chips, right of the mixer, arrows into it
GX, GW = 566, 166
txt(GX, 232, "four per-token gates,", 9.5, "start", MUT)
txt(GX, 244, "each &#963;(learned proj of h", 9.5, "start", MUT)
a(f'<text x="{GX+114}" y="247" font-size="7" fill="{MUT}">t</text>')
a(f'<text x="{GX+118}" y="244" font-size="9.5" fill="{MUT}">):</text>')
def gate_chip(cy, name, sub, col):
    box(GX, cy, GW, 32, col, col, 1.2, 4, 0.10)
    txt(GX + GW / 2, cy + 13, name, 9.5, "middle", col, "600")
    txt(GX + GW / 2, cy + 26, sub, 8, "middle", MUT)
    a(f'<path d="M{GX} {cy+16} C {GX-24} {cy+16}, {BCX+BBW/2+20} 376, {BCX+BBW/2+2} 376" '
      f'fill="none" stroke="{col}" stroke-width="0.9" opacity="0.55" '
      f'marker-end="url(#arr{"B" if col==BLU else "O"})"/>')
gate_chip(256, "&#945; &#183; KDA forget",         "diag decay of the state S",       BLU)
gate_chip(296, "&#946; &#183; KDA write strength", "scales the delta-rule write",     BLU)
gate_chip(336, "g_w &#183; memory write",          "&#963;(0) = 0.50 at init",        ORA)
gate_chip(376, "g_r &#183; memory read",           "&#963;(&#8722;1) = 0.27 at init", ORA)
txt(GX, 428, "all gate parameters kept in fp32", 8, "start", MUT)
txt(GX, 439, "(scalar gates freeze under bf16 rounding)", 8, "start", MUT)

render("fig_arch", 740, 480,
       "Mnemos-DNA architecture, drawn vertically with the input at the bottom: the eight-block "
       "stack with block 4 replaced by local attention in Mnemos-DNA-C, and one block's pre-norm "
       "residual datapath with its four sigmoid gates (KDA forget and write strength; k-mer "
       "memory write and read gates).")

# ================================================================ fig_mixer
# ---------------------------------------------------------------- a: inside the mixer
txt(8, 18, "a &#183; inside the mixer: two memories, read in parallel from the same q, k, v",
    12.5, "start", INK, "600")
box(8, 26, 724, 414, RULE, "none", 1.0, 6)

grid(22, 168, 1, 8, 10)
txt(27, 162, "h", 11, "middle"); txt(33, 165, "t", 8, "middle")
txt(27, 262, "d = 128", 9, "middle", MUT)
arrow(36, 208, 50, 208, MUT)
box(50, 182, 86, 56, MUT)
txt(93, 200, "short causal", 10, "middle")
txt(93, 213, "conv, k = 4", 10, "middle")
txt(93, 229, "&#8467;2-norm q, k", 9, "middle", MUT)
arrow(138, 208, 150, 208, MUT)
for i, lab in enumerate("qkv"):
    grid(154 + i * 18, 168, 1, 8, 10)
    txt(159 + i * 18, 262, lab, 10, "middle")
txt(177, 276, "per head, 32-dim", 9, "middle", MUT)
a(f'<path d="M196 182 C 216 182, 214 108, 232 102" fill="none" stroke="{BLU}" '
  f'stroke-width="1.3" marker-end="url(#arrB)"/>')
a(f'<path d="M196 234 C 216 234, 214 310, 232 316" fill="none" stroke="{ORA}" '
  f'stroke-width="1.3" marker-end="url(#arrO)"/>')

# lane 1: KDA recurrent state
txt(238, 48, "1 &#183; KDA recurrent state &#8212; one matrix for the whole sequence",
    11, "start", BLU, "600")
grid(238, 58, 8, 8, 11, heat(BLU, SV))
txt(282, 162, "S", 11, "middle", BLU, "600", "italic"); txt(289, 165, "t", 8, "middle", BLU)
txt(282, 175, "32 &#215; 32 per head", 9, "middle", MUT)
txt(334, 110, "&#183;", 14, "middle", MUT)
grid(344, 58, 1, 8, 11, heat(BLU, [[v] for v in COL]))
txt(349, 162, "q", 11, "middle", BLU, "600", "italic"); txt(356, 165, "t", 8, "middle", BLU)
txt(372, 108, "=", 12, "middle", MUT)
grid(384, 58, 1, 8, 11, heat(BLU, [[v] for v in OUTV]))
txt(389, 175, "read-out", 9, "middle", BLU)
txt(414, 68, "update, every token:", 9.5, "start", MUT)
eqn(414, 86, [("S", "i"), ("t", "s"), (" = diag(a", "n"), ("t", "s"), (") S", "n"),
              ("t&#8722;1", "s"), (" + &#946;", "n"), ("t", "s"), (" ( v", "n"), ("t", "s"),
              (" &#8722; S", "n"), ("t&#8722;1", "s"), (" k", "n"), ("t", "s"),
              (" ) k", "n"), ("&#8868;", "u")], 10, BLU)
txt(414, 104, "channel-wise forget gate a &#8712; (0,1);", 9.5, "start", MUT)
txt(414, 117, "the delta rule writes v while erasing what S predicts.", 9.5, "start", MUT)
txt(414, 135, "O(1) memory &#8212; but one shared matrix, so every new", 9.5, "start", INK)
txt(414, 148, "8-mer competes for the same fixed capacity.", 9.5, "start", INK)

# lane 2: per-k-mer fast weights
txt(238, 294, "2 &#183; per-8-mer fast weights &#8212; one matrix per address",
    11, "start", ORA, "600")
txt(238, 312, "address = the last 8 bases", 9.5, "start", MUT)
for i, ch in enumerate("ACGTGGCA"):
    box(238 + i * 15, 318, 14, 17, ORA, ORA, 1.0, 2, 0.10)
    txt(245 + i * 15, 330, ch, 9.5, "middle", ORA)
arrow(360, 328, 372, 328, ORA)
txt(376, 325, "a = 4&#8312; = 65,536 slots (hashed, not learned)", 9.5, "start")
for j in range(6):
    hl = j == 2
    box(238, 346 + j * 13, 80, 12, ORA if hl else RULE,
        ORA if hl else "#ffffff", 1.4 if hl else 0.9, 2, 0.16 if hl else 1)
txt(278, 381, "slot a", 8.5, "middle", ORA)
txt(238, 438, "65,536 slots", 9, "start", MUT)
arrow(322, 378, 336, 378, ORA)
grid(342, 334, 8, 8, 11, heat(ORA, WV))
txt(386, 438, "W", 10, "middle", ORA, "600", "italic"); txt(394, 441, "a", 8, "middle", ORA)
txt(403, 438, "&#183; 16 &#215; 16, private to this 8-mer", 9, "start", MUT)
txt(444, 352, "write (gate g", 9.5, "start", MUT)
a(f'<text x="500" y="355" font-size="7" fill="{MUT}">w</text>')
a(f'<text x="507" y="352" font-size="9.5" fill="{MUT}">, &#963;(0) = 0.50 at init):</text>')
eqn(444, 370, [("W", "i"), ("a", "s"), (" &#8592; W", "n"), ("a", "s"), (" + g", "n"),
               ("w", "s"), (" ( v &#8722; W", "n"), ("a", "s"), (" k ) k", "n"),
               ("&#8868;", "u")], 10, ORA)
txt(444, 390, "read (gate g", 9.5, "start", MUT)
a(f'<text x="496" y="393" font-size="7" fill="{MUT}">r</text>')
a(f'<text x="500" y="390" font-size="9.5" fill="{MUT}">, &#963;(&#8722;1) = 0.27 at init):</text>')
eqn(444, 408, [("g", "i"), ("r", "s"), (" &#183; W", "n"), ("a", "s"), (" q", "n")], 10, ORA)

# left-bottom note
txt(22, 306, "The two memories differ in kind,", 9.5, "start", INK)
txt(22, 319, "not in size:", 9.5, "start", INK)
txt(22, 337, "S &#8212; one matrix, shared by every", 9.5, "start", MUT)
txt(22, 350, "position in the sequence.", 9.5, "start", MUT)
txt(22, 368, "W &#8212; 65,536 matrices, each one", 9.5, "start", MUT)
txt(22, 381, "owned by a single exact 8-mer.", 9.5, "start", MUT)
txt(22, 404, "+~3% parameters; the block is", 9.5, "start", INK)
txt(22, 417, "still O(N) in sequence length.", 9.5, "start", INK)

# merge
a(f'<path d="M392 188 C 420 212, 470 226, 546 228" fill="none" stroke="{BLU}" '
  f'stroke-width="1.3" marker-end="url(#arrB)"/>')
a(f'<path d="M552 406 L688 406 L688 270 L570 270 L570 241" fill="none" stroke="{ORA}" '
  f'stroke-width="1.3" marker-end="url(#arrO)"/>')
a(f'<circle cx="560" cy="228" r="11" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>')
txt(560, 233, "+", 13, "middle")
arrow(572, 228, 584, 228)
box(584, 216, 56, 24); txt(612, 232, "o_proj", 10, "middle")
arrow(640, 228, 652, 228)
txt(656, 232, "out", 10, "start", MUT)

# ---------------------------------------------------------------- b: curriculum
txt(8, 470, "b &#183; Mnemos-DNA-C training-length curriculum (Mnemos-DNA trains at a fixed 131K throughout)",
    12.5, "start", ORA, "600")
bx, by, bw = 150, 496, 560
a(f'<path d="M{bx} {by+108} L {bx+56} {by+4}" stroke="{ORA}" stroke-width="1.6" fill="none"/>')
a(f'<line x1="{bx+56}" y1="{by}" x2="{bx+56}" y2="{by+108}" stroke="{MUT}" '
  f'stroke-width="0.9" stroke-dasharray="3 3"/>')
txt(bx, by - 6, "1K &#8594; 131K warmup over the first 10% of steps", 9.5, "start", ORA)
yy = by + 6
for lab, op in (("131K &#183; 50% of steps", 0.30), ("16K &#183; 25%", 0.20), ("1K &#183; 25%", 0.10)):
    a(f'<rect x="{bx+60}" y="{yy}" width="{bw-60}" height="28" fill="{ORA}" '
      f'fill-opacity="{op}" stroke="{ORA}" stroke-width="0.8"/>')
    txt(bx + 60 + (bw - 60) / 2, yy + 18, lab, 10, "middle")
    yy += 31
a(f'<line x1="{bx}" y1="{by+110}" x2="{bx+bw}" y2="{by+110}" stroke="{INK}" '
  f'stroke-width="1.1" marker-end="url(#arr)"/>')
txt(bx + bw, by + 126, "training steps", 9.5, "end", MUT)
txt(bx, by + 146, "then a fixed per-step mix &#8212; tokens/step stays 131,072, so the token "
    "budget and LR schedule match Mnemos-DNA exactly", 9.5, "start", MUT)

render("fig_mixer", 740, 656,
       "Inside the Mnemos-DNA mixer: the shared 32x32 KDA recurrent state and the 65,536 "
       "per-8-mer 16x16 fast-weight matrices, with the gated write and read equations; and the "
       "Mnemos-DNA-C training-length curriculum.")
