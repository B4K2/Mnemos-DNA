/**
 * dna_model_comparison.jsx
 * ---------------------------------------------------------------------------
 * One page, all arms. Every trained arm in results/ that has a benchmark sweep
 * (results/<arm>/benchmark_cuda*.{json,csv}, produced by scripts/benchmark.py)
 * appears on every chart at once — there is no per-model view to switch between.
 *
 * Data flow, fully automatic:
 *   scripts/benchmark.py  ->  results/<arm>/benchmark_cuda*.json
 *   scripts/gen_dashboard_data.py  ->  template/src/results_data.json
 *   this file imports that JSON and derives the arm list from it.
 * `npm run dev` / `npm run build` re-run the generator first (package.json
 * predev/prebuild), so a newly benchmarked arm shows up with no code edits.
 *
 * Published baselines live in src/published_baselines.js, transcribed from the
 * HyenaDNA (arXiv:2306.15794) and Caduceus (arXiv:2403.03234) paper tables.
 *
 * Separately from those transcribed numbers, the released HyenaDNA and Caduceus
 * checkpoints are also run through our own harness by scripts/baselines_panel.sh
 * (results/<arm>/panel_cuda.json, species + GLRB, frozen probe). Those arrive as
 * their own arms (hyenadna_1m, caduceus_ph, caduceus_ps) alongside d02_panel /
 * d06_panel / d07_panel — ours re-measured in the same pass. Panel numbers are a
 * separate measurement from the main suite's and compare only with each other.
 *
 * Run:  npm install && npm run dev     (build: npm run build -> dist/)
 * PDF:  the "Download PDF" button prints the whole page via the browser's
 *       print-to-PDF, with a print stylesheet that expands every section.
 */
import React from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer, LabelList, Cell,
} from "recharts";
import RESULTS from "./src/results_data.json";
import { GB_BASELINES, NT_BASELINES, BASELINE_PARAMS, PAPER_SOURCES } from "./src/published_baselines.js";

// ===========================================================================
// Arm discovery — anything with real benchmark rows is shown.
// ===========================================================================
// Presentation-only overrides. An arm missing from here still renders, using
// its run name from the benchmark JSON; only the prose description is lost.
const ARM_INFO = {
  d01:      { label: "d01",      what: "Pure KDA, no positional encoding. The first arm — 120M params, before the matrix was re-scaled to ~2.25M." },
  d01_new:  { label: "d01_new",  what: "Pure KDA, NoPE, 2.25M params, 262K trained context. The NoPE reference point." },
  d01_rope: { label: "d01_rope", what: "KDA + NTK-scaled RoPE, 262K trained context. Isolates the positional-encoding axis against d01_new." },
  d02:      { label: "d02",      what: "KDA + windowed local attention, NTK-RoPE. Adds a short exact-attention window, still O(N)." },
  d04:      { label: "d04_hola", what: "KDA + a bounded exact key-value cache evicted by error magnitude (HOLA, arXiv:2607.02303), NoPE, 65K trained context." },
  d05_rope: { label: "d05_rope", what: "KDA + set-associative k-mer memory, NTK-RoPE." },
  d06:      { label: "d06",      what: "KDA + per-slot fast-weight k-mer memory (16x16 delta-rule matrices), NoPE, 131K trained context." },
  d07_ckpt24490: { label: "d07@24490", what: "The previous d07 checkpoint (step 24,490, 3.21B tokens), kept beside the current step-33,900 one because it still scores higher on a few downstream tasks. Same run, same protocol — an earlier point on the same training curve, not a separate arm." },
  d07:      { label: "d07",      what: "d06 + windowed local attention (w=256) + a 1K->131K sequence-length curriculum, NoPE, 131K trained context. Downstream rows use the post-2026-08-30 fine-tune head (positional bins + reverse-complement), which earlier arms do not — see the head note on the fine-tune tables." },
  // --- The baselines panel (scripts/baselines_panel.sh): released HyenaDNA and
  // Caduceus checkpoints run through OUR harness — same hg38 windows, same seeds,
  // same frozen probe — plus our own arms re-measured in the same pass. It covers
  // species and GLRB only, and its numbers are a separate measurement from the main
  // suite's (different ctx points and resampling), so panel rows compare with panel
  // rows: read hyenadna_1m / caduceus_* against d02_panel / d06_panel / d07_panel,
  // not against the plain d02 / d06 / d07 rows.
  hyenadna_1m: { label: "HyenaDNA-1M", what: "Released LongSafari/hyenadna-large-1m-seqlen checkpoint (6.55M params, 1M trained context), run through our harness. Pretraining budget is not matched to ours — every comparison with it is across different compute." },
  caduceus_ph: { label: "Caduceus-Ph", what: "Released kuleshov-group/caduceus-ph 131K checkpoint (7.73M params), run through our harness. A bidirectional masked LM: lm and needle do not apply to it, so it appears on the probe benchmarks only. Pretraining budget not matched to ours." },
  caduceus_ps: { label: "Caduceus-PS", what: "Released kuleshov-group/caduceus-ps 131K checkpoint (7.73M params; RCPS doubles the hidden width). Bidirectional MLM, probe benchmarks only. Pretraining budget not matched to ours." },
  d02_panel:   { label: "d02 (panel)", what: "d02 re-measured in the baselines panel — the row to compare with HyenaDNA/Caduceus, since it was measured in the same pass at the same context points." },
  d06_panel:   { label: "d06 (panel)", what: "d06 re-measured in the baselines panel — the row to compare with HyenaDNA/Caduceus." },
  d07_panel:   { label: "d07 (panel)", what: "d07 re-measured in the baselines panel — the row to compare with HyenaDNA/Caduceus." },
};

// The panel arms (and the bidirectional Caduceus checkpoints) have neither an lm
// sweep nor Genomic Benchmarks — species/GLRB rows alone are enough to render.
function hasData(v) {
  return (
    (v?.lm?.length ?? 0) > 0 ||
    Object.keys(v?.genomic_benchmarks ?? {}).length > 0 ||
    (v?.species?.length ?? 0) > 0 ||
    Object.keys(v?.lrb ?? {}).length > 0
  );
}

const ALL_ARMS = Object.entries(RESULTS)
  .filter(([, v]) => hasData(v))
  .map(([key, v]) => ({
    key,
    label: ARM_INFO[key]?.label ?? key,
    what: ARM_INFO[key]?.what ?? v.meta.runName,
    ...v.meta,
    training: v.meta.training ?? null,
    source: v._source,
  }))
  .sort((a, b) => a.key.localeCompare(b.key));

// Two disjoint populations, and they must never share a chart:
//   suite  — our arms as measured by scripts/benchmark.py (everything on this page
//            except section 09).
//   panel  — scripts/baselines_panel.sh: the released HyenaDNA/Caduceus checkpoints
//            run through our harness, plus our own arms re-measured in that same
//            pass. Same protocol as each other, a *different* measurement from the
//            suite (different context points and resampling), so section 09 owns
//            them exclusively and the rest of the page ignores them.
const ARMS = ALL_ARMS.filter((a) => a.group !== "panel");
const PANEL_ARMS = ALL_ARMS.filter((a) => a.group === "panel");

const ARM_KEYS = ARMS.map((a) => a.key);
const PANEL_KEYS = PANEL_ARMS.map((a) => a.key);
const armLabel = (k) => ALL_ARMS.find((a) => a.key === k)?.label ?? k;

// ===========================================================================
// Styling
// ===========================================================================
const INK = "#1a1d29";
const PAPER = "#f7f5f0";
const GRID = "#e2ddd0";
const MONO = "'IBM Plex Mono', ui-monospace, monospace";
// One stable colour per arm, assigned by position in the (sorted) arm list.
const ARM_COLORS = ["#2f6f5e", "#5c7a99", "#a3673f", "#6b5b95", "#3f7d8c", "#8a7a2f", "#96455f"];
const BASELINE_GREY = ["#c9c2b2", "#b3a893", "#a89b82", "#9c8f78", "#8f8268", "#83755c"];

const HIGHLIGHT_KEY = "d07";
const HIGHLIGHT_COLOR = "#c23b22";

// Externally pretrained checkpoints get their own (warmer, dashed-in-spirit) palette so
// they never read as one of our arms on the section-09 charts.
const EXTERNAL_COLORS = { hyenadna_1m: "#b8863b", hyenadna_450k: "#8a6a2f", caduceus_ph: "#7a4b7d", caduceus_ps: "#4a6b8a" };

// An arm's colour is stable across every chart. A panel re-measurement of one of our
// arms (d06_panel) deliberately keeps its parent's colour — same model, second pass.
const armColor = (k) => {
  if (k === HIGHLIGHT_KEY) return HIGHLIGHT_COLOR;
  if (EXTERNAL_COLORS[k]) return EXTERNAL_COLORS[k];
  const base = k.endsWith("_panel") ? k.slice(0, -"_panel".length) : k;
  if (base === HIGHLIGHT_KEY) return HIGHLIGHT_COLOR;
  const i = ARM_KEYS.indexOf(base);
  return ARM_COLORS[(i < 0 ? ALL_ARMS.findIndex((a) => a.key === k) : i) % ARM_COLORS.length];
};

const tickStyle = (size = 11, fill = INK) => ({ fontFamily: MONO, fontSize: size, fill });
const tooltipStyle = { fontFamily: MONO, fontSize: 12, border: `1px solid ${INK}`, borderRadius: 2 };
const CTX_LADDER = ["8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"];

// ===========================================================================
// Data shaping — one helper per benchmark, all keyed by arm.
// ===========================================================================
const get = (armKey, bench) => RESULTS[armKey]?.[bench];

// Length-swept benchmarks -> [{ ctx, <arm>: value, ... }] on the shared ladder.
// `keys` defaults to our suite arms; section 09 passes the panel arms instead.
function byCtx(bench, field, scale = 1, keys = ARM_KEYS) {
  return CTX_LADDER.map((ctx) => {
    const row = { ctx };
    for (const k of keys) {
      const hit = (get(k, bench) ?? []).find((r) => r.ctx === ctx);
      row[k] = hit && hit[field] != null ? hit[field] * scale : null;
    }
    return row;
  }).filter((row) => keys.some((k) => row[k] != null));
}

// Fixed-length suites -> per-task score for each arm.
const suiteScore = (armKey, bench, task, field) => get(armKey, bench)?.[task]?.[field] ?? null;
// "probe" (frozen-backbone linear probe) or "finetune" (whole-model fine-tune, eval_all.sh's
// second pass) — set per row by benchmarks/{probe,finetune}.py. Labels must reflect this per
// arm/task since some arms (e.g. d06) have been fine-tuned while others are still probe-only.
const suiteMode = (armKey, bench, task) => get(armKey, bench)?.[task]?.mode ?? "probe";

// Pick the metric to plot for one arm on one task. `ourKey` is the metric that matches
// what the paper reports; `fallbackKey` covers arms benchmarked before that metric was
// recorded (F1 landed in benchmarks/{probe,finetune}.py after d01-d06 were run). Returns
// the key actually used so the bar can be labelled with it and flagged when it fell back.
function resolveMetric(armKey, bench, task, spec) {
  const primary = suiteScore(armKey, bench, task, spec.ourKey);
  if (primary != null) return { key: spec.ourKey, value: primary, fellBack: false };
  if (!spec.fallbackKey) return { key: spec.ourKey, value: null, fellBack: false };
  const alt = suiteScore(armKey, bench, task, spec.fallbackKey);
  return { key: spec.fallbackKey, value: alt, fellBack: alt != null };
}

// Needle protocol differs by arm: the older arms measured key->value recall
// accuracy (chance 25%), d04 measures zero-shot repeat-copy advantage in bits.
// Detected from the metric keys present rather than hard-coded per arm.
function needleModeOf(armKey) {
  const first = Object.values(get(armKey, "needle")?.byCtx ?? {})[0]?.[0];
  if (!first) return null;
  return "copy_advantage" in first ? "copy" : "recall";
}
const NEEDLE_MODE = Object.fromEntries(ARM_KEYS.map((k) => [k, needleModeOf(k)]));
const RECALL_ARMS = ARM_KEYS.filter((k) => NEEDLE_MODE[k] === "recall");
const COPY_ARMS = ARM_KEYS.filter((k) => NEEDLE_MODE[k] === "copy");

// Needle at each arm's own trained context (the length it was actually trained for).
// ctx labels on the ladder, in tokens ("64K" -> 65536).
const CTX_TOKENS = { "8K": 8192, "16K": 16384, "32K": 32768, "64K": 65536, "128K": 131072, "256K": 262144, "512K": 524288, "1M": 1048576 };

// Each arm's needle curve at the longest swept context that does not exceed the
// length it was trained for — comparing arms at their own operating point.
function needleAtTrained(arms, field, scale = 1) {
  const perArm = {};
  const dists = new Set();
  for (const k of arms) {
    const byCtxMap = get(k, "needle")?.byCtx ?? {};
    const trained = ARMS.find((a) => a.key === k)?.trainedContext ?? Infinity;
    const ctx = Object.keys(byCtxMap)
      .filter((c) => (CTX_TOKENS[c] ?? Infinity) <= trained)
      .sort((a, b) => CTX_TOKENS[a] - CTX_TOKENS[b])
      .pop();
    if (!ctx) continue;
    perArm[k] = { ctx, rows: byCtxMap[ctx] };
    byCtxMap[ctx].forEach((r) => dists.add(r.distance_bp));
  }
  return {
    ctxUsed: Object.fromEntries(Object.entries(perArm).map(([k, v]) => [k, v.ctx])),
    data: [...dists].sort((a, b) => a - b).map((d) => {
      const row = { dist: d < 1000 ? `${d}bp` : `${Math.round(d / 1024)}Kbp` };
      for (const [k, v] of Object.entries(perArm)) {
        const hit = v.rows.find((r) => r.distance_bp === d);
        row[k] = hit && hit[field] != null ? hit[field] * scale : null;
      }
      return row;
    }),
  };
}

const LM_BPN = byCtx("lm", "bpn");
const LM_ACC = byCtx("lm", "acc", 100);
const SPECIES_ACC = byCtx("species", "acc", 100);
const SCALING_INFER = byCtx("scaling", "infer_peak_gb");
const SCALING_TRAIN = byCtx("scaling", "train_peak_gb");
const SCALING_TPS = byCtx("scaling", "prefill_tok_per_s");
const NEEDLE_RECALL = needleAtTrained(RECALL_ARMS, "recall_acc", 100);
const NEEDLE_COPY = needleAtTrained(COPY_ARMS, "copy_advantage");

// LRB: one series per arm per task, over context.
const LRB_TASKS = [...new Set(ARM_KEYS.flatMap((k) => Object.keys(get(k, "lrb") ?? {})))];
function lrbByCtx(task, field, keys = ARM_KEYS) {
  return CTX_LADDER.map((ctx) => {
    const row = { ctx };
    for (const k of keys) {
      const hit = (get(k, "lrb")?.[task] ?? []).find((r) => r.ctx === ctx);
      row[k] = hit?.[field] ?? null;
    }
    return row;
  }).filter((row) => keys.some((k) => row[k] != null));
}

// --- Section 09, the baselines panel. Same shapes, panel arms only. ---------
const PANEL_SPECIES = byCtx("species", "acc", 100, PANEL_KEYS);
const PANEL_LRB_TASKS = [...new Set(PANEL_KEYS.flatMap((k) => Object.keys(get(k, "lrb") ?? {})))];
const lrbField = (task) => (task === "chromatin" ? "mean_auroc" : "auroc");

// GUE (DNABERT-2 suite, community mirror) — arms that ran it, mean MCC over tasks.
const GUE_TASKS = [...new Set(ARM_KEYS.flatMap((k) => Object.keys(get(k, "gue") ?? {})))].sort();
const GUE_ARMS = ARM_KEYS.filter((k) => Object.keys(get(k, "gue") ?? {}).length > 0);
// Arms whose downstream rows come from whole-model fine-tuning rather than a frozen probe.
// Detected from the rows themselves so a newly fine-tuned arm joins without a code edit.
// GUE rows can be probe or fine-tune per arm; the table footnote states which.
const gueMode = (k) => {
  const modes = new Set(Object.values(get(k, "gue") ?? {}).map((t) => t.mode ?? "probe"));
  return modes.size === 1 ? [...modes][0] : "mixed";
};

const FINETUNED_ARMS = ARM_KEYS.filter((k) =>
  Object.values(get(k, "genomic_benchmarks") ?? {}).some((t) => t.mode === "finetune"));

// Model-size chart: our arms (measured) + published baselines.
const PARAM_ROWS = [
  ...ARMS.map((a) => ({ model: a.label, params: a.params / 1e6, mine: true })),
  ...Object.entries(BASELINE_PARAMS).map(([model, params]) => ({ model, params, mine: false })),
].sort((a, b) => a.params - b.params);

// ===========================================================================
// Presentational helpers
// ===========================================================================
// Section index for the table of contents. Kept beside the Section component so a
// renumbering has one place to go wrong instead of two; each entry's `num` must match
// the `num` prop of the corresponding <Section>.
const CONTENTS = [
  ["00", "What this page is", "the arms, and which comparisons are legitimate"],
  ["01", "Model size", "weight class, ours vs. the literature"],
  ["02", "Genomic Benchmarks", "8 short-sequence classification tasks"],
  ["03", "Nucleotide Transformer tasks", "promoters, enhancers, splice sites"],
  ["04", "Held-out language modelling", "bits per nucleotide, 8K → 1M"],
  ["05", "Species classification", "does more context help on real DNA?"],
  ["06", "Long-range exact recall", "the recall bound, measured"],
  ["07", "Memory and throughput", "the O(N) claim on an 8GB card"],
  ["08", "Genomics Long-Range Benchmark", "variant effect, chromatin, regulatory"],
  ["09", "Head-to-head with released checkpoints", "HyenaDNA / Caduceus in our harness"],
  ["10", "GUE (DNABERT-2 suite)", "37 tasks, MCC"],
];

function Contents() {
  return (
    <nav className="no-print" style={{ margin: "16px 0 0", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "2px 20px" }}>
      {CONTENTS.map(([num, title, blurb]) => (
        <a key={num} href={`#s${num}`} style={{ fontFamily: MONO, fontSize: 10.5, color: "#6b6555", textDecoration: "none", padding: "3px 0", borderBottom: `1px solid ${GRID}` }}>
          <span style={{ color: INK, fontWeight: 700 }}>{num}</span>&nbsp;&nbsp;{title}
          <div style={{ color: "#8f8268", fontSize: 10, paddingLeft: 22 }}>{blurb}</div>
        </a>
      ))}
    </nav>
  );
}

function Section({ num, title, right, children }) {
  return (
    <section id={`s${num}`} className="pg" style={{ marginTop: 38, scrollMarginTop: 56 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <h2 className="serif" style={{ fontSize: 15, fontWeight: 700, margin: "0 0 4px" }}>{num}&nbsp;&nbsp;{title}</h2>
        {right ? <span style={{ fontSize: 10.5, color: "#8f8268" }}>{right}</span> : null}
      </div>
      {children}
    </section>
  );
}

// The one-paragraph "what is this benchmark" note that heads every section.
function What({ children, tone = "plain" }) {
  const border = tone === "warn" ? "#a3673f" : tone === "key" ? "#2f6f5e" : GRID;
  return (
    <div style={{ fontSize: 11.5, lineHeight: 1.65, color: "#5c5748", borderLeft: `3px solid ${border}`, padding: "6px 0 6px 12px", margin: "10px 0 16px" }}>
      {children}
    </div>
  );
}

// Marks the winning cell in a per-task table row.
function Best({ on, children }) {
  if (!on) return <>{children}</>;
  return (
    <span style={{ fontWeight: 700, color: INK, background: "rgba(194, 59, 34, 0.14)", padding: "1px 5px", borderRadius: 2 }}>
      {children}
    </span>
  );
}

function Table({ columns, rows, highlightRow, dimRow }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontFamily: MONO, fontSize: 11.5 }}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} style={{ textAlign: "left", padding: "6px 10px 6px 0", borderBottom: `1px solid ${INK}`, color: "#5c5748", fontWeight: 500, whiteSpace: "nowrap" }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const hi = highlightRow?.(i);
            const dim = dimRow?.(i);
            return (
              <tr key={i} style={{
                ...(hi ? { background: "rgba(194, 59, 34, 0.07)" } : null),
                ...(dim ? { opacity: 0.4 } : null),
              }}>
                {r.map((cell, j) => (
                  <td
                    key={j}
                    style={{
                      padding: "5px 10px 5px 0",
                      borderBottom: `1px solid ${GRID}`,
                      whiteSpace: "nowrap",
                      color: j === 0 ? INK : "#5c5748",
                      fontWeight: hi ? 700 : undefined,
                    }}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Horizontal bars: our arms coloured, published baselines grey.
function ScoreBars({ rows, height, fmt = (v) => v.toFixed(3), domain = [0, 1] }) {
  const sorted = [...rows].sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(200, sorted.length * 30)}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 52, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="0" horizontal={false} stroke={GRID} />
        <XAxis type="number" domain={domain} tick={tickStyle(10, "#6b6555")} axisLine={{ stroke: GRID }} tickLine={false} />
        <YAxis type="category" dataKey="model" width={168} tick={tickStyle(10.5)} axisLine={{ stroke: GRID }} tickLine={false} />
        <Tooltip formatter={(v) => [fmt(v), "score"]} contentStyle={tooltipStyle} />
        <Bar isAnimationActive={false} dataKey="value" radius={[0, 2, 2, 0]} barSize={14}>
          <LabelList
            dataKey="value" position="right" formatter={fmt} offset={8}
            content={({ x, y, width, height, value }) => {
              const label = fmt(value);
              const tx = x + width + 8;
              const ty = y + height / 2;
              // Paper-colour chip behind the digits so a bar's stroke/fill never
              // bleeds into the text and thickens it into illegibility.
              return (
                <g>
                  <rect x={tx - 2} y={ty - 7} width={label.length * 6 + 4} height={14} fill={PAPER} />
                  <text x={tx} y={ty} dy={4} fontFamily={MONO} fontSize={10} fontWeight={400} fill={INK}>
                    {label}
                  </text>
                </g>
              );
            }}
          />
          {sorted.map((r, i) => (
            <Cell
              key={i}
              fill={r.armKey ? armColor(r.armKey) : BASELINE_GREY[i % BASELINE_GREY.length]}
              stroke={r.armKey === HIGHLIGHT_KEY ? INK : "none"}
              strokeWidth={r.armKey === HIGHLIGHT_KEY ? 1.5 : 0}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// One line per arm over the shared context ladder.
function ArmLines({ data, height = 260, yLabel, domain, arms = ARM_KEYS, xKey = "ctx", logY = false, fmt = (v) => v }) {
  const ordered = [...arms].sort((a, b) => (a === HIGHLIGHT_KEY ? 1 : b === HIGHLIGHT_KEY ? -1 : 0));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 18, left: 4, bottom: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis dataKey={xKey} tick={tickStyle(11)} axisLine={{ stroke: GRID }} tickLine={false} />
        <YAxis
          scale={logY ? "log" : "auto"} domain={domain ?? ["auto", "auto"]} allowDataOverflow={!!domain}
          tick={tickStyle(10.5, "#6b6555")} axisLine={{ stroke: GRID }} tickLine={false}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", style: { fontFamily: MONO, fontSize: 10.5, fill: "#6b6555" } } : undefined}
        />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
        <Legend wrapperStyle={{ fontFamily: MONO, fontSize: 10.5 }} />
        {arms.map((k) => (
          <Line isAnimationActive={false} key={k} type="monotone" dataKey={k} name={armLabel(k)} stroke={armColor(k)} strokeWidth={2} dot={{ r: 2.5 }} connectNulls={false} />
        ))}
        {ordered.map((k) => (
          <Line
            isAnimationActive={false}
            key={k}
            type="monotone"
            dataKey={k}
            name={armLabel(k)}
            stroke={armColor(k)}
            strokeWidth={k === HIGHLIGHT_KEY ? 3.5 : 1.5}
            dot={{ r: k === HIGHLIGHT_KEY ? 3.5 : 2 }}
            connectNulls={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ===========================================================================
// Page
// ===========================================================================
export default function DNAModelComparison() {
  // Which arms are drawn. Defaults to all of them, matching how the page reads on load.
  const [selected, setSelected] = React.useState(() => new Set(ARM_KEYS));
  const SEL = ARM_KEYS.filter((k) => selected.has(k));
  const toggleArm = (k) => setSelected((prev) => {
    const next = new Set(prev);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  });
  const setArms = (keys) => setSelected(new Set(keys));
  const GUE_SHOWN = GUE_ARMS.filter((k) => selected.has(k));
  // Baselines always stay; only our own arms are filtered out of the size chart.
  const PARAM_SHOWN = PARAM_ROWS.filter((r) => !r.mine || selected.has(ARMS.find((a) => a.label === r.model)?.key));

  return (
    <div style={{ background: PAPER, color: INK, minHeight: "100vh", padding: "40px 22px 90px", fontFamily: "Georgia, 'Times New Roman', serif" }}>
      <style>{`
        .serif { font-family: Georgia, 'Times New Roman', serif; }
        @media print {
          @page { size: A4 portrait; margin: 12mm; }
          body { background: #fff !important; }
          .no-print { display: none !important; }
          .pg { break-inside: avoid; page-break-inside: avoid; }
          .recharts-wrapper, table { break-inside: avoid; page-break-inside: avoid; }
          * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        }
      `}</style>

      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        <header style={{ borderBottom: `2px solid ${INK}`, paddingBottom: 14, display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 16, flexWrap: "wrap" }}>
          <div>
            <h1 className="serif" style={{ fontSize: 26, fontWeight: 700, margin: 0, lineHeight: 1.25 }}>
              DNA-KDA arms vs. published DNA foundation models
            </h1>
            <p style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", margin: "8px 0 0" }}>
              {SEL.length} of {ARMS.length} arm{ARMS.length === 1 ? "" : "s"} shown ·{" "}
              {SEL.length ? SEL.map((k) => armLabel(k)).join(" · ") : "none selected"} · RTX 5060 8GB
            </p>
          </div>
          <button
            className="no-print"
            onClick={() => window.print()}
            style={{ fontFamily: MONO, fontSize: 12, padding: "8px 16px", border: `1px solid ${INK}`, background: INK, color: PAPER, borderRadius: 3, cursor: "pointer" }}
          >
            Download PDF
          </button>
        </header>

        <Contents />

        <ArmPicker arms={ARMS} selected={selected} onToggle={toggleArm} onSet={setArms} />

        {/* 00 ------------------------------------------------------------ */}
        <Section num="00" title="What this page is">
          <div style={{ fontSize: 12, lineHeight: 1.7, color: "#3d3a30", margin: "10px 0 18px" }}>
            <p style={{ margin: "0 0 10px" }}>
              <strong>The question.</strong> Does gated delta-rule linear attention (Kimi Delta Attention)
              work as the sequence mixer for a DNA foundation model, and where does its fixed-size recurrent
              state stop being able to recall what it read? Every arm below is the <em>same</em> ~2.25M-parameter
              backbone trained on the same hg38 tokens for the same budget on one RTX 5060 — only the mixer,
              the memory and the positional scheme change. So a difference between two arms is attributable;
              a difference against a published model is not.
            </p>
            <p style={{ margin: "0 0 10px" }}>
              <strong>Three kinds of number appear on this page, and mixing them is the one way to misread it:</strong>
            </p>
            <ol style={{ margin: "0 0 10px 18px", padding: 0 }}>
              <li style={{ marginBottom: 5 }}>
                <strong>Our arms</strong> (coloured, named <code>d01</code>–<code>d07</code>) — measured here,
                sections 01–08 and 10. These are the controlled comparisons.
              </li>
              <li style={{ marginBottom: 5 }}>
                <strong>Released checkpoints run through our harness</strong> — HyenaDNA and Caduceus weights,
                our data and our probe, in <strong>section 09 only</strong>. Same protocol as us, far more
                pretraining compute than us.
              </li>
              <li>
                <strong>Published paper numbers</strong> (grey bars) — transcribed from the HyenaDNA and
                Caduceus tables. Different data pipeline, different budget, fine-tuned. Context, not a contest.
              </li>
            </ol>
            <p style={{ margin: 0 }}>
              <strong>How to use it.</strong> The sticky bar at the top toggles arms; every chart in sections
              01–08 and 10 redraws to just the arms you leave on, so the fastest way to read any panel is to
              switch off everything but the two arms you actually want to compare. Section 09 keeps its own
              fixed arm set. "Download PDF" prints the page with every section expanded.
            </p>
          </div>
          <details style={{ fontSize: 11.5, lineHeight: 1.7, color: "#5c5748", margin: "0 0 16px", border: `1px solid ${GRID}`, borderRadius: 3, padding: "8px 12px" }}>
            <summary style={{ cursor: "pointer", fontFamily: MONO, fontSize: 11, color: "#6b6555" }}>
              glossary — what each metric on this page means
            </summary>
            <ul style={{ margin: "8px 0 0 16px", padding: 0 }}>
              <li><strong>arm</strong> — one trained model variant (one row of the model matrix).</li>
              <li><strong>ctx / context length</strong> — how many nucleotides the model reads at once. Every
                length-swept chart uses the same 8K&rarr;1M ladder; a line that stops early hit an out-of-memory
                wall at the next rung, which is itself a result.</li>
              <li><strong>bits per nucleotide (bpn)</strong> — language-modelling loss. Lower is better;
                2.00 is chance (uniform over A/C/G/T).</li>
              <li><strong>probe vs. finetune</strong> — <em>probe</em> freezes the pretrained model and trains
                only a linear classifier on top: cheap, and a lower bound. <em>Finetune</em> trains the whole
                model per task, which is what the published papers do. Never compare a probe bar with a
                fine-tune bar.</li>
              <li><strong>AUROC / MCC / F1</strong> — classification scores used because most of these tasks
                have lopsided classes, where plain accuracy flatters a model that always guesses the majority.
                Higher is better; 0.5 AUROC and 0 MCC are chance.</li>
              <li><strong>copy advantage</strong> — the recall probe: bits of loss saved when a segment the
                model has already seen is repeated, versus an unseen one. 0 means no evidence the model
                retained it.</li>
              <li><strong>peak VRAM / prefill throughput</strong> — the efficiency claim, measured on the 8GB
                card rather than derived from the complexity class.</li>
            </ul>
          </details>
          <What tone="warn">
            Every "this run" number is measured by <code>scripts/benchmark.py</code> on an RTX 5060 and read
            straight out of <code>results/&lt;arm&gt;/benchmark_cuda*.json</code>. Read the{" "}
            <strong>mode</strong> on every downstream bar before comparing anything.{" "}
            <strong>d06</strong> and <strong>d07</strong> are whole-model <strong>fine-tuned</strong>{" "}
            (<code>benchmarks/finetune.py</code>, via <code>eval_all.sh</code>); every other arm is still a{" "}
            <strong>frozen-backbone linear probe</strong> (<code>benchmarks/probe.py</code>), which is a lower
            bound on fine-tuned accuracy. Probe bars and fine-tune bars are therefore not comparable to each
            other, and only the two fine-tuned arms sit in the same protocol family as the published numbers.
          </What>
          <What tone="warn">
            Two further caveats specific to the fine-tuned arms. First, the <strong>head differs</strong>:
            d07 was fine-tuned with the post-2026-08-30 pooling head (positional bins + a reverse-complement
            pass), d06 with the older mean+last head, so a d06&rarr;d07 gap on a downstream task mixes the head
            change with the architecture change — the splice-site panels in §03 are the extreme case. Second,
            <strong> F1 only exists for d07</strong>: it was added to the eval code after the earlier arms ran,
            so on the three NT tasks whose published metric is F1, d07 plots true F1 and everyone else falls
            back to accuracy. Each bar names the metric it used. Published baselines are fine-tuned under their
            own papers' compute budgets, so nothing here is a controlled comparison against the literature —
            the controlled comparisons are between the arms themselves.
          </What>
          <Table
            columns={["arm", "what it is", "params", "trained ctx", "step", "trained", "source"]}
            highlightRow={(i) => ARMS[i].key === HIGHLIGHT_KEY}
            dimRow={(i) => !selected.has(ARMS[i].key)}
            rows={ARMS.map((a) => [
              <span style={{ color: armColor(a.key), fontWeight: 700 }}>{a.label}</span>,
              <span style={{ whiteSpace: "normal" }}>{a.what}</span>,
              `${(a.params / 1e6).toFixed(2)}M`,
              a.trainedContext.toLocaleString(),
              a.checkpointStep?.toLocaleString() ?? "—",
              <TrainedCell t={a.training} />,
              a.source,
            ])}
          />
          <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8f8268", marginTop: 6, lineHeight: 1.5 }}>
            ★ &nbsp;5B-token run — the planned budget for every arm except the three marked otherwise.<br />
            <span style={{ color: "#a3673f" }}>d01</span> is the exception worth knowing: it is the original{" "}
            120M-parameter model on a <strong>1B</strong>-token budget, so its 99.8% is 1B tokens, not 5B —
            roughly a fifth of the tokens the 2.25M-param arms get. d01_new ran to 2B and d03 to 3B.
            Percentage is always against that arm's own <code>training_tokens</code>.
          </div>
          <What>
            Published baselines come from the two papers this work is positioned against:{" "}
            {Object.values(PAPER_SOURCES).map((s) => s.cite).join("; ")}. Where both report the same model on
            the same task, the higher published value is shown.
          </What>
        </Section>

        {/* 01 ------------------------------------------------------------ */}
        <Section num="01" title="Model size" right="parameters, millions, log scale">
          <What>
            Weight class, for scale. The arms here are ~2M-parameter models trained overnight on one consumer
            GPU; the attention baselines are 100M–500M parameters trained on cluster budgets.
          </What>
          <ResponsiveContainer width="100%" height={Math.max(240, PARAM_SHOWN.length * 26)}>
            <BarChart data={PARAM_SHOWN} layout="vertical" margin={{ top: 4, right: 56, left: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="0" horizontal={false} stroke={GRID} />
              <XAxis type="number" scale="log" domain={[0.2, 900]} tick={tickStyle(10, "#6b6555")} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis type="category" dataKey="model" width={150} tick={tickStyle(10.5)} axisLine={{ stroke: GRID }} tickLine={false} />
              <Tooltip formatter={(v) => [`${v}M params`, ""]} contentStyle={tooltipStyle} />
              <Bar isAnimationActive={false} dataKey="params" barSize={13} radius={[0, 2, 2, 0]}>
                <LabelList dataKey="params" position="right" formatter={(v) => (v >= 10 ? `${Math.round(v)}M` : `${v.toFixed(2)}M`)} style={{ fontFamily: MONO, fontSize: 10, fill: INK }} />
                {PARAM_SHOWN.map((r, i) => <Cell key={i} fill={r.mine ? armColor(ARMS.find((a) => a.label === r.model)?.key) : BASELINE_GREY[i % BASELINE_GREY.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>

        {/* 02 ------------------------------------------------------------ */}
        <Section num="02" title="Genomic Benchmarks — top-1 accuracy" right="8 tasks, 200–4776bp">
          <What tone="key">
            Eight regulatory-element classification tasks (Grešová et al. 2023): given a short DNA sequence,
            is it an enhancer, a promoter, coding, human-vs-worm, and so on. Sequences are 200–4776bp, so this
            measures how good the learned representation is, <em>not</em> long-context ability. All eight
            tasks are shown; baselines are fine-tuned, and each arm's bar is labelled <strong>(probe)</strong>{" "}
            or <strong>(finetune)</strong> per <code>--downstream-mode</code> — <strong>d06</strong> and{" "}
            <strong>d07</strong> have been fine-tuned (<code>eval_all.sh</code>) so far, the rest are still
            frozen-backbone probes. d07 also uses the newer pooling head, so read §00's caveat before treating
            a d06&rarr;d07 gap as architectural.
          </What>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
            {Object.entries(GB_BASELINES).map(([task, spec]) => (
              <div key={task} className="pg">
                <div style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 700, marginBottom: 2 }}>{spec.label}</div>
                <div style={{ fontFamily: MONO, fontSize: 10, color: "#8f8268", marginBottom: 4 }}>{spec.seqLen}bp</div>
                <ScoreBars
                  domain={[0, 1]}
                  rows={[
                    ...SEL.map((k) => ({ model: `${armLabel(k)} (${suiteMode(k, "genomic_benchmarks", task)})`, value: suiteScore(k, "genomic_benchmarks", task, "acc"), armKey: k })).filter((r) => r.value != null),
                    ...spec.rows,
                  ]}
                />
              </div>
            ))}
          </div>
        </Section>

        {/* 03 ------------------------------------------------------------ */}
        <Section num="03" title="Nucleotide Transformer tasks" right="200–600bp, metric varies by task">
          <What tone="key">
            Six of the 18 NT tasks (Dalla-Torre et al. 2023): promoter annotation, enhancer annotation, splice
            sites, and one histone mark. The papers report MCC for enhancer/histone tasks, F1 for promoter and
            splice-donor, accuracy for splice-all. Ours is plotted on the paper's own metric wherever we
            recorded it: MCC and accuracy throughout, and now <strong>true F1</strong> on the three F1 tasks
            for <strong>d07</strong>. F1 was only added to the eval code after d01–d06 were benchmarked, so
            those arms still fall back to accuracy on the F1 panels and are <strong>not</strong> directly
            comparable there — every bar names the metric it actually used. As in §02, only{" "}
            <strong>d06</strong> and <strong>d07</strong> are fine-tuned; every other arm's bar is a linear
            probe — check each bar's own label.
          </What>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
            {Object.entries(NT_BASELINES).map(([task, spec]) => {
              const ours = SEL.map((k) => {
                const m = resolveMetric(k, "nt", task, spec);
                return {
                  model: `${armLabel(k)} (${suiteMode(k, "nt", task)} ${m.key})`,
                  value: m.value, armKey: k, fellBack: m.fellBack,
                };
              }).filter((r) => r.value != null);
              const nBack = ours.filter((r) => r.fellBack).length;
              const matched = ours.length - nBack;
              return (
              <div key={task} className="pg">
                <div style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 700, marginBottom: 2 }}>{spec.label}</div>
                <div style={{ fontFamily: MONO, fontSize: 10, color: nBack ? "#a3673f" : "#8f8268", marginBottom: 4 }}>
                  published metric: {spec.metric}
                  {nBack > 0
                    ? ` — ${matched} of our ${ours.length} bars are ${spec.ourKey}; the other ${nBack} predate ${spec.ourKey} and fall back to ${spec.fallbackKey} (not directly comparable — see each bar's label)`
                    : ""}
                </div>
                <ScoreBars domain={[0, 1]} rows={[...ours, ...spec.rows]} />
              </div>
              );
            })}
          </div>
        </Section>

        {/* 04 ------------------------------------------------------------ */}
        <Section num="04" title="Held-out language modelling, 8K → 1M nucleotides" right="bits per nucleotide, lower is better">
          <What tone="key">
            Next-nucleotide prediction on held-out hg38 chromosomes, swept over context length. Chance for a
            4-letter alphabet is 2.00 bits. A flat or falling line means the model still uses context far past
            where it was trained; a rising line means length generalization is breaking down.
          </What>
          <ArmLines data={LM_BPN} arms={SEL} yLabel="bits / nucleotide" domain={[1.3, 2.05]} fmt={(v) => v?.toFixed(4)} />
          <ReferenceNote />
          <div style={{ marginTop: 18 }} className="pg">
            <div style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", marginBottom: 4 }}>next-nucleotide accuracy (%), same sweep</div>
            <ArmLines data={LM_ACC} height={230} arms={SEL} yLabel="accuracy %" fmt={(v) => `${v?.toFixed(2)}%`} />
          </div>
        </Section>

        {/* 05 ------------------------------------------------------------ */}
        <Section num="05" title="Species classification, by context length" right="5 classes, chance = 20%">
          <What tone="key">
            Classify a genome window as fly / human / mouse / worm / zebrafish, linear probe on mean-pooled
            features. Unlike the fixed-length suites above, the input length is free — so this is one of the
            two places on this page where "does more context actually help on real biology" gets an answer.
          </What>
          {SPECIES_ACC.length ? (
            <ArmLines data={SPECIES_ACC} arms={SEL.filter((k) => (get(k, "species") ?? []).length)} yLabel="accuracy %" domain={[0, 100]} fmt={(v) => `${v?.toFixed(2)}%`} />
          ) : (
            <What>No arm has species data yet — run <code>scripts/fetch_eval_data.py species</code>, then re-benchmark.</What>
          )}
        </Section>

        {/* 06 ------------------------------------------------------------ */}
        <Section num="06" title="Long-range exact recall" right="two protocols — not on the same scale">
          <What tone="warn">
            The mechanistic centrepiece (CLAUDE.md §6.4): can the model still retrieve something it saw N
            nucleotides ago? Linear attention compresses the past into a fixed-size state, so recall should
            fall off past some distance. The protocol changed mid-project, so the two panels are separate and
            must not be read against each other: <strong>recall accuracy</strong> (plant a key, ask for its
            value; chance = 25%) for the earlier arms, and <strong>copy advantage</strong> (bits of
            cross-entropy saved on a repeated segment vs. an unseen one; 0 = no evidence of recall) for the
            newer ones. Each arm is shown at its own trained context length.
          </What>
          {NEEDLE_RECALL.data.length ? (
            <div className="pg">
              <div style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", marginBottom: 4 }}>
                key→value recall accuracy · chance 25% · ctx: {Object.entries(NEEDLE_RECALL.ctxUsed).map(([k, c]) => `${armLabel(k)}=${c}`).join(", ")}
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={NEEDLE_RECALL.data} margin={{ top: 8, right: 18, left: 4, bottom: 4 }}>
                  <CartesianGrid stroke={GRID} />
                  <XAxis dataKey="dist" tick={tickStyle(11)} axisLine={{ stroke: GRID }} tickLine={false} />
                  <YAxis domain={[0, 60]} tick={tickStyle(10.5, "#6b6555")} axisLine={{ stroke: GRID }} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${v?.toFixed(1)}%`} />
                  <Legend wrapperStyle={{ fontFamily: MONO, fontSize: 10.5 }} />
                  <ReferenceLine y={25} stroke="#a3673f" strokeDasharray="4 3" label={{ value: "chance 25%", style: { fontFamily: MONO, fontSize: 10, fill: "#a3673f" } }} />
                  {RECALL_ARMS.filter((k) => selected.has(k)).map((k) => <Line isAnimationActive={false} key={k} type="monotone" dataKey={k} name={armLabel(k)} stroke={armColor(k)} strokeWidth={2} dot={{ r: 2.5 }} />)}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : null}
          {NEEDLE_COPY.data.length ? (
            <div className="pg" style={{ marginTop: 20 }}>
              <div style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", marginBottom: 4 }}>
                repeat-copy advantage (bits saved) · 0 = no recall · ctx: {Object.entries(NEEDLE_COPY.ctxUsed).map(([k, c]) => `${armLabel(k)}=${c}`).join(", ")}
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={NEEDLE_COPY.data} margin={{ top: 8, right: 18, left: 4, bottom: 4 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="dist" tick={tickStyle(11)} axisLine={{ stroke: GRID }} tickLine={false} />
                  <YAxis tick={tickStyle(10.5, "#6b6555")} axisLine={{ stroke: GRID }} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${v?.toFixed(4)} bits`} />
                  <Legend wrapperStyle={{ fontFamily: MONO, fontSize: 10.5 }} />
                  <ReferenceLine y={0} stroke={INK} />
                  {COPY_ARMS.filter((k) => selected.has(k)).map((k) => <Bar isAnimationActive={false} key={k} dataKey={k} name={armLabel(k)} fill={armColor(k)} barSize={18} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : null}
        </Section>

        {/* 07 ------------------------------------------------------------ */}
        <Section num="07" title="Memory and throughput — the O(N) claim, measured" right="RTX 5060, 8GB, bf16">
          <What tone="key">
            Peak VRAM and prefill speed at each context length, on random tokens — this measures the
            architecture, not the weights. Linear attention should show memory growing roughly proportionally
            with length instead of quadratically, which is what lets these arms reach 1M nucleotides on an 8GB
            card at all. A context an arm OOM'd at is simply absent from its line.
          </What>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
            <div className="pg">
              <div style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", marginBottom: 4 }}>inference peak VRAM (GB)</div>
              <ArmLines data={SCALING_INFER} height={240} arms={SEL} logY yLabel="GB" fmt={(v) => `${v?.toFixed(3)} GB`} />
            </div>
            <div className="pg">
              <div style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", marginBottom: 4 }}>training peak VRAM (GB)</div>
              <ArmLines data={SCALING_TRAIN} height={240} arms={SEL} logY yLabel="GB" fmt={(v) => `${v?.toFixed(3)} GB`} />
            </div>
            <div className="pg">
              <div style={{ fontFamily: MONO, fontSize: 11, color: "#6b6555", marginBottom: 4 }}>prefill throughput (tokens/s)</div>
              <ArmLines data={SCALING_TPS} height={240} arms={SEL} logY yLabel="tok/s" fmt={(v) => `${Math.round(v).toLocaleString()} tok/s`} />
            </div>
          </div>
        </Section>

        {/* 08 ------------------------------------------------------------ */}
        <Section num="08" title="Genomics Long-Range Benchmark" right="free-length genome windows">
          <What tone="key">
            Real biology on windows cut straight from the hg38 FASTA, so the input length is free: variant
            effect prediction (eQTL), chromatin accessibility, enhancer and promoter annotation. Together with
            section 05 this is where extra context can actually pay off. Chromatin is scored by mean AUROC
            across 37 tracks because raw accuracy there is dominated by the majority class.
          </What>
          {LRB_TASKS.length ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
              {LRB_TASKS.map((task) => {
                const field = task === "chromatin" ? "mean_auroc" : "auroc";
                const data = lrbByCtx(task, field);
                if (!data.length) return null;
                return (
                  <div key={task} className="pg">
                    <div style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 700, marginBottom: 4 }}>{task} · {field === "mean_auroc" ? "mean AUROC" : "AUROC"}</div>
                    <ArmLines data={data} height={230} arms={SEL.filter((k) => (get(k, "lrb")?.[task] ?? []).length)} domain={[0.4, 0.8]} fmt={(v) => v?.toFixed(3)} />
                  </div>
                );
              })}
            </div>
          ) : (
            <What>No arm has LRB data — it needs <code>scripts/fetch_eval_data.py hg38</code> before benchmarking.</What>
          )}
        </Section>

        {/* 09 ------------------------------------------------------------ */}
        {PANEL_KEYS.length ? (
          <Section num="09" title="Head-to-head with the released HyenaDNA / Caduceus checkpoints" right="our harness, frozen probe, species + GLRB">
            <What tone="key">
              Everywhere else on this page, HyenaDNA and Caduceus appear only as numbers transcribed from
              their papers. Here they are the <strong>actual released checkpoints</strong>, downloaded and run
              through <em>our</em> harness by <code>scripts/baselines_panel.sh</code> — same hg38 windows, same
              seeds, same frozen linear probe, same bootstrap — next to our own arms re-measured in that same
              pass. That makes everything comparable except the one thing that cannot be matched:{" "}
              <strong>their pretraining budget is orders of magnitude larger than ours</strong>, and they are
              3–4&times; our parameter count. Read a gap here as "what a big-budget checkpoint buys", never as
              an architecture verdict.
            </What>
            <What tone="warn">
              These rows are a <strong>separate measurement</strong> from the rest of the page
              (<code>panel_cuda.json</code>, different context points and resampling than{" "}
              <code>benchmark_cuda.json</code>), so <code>d06 (panel)</code> and plain <code>d06</code> can and
              do differ by a point or two. Compare panel rows with panel rows only — which is why this section
              has its own arm set and ignores the picker at the top. Caduceus is a bidirectional masked LM, so
              it has no language-modelling or needle rows anywhere on this page; the probe benchmarks below are
              the whole of its comparison.
            </What>
            <Table
              columns={["arm", "kind", "params", "trained ctx", "what it is"]}
              rows={PANEL_ARMS.map((a) => [
                <span style={{ color: armColor(a.key), fontWeight: 700 }}>{a.label}</span>,
                a.external ? "released checkpoint" : "ours",
                `${(a.params / 1e6).toFixed(2)}M`,
                a.trainedContext.toLocaleString(),
                <span style={{ whiteSpace: "normal" }}>{a.what}</span>,
              ])}
            />
            <div style={{ marginTop: 18 }}>
              <div style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 700, marginBottom: 4 }}>
                species classification · accuracy %, chance = 20%
              </div>
              {PANEL_SPECIES.length ? (
                <ArmLines
                  data={PANEL_SPECIES} yLabel="accuracy %" domain={[0, 100]}
                  arms={PANEL_KEYS.filter((k) => (get(k, "species") ?? []).length)}
                  fmt={(v) => `${v?.toFixed(2)}%`}
                />
              ) : (
                <What>No species rows in the panel yet.</What>
              )}
              <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8f8268", marginTop: 2 }}>
                a line that stops early ran out of VRAM at the next length — for the external arms that ceiling
                is itself the result (no exact chunked-prefill path exists for them, so <code>--offload</code>
                is not available and the OOM is real).
              </div>
            </div>
            {PANEL_LRB_TASKS.length ? (
              <div style={{ marginTop: 20, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
                {PANEL_LRB_TASKS.map((task) => {
                  const data = lrbByCtx(task, lrbField(task), PANEL_KEYS);
                  if (!data.length) return null;
                  return (
                    <div key={task} className="pg">
                      <div style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 700, marginBottom: 4 }}>
                        {task} · {lrbField(task) === "mean_auroc" ? "mean AUROC" : "AUROC"}
                      </div>
                      <ArmLines
                        data={data} height={230} domain={[0.4, 0.8]} fmt={(v) => v?.toFixed(3)}
                        arms={PANEL_KEYS.filter((k) => (get(k, "lrb")?.[task] ?? []).length)}
                      />
                    </div>
                  );
                })}
              </div>
            ) : null}
            <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8f8268", marginTop: 8 }}>
              source — {PANEL_ARMS.map((a) => a.source).join(", ")}
            </div>
          </Section>
        ) : null}

        {/* 10 ------------------------------------------------------------ */}
        {GUE_SHOWN.length ? (
          <Section num="10" title="GUE (DNABERT-2 suite)" right="70–1000bp, MCC per task">
            <What>
              The 37-task DNABERT-2 evaluation suite, fetched from the <code>leannmlindsey/GUE</code>{" "}
              HuggingFace mirror (cite it as a community mirror, not the official release). Shown as MCC per task
              for the arms that ran it; no published baselines are transcribed here. Read the mode listed
              under the table before comparing columns — a fine-tuned column is not comparable to a probe one.
            </What>
            <Table
              columns={["task", ...GUE_SHOWN.map((k) => `${armLabel(k)} MCC`)]}
              rows={GUE_TASKS.map((t) => {
                // Best MCC on this task, so the winning cell can be marked. Only worth
                // marking when at least two arms actually have a number for the task.
                const vals = GUE_SHOWN.map((k) => suiteScore(k, "gue", t, "mcc"));
                const present = vals.filter((v) => v != null);
                const best = present.length > 1 ? Math.max(...present) : null;
                return [
                  t.replace(/^GUE\//, ""),
                  ...vals.map((v) => (v == null ? "—" : <Best on={v === best}>{v.toFixed(3)}</Best>)),
                ];
              })}
            />
            <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8f8268", marginTop: 6 }}>
              highlighted = best MCC on that task among the arms shown ·{" "}
              {GUE_SHOWN.map((k) => `${armLabel(k)} ${gueMode(k)}`).join(" · ")}
            </div>
          </Section>
        ) : null}

        {/* ---------------------------------------------------------------- */}
        <footer style={{ marginTop: 44, borderTop: `1px solid ${GRID}`, paddingTop: 12, fontFamily: MONO, fontSize: 10.5, color: "#8f8268" }}>
          Sources — {Object.values(PAPER_SOURCES).map((s) => s.cite).join(" · ")}. Our numbers:{" "}
          {ARMS.map((a) => a.source).join(", ")}.
        </footer>
      </div>
    </div>
  );
}

// Training progress for one arm: percent of its own planned token budget, with a
// star for the arms on the standard 5B-token run. The denominator is per-arm on
// purpose — d01 is a 120M-param model on a 1B budget and would otherwise read as
// "done" next to 2.25M-param arms that have seen five times the tokens.
const FULL_RUN_TOKENS = 5e9;

function TrainedCell({ t }) {
  if (!t) return <span style={{ color: "#8f8268" }}>—</span>;
  const standard = t.tokensPlanned === FULL_RUN_TOKENS;
  const done = t.pct >= 99.5;
  return (
    <span style={{ fontFamily: MONO, whiteSpace: "nowrap" }}>
      <span style={{ display: "inline-block", width: 46, height: 7, background: "#e2ddd0",
                     borderRadius: 2, overflow: "hidden", verticalAlign: "middle", marginRight: 7 }}>
        <span style={{ display: "block", width: `${Math.min(t.pct, 100)}%`, height: "100%",
                       background: done ? "#2f6f5e" : "#a3673f" }} />
      </span>
      <span style={{ fontWeight: 700 }}>{t.pct}%</span>
      <span style={{ color: "#8f8268" }}>
        {" "}({(t.tokensDone / 1e9).toFixed(2)}B/{(t.tokensPlanned / 1e9).toFixed(0)}B)
      </span>
      {standard ? " ★" : ""}
    </span>
  );
}

// Arm picker. Every chart draws the arms selected here; deselecting is how you get a
// clean two-way read (e.g. d06 vs d07 alone) out of a page that otherwise overlays all
// of them. Published baselines are never filtered — they are the fixed reference.
function ArmPicker({ arms, selected, onToggle, onSet }) {
  const btn = (label, onClick, on) => (
    <button onClick={onClick} style={{
      fontFamily: MONO, fontSize: 10.5, padding: "3px 9px", borderRadius: 3, cursor: "pointer",
      border: `1px solid ${on ? INK : GRID}`, background: on ? INK : "transparent",
      color: on ? PAPER : "#6b6555" }}>{label}</button>
  );
  return (
    <div className="no-print" style={{
      position: "sticky", top: 0, zIndex: 20, background: PAPER, borderBottom: `1px solid ${GRID}`,
      padding: "10px 0", marginTop: 10, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <span style={{ fontFamily: MONO, fontSize: 10.5, color: "#6b6555", textTransform: "uppercase", letterSpacing: ".08em" }}>
        compare
      </span>
      {arms.map((a) => {
        const on = selected.has(a.key);
        return (
          <button key={a.key} onClick={() => onToggle(a.key)} title={a.what}
            aria-pressed={on}
            style={{
              fontFamily: MONO, fontSize: 11, padding: "4px 10px", borderRadius: 3, cursor: "pointer",
              border: `1px solid ${on ? armColor(a.key) : GRID}`,
              background: on ? armColor(a.key) : "transparent",
              color: on ? PAPER : "#8f8268",
              fontWeight: a.key === HIGHLIGHT_KEY ? 700 : 400,
            }}>
            {a.label}{a.training ? ` ${a.training.pct}%` : ""}
          </button>
        );
      })}
      <span style={{ width: 10 }} />
      {btn("all", () => onSet(arms.map((a) => a.key)), false)}
      {btn("none", () => onSet([]), false)}
      {btn(`only ${HIGHLIGHT_KEY}`, () => onSet([HIGHLIGHT_KEY]), false)}
      {btn("finetuned", () => onSet(FINETUNED_ARMS), false)}
    </div>
  );
}

// Chance line reminder used under the language-modelling chart.
function ReferenceNote() {
  return (
    <div style={{ fontFamily: MONO, fontSize: 10.5, color: "#8f8268", marginTop: 2 }}>
      chance = 2.00 bits/nucleotide (uniform over A/C/G/T)
    </div>
  );
}
