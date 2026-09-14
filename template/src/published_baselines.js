/**
 * published_baselines.js
 * ---------------------------------------------------------------------------
 * Published, fine-tuned scores for the DNA models this page compares against.
 * Every number is transcribed from a paper table — sources are named per block
 * so any value can be checked against the PDF.
 *
 *   [H]  HyenaDNA, arXiv:2306.15794 — Table 4.1 (Genomic Benchmarks, top-1 %)
 *                                     Table 4.2 (NT benchmarks, MCC / F1)
 *   [C]  Caduceus, arXiv:2403.03234 — Table 1 (Genomic Benchmarks, 5-fold CV acc)
 *                                     Table 2 (NT tasks, 10-fold CV, metric per task)
 *
 * Where both papers report the same model on the same task, the BEST published
 * value is kept and `src` records which paper it came from. Caduceus re-trains
 * HyenaDNA at 436K params and gets lower numbers than the HyenaDNA paper's own
 * 1.6M-param fine-tune; the higher (original-paper) value is the one shown.
 */

// Model -> parameter count in millions, for the size axis and chart labels.
export const BASELINE_PARAMS = {
  "CNN (scratch)": 0.264,
  "HyenaDNA": 1.6,
  "Mamba (NTP)": 0.468,
  "Caduceus-Ph": 0.47,
  "Caduceus-PS": 0.47,
  "DNABERT": 110,
  "DNABERT-2": 117,
  "NT-v2 (500M)": 500,
  "Enformer": 252,
};

// --- Genomic Benchmarks: top-1 accuracy (fraction). ------------------------
// CNN/HyenaDNA/DNABERT columns: [H] Table 4.1. Mamba/Caduceus: [C] Table 1.
// CNN keeps the higher of [H] 4.1 and [C] Table 1 per task.
export const GB_BASELINES = {
  mouse_enhancers:        { label: "Mouse Enhancers",          seqLen: 4776, rows: [
    { model: "CNN (scratch)", value: 0.715, src: "C" }, { model: "HyenaDNA", value: 0.851, src: "H" },
    { model: "Mamba (NTP)", value: 0.743, src: "C" }, { model: "Caduceus-Ph", value: 0.754, src: "C" },
    { model: "Caduceus-PS", value: 0.793, src: "C" }, { model: "DNABERT", value: 0.669, src: "H" } ] },
  coding_vs_intergenomic: { label: "Coding vs Intergenomic",   seqLen: 200, rows: [
    { model: "CNN (scratch)", value: 0.892, src: "C" }, { model: "HyenaDNA", value: 0.913, src: "H" },
    { model: "Mamba (NTP)", value: 0.904, src: "C" }, { model: "Caduceus-Ph", value: 0.915, src: "C" },
    { model: "Caduceus-PS", value: 0.910, src: "C" }, { model: "DNABERT", value: 0.925, src: "H" } ] },
  human_vs_worm:          { label: "Human vs Worm",            seqLen: 200, rows: [
    { model: "CNN (scratch)", value: 0.942, src: "C" }, { model: "HyenaDNA", value: 0.966, src: "H" },
    { model: "Mamba (NTP)", value: 0.967, src: "C" }, { model: "Caduceus-Ph", value: 0.973, src: "C" },
    { model: "Caduceus-PS", value: 0.968, src: "C" }, { model: "DNABERT", value: 0.965, src: "H" } ] },
  human_enhancers_cohn:   { label: "Human Enhancers (Cohn)",   seqLen: 500, rows: [
    { model: "CNN (scratch)", value: 0.702, src: "C" }, { model: "HyenaDNA", value: 0.742, src: "H" },
    { model: "Mamba (NTP)", value: 0.732, src: "C" }, { model: "Caduceus-Ph", value: 0.747, src: "C" },
    { model: "Caduceus-PS", value: 0.745, src: "C" }, { model: "DNABERT", value: 0.740, src: "H" } ] },
  human_enhancers_ensembl:{ label: "Human Enhancers (Ensembl)",seqLen: 573, rows: [
    { model: "CNN (scratch)", value: 0.744, src: "C" }, { model: "HyenaDNA", value: 0.892, src: "H" },
    { model: "Mamba (NTP)", value: 0.862, src: "C" }, { model: "Caduceus-Ph", value: 0.893, src: "C" },
    { model: "Caduceus-PS", value: 0.900, src: "C" }, { model: "DNABERT", value: 0.857, src: "H" } ] },
  human_regulatory:       { label: "Human Regulatory",         seqLen: 802, rows: [
    { model: "CNN (scratch)", value: 0.933, src: "H" }, { model: "HyenaDNA", value: 0.938, src: "H" },
    { model: "Mamba (NTP)", value: 0.814, src: "C" }, { model: "Caduceus-Ph", value: 0.872, src: "C" },
    { model: "Caduceus-PS", value: 0.873, src: "C" }, { model: "DNABERT", value: 0.881, src: "H" } ] },
  human_nontata_promoters:{ label: "Human NonTATA Promoters",  seqLen: 251, rows: [
    { model: "CNN (scratch)", value: 0.861, src: "C" }, { model: "HyenaDNA", value: 0.966, src: "H" },
    { model: "Mamba (NTP)", value: 0.933, src: "C" }, { model: "Caduceus-Ph", value: 0.946, src: "C" },
    { model: "Caduceus-PS", value: 0.945, src: "C" }, { model: "DNABERT", value: 0.856, src: "H" } ] },
  human_ocr_ensembl:      { label: "Human OCR (Ensembl)",      seqLen: 593, rows: [
    { model: "CNN (scratch)", value: 0.698, src: "C" }, { model: "HyenaDNA", value: 0.809, src: "H" },
    { model: "Mamba (NTP)", value: 0.815, src: "C" }, { model: "Caduceus-Ph", value: 0.828, src: "C" },
    { model: "Caduceus-PS", value: 0.818, src: "C" }, { model: "DNABERT", value: 0.751, src: "H" } ] },
};

// --- Nucleotide Transformer tasks. -----------------------------------------
// `metric` is the metric the PAPERS report for that task. `ourKey` is the metric
// of ours plotted on the same axis; `fallbackKey` is used for arms whose benchmark
// run predates that metric being recorded, and any bar falling back is labelled
// as not-comparable. F1 was only added to benchmarks/{probe,finetune}.py after the
// earlier arms were run, so on the F1 tasks d07 plots true F1 while d01-d06 fall
// back to accuracy — hence the per-bar metric name in each label.
// Enformer/DNABERT-2/NT-v2/Caduceus: [C] Table 2. HyenaDNA: best of [H] 4.2 and [C] 2.
export const NT_BASELINES = {
  enhancers:           { label: "Enhancer annotation", metric: "MCC", ourKey: "mcc", rows: [
    { model: "Enformer", value: 0.451, src: "C" }, { model: "DNABERT-2", value: 0.516, src: "C" },
    { model: "NT-v2 (500M)", value: 0.548, src: "C" }, { model: "HyenaDNA", value: 0.626, src: "H" },
    { model: "Caduceus-Ph", value: 0.546, src: "C" }, { model: "Caduceus-PS", value: 0.491, src: "C" } ] },
  H3K4me3:             { label: "H3K4me3 (histone mark)", metric: "MCC", ourKey: "mcc", rows: [
    { model: "Enformer", value: 0.158, src: "C" }, { model: "DNABERT-2", value: 0.352, src: "C" },
    { model: "NT-v2 (500M)", value: 0.410, src: "C" }, { model: "HyenaDNA", value: 0.612, src: "H" },
    { model: "Caduceus-Ph", value: 0.544, src: "C" }, { model: "Caduceus-PS", value: 0.440, src: "C" } ] },
  promoter_all:        { label: "Promoter: all", metric: "F1", ourKey: "f1", fallbackKey: "acc", rows: [
    { model: "Enformer", value: 0.954, src: "C" }, { model: "DNABERT-2", value: 0.971, src: "C" },
    { model: "NT-v2 (500M)", value: 0.976, src: "C" }, { model: "HyenaDNA", value: 0.965, src: "H" },
    { model: "Caduceus-Ph", value: 0.970, src: "C" }, { model: "Caduceus-PS", value: 0.967, src: "C" } ] },
  promoter_no_tata:    { label: "Promoter: non-TATA", metric: "F1", ourKey: "f1", fallbackKey: "acc", rows: [
    { model: "Enformer", value: 0.955, src: "C" }, { model: "DNABERT-2", value: 0.972, src: "C" },
    { model: "NT-v2 (500M)", value: 0.976, src: "C" }, { model: "HyenaDNA", value: 0.966, src: "H" },
    { model: "Caduceus-Ph", value: 0.969, src: "C" }, { model: "Caduceus-PS", value: 0.968, src: "C" } ] },
  splice_sites_donors: { label: "Splice site: donor", metric: "F1", ourKey: "f1", fallbackKey: "acc", rows: [
    { model: "Enformer", value: 0.906, src: "C" }, { model: "DNABERT-2", value: 0.963, src: "C" },
    { model: "NT-v2 (500M)", value: 0.985, src: "C" }, { model: "HyenaDNA", value: 0.973, src: "H" },
    { model: "Caduceus-Ph", value: 0.948, src: "C" }, { model: "Caduceus-PS", value: 0.874, src: "C" } ] },
  splice_sites_all:    { label: "Splice site: all (3-class)", metric: "accuracy", ourKey: "acc", rows: [
    { model: "Enformer", value: 0.848, src: "C" }, { model: "DNABERT-2", value: 0.939, src: "C" },
    { model: "NT-v2 (500M)", value: 0.983, src: "C" }, { model: "HyenaDNA", value: 0.979, src: "H" },
    { model: "Caduceus-Ph", value: 0.940, src: "C" }, { model: "Caduceus-PS", value: 0.927, src: "C" } ] },
};

export const PAPER_SOURCES = {
  H: { label: "HyenaDNA", cite: "Nguyen et al. 2023, arXiv:2306.15794 (Tables 4.1, 4.2)" },
  C: { label: "Caduceus", cite: "Schiff et al. 2024, arXiv:2403.03234 (Tables 1, 2)" },
};
