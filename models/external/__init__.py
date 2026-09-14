"""External *pretrained* reference arms (HyenaDNA, Caduceus-Ph/PS), wrapped so they run through
the same benchmark harness as D01-D07 — same windows, same seeds, same probe/head, same
bootstrap.

These are NOT matched arms: they were pretrained on a different token/compute budget (HyenaDNA
~5B tokens of hg38 at 1k-1M context; Caduceus 131k-context hg38), so every table containing
them must say so. What they *are* is the apples-to-apples half of the comparison that published
numbers cannot give: identical evaluation code, identical windows, identical linear probe.

Two things differ per family and are handled here rather than in the benchmarks:

* **Vocabulary.** Each release has its own char-tokenizer ordering. Inputs are remapped
  id->id on the way in and LM logits are gathered back into `dataset.tokenizer.VOCAB` order on
  the way out, so a benchmark indexing `logits[..., TOKEN_TO_ID['A']]` is unchanged.
* **Directionality.** HyenaDNA is causal, so `lm` (next-nucleotide CE) is meaningful. Caduceus
  is a bidirectional masked LM: a causal-CE number from it would be neither its training
  objective nor comparable to ours, so `lm`/`needle` raise SkipBenchmark instead of producing a
  misleading row. The probe benchmarks (species, lrb, the fixed-length suites) are unaffected —
  they read pooled hidden states, which is exactly how these models are used downstream.

Config keys: `external_family` (hyenadna|caduceus), `hf_model_id`.
"""
import torch
import torch.nn as nn

from benchmarks.harness import SkipBenchmark
from dataset.tokenizer import VOCAB, VOCAB_SIZE


class _ExternalWrapper(nn.Module):
    """Shared surface: `.config`, `.tok_emb`, `.encode(ids, layers, reduce)`, `forward -> (logits, None)`."""

    causal = True

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.config["vocab_size"] = VOCAB_SIZE
        self.gradient_checkpointing = False

    def _vocab_map(self, hf_vocab: dict, aliases: dict):
        """our id -> hf id, falling back through `aliases` then to the model's unk/pad."""
        unk = next((hf_vocab[k] for k in ("[UNK]", "<unk>", "[MASK]", "[PAD]") if k in hf_vocab), 0)
        ids = []
        for tok in VOCAB:
            cand = [tok] + list(aliases.get(tok, ()))
            ids.append(next((hf_vocab[c] for c in cand if c in hf_vocab), unk))
        return torch.tensor(ids, dtype=torch.long)

    @property
    def tok_emb(self):
        return self.hf.get_input_embeddings()

    def _hidden(self, ids, want_all: bool):
        raise NotImplementedError

    def encode(self, ids: torch.Tensor, layers=None, reduce=None):
        if layers is None:
            return self._hidden(ids, want_all=False)
        reduce = reduce or (lambda h: h)
        hs = self._hidden(ids, want_all=True)
        n = len(hs) - 1                      # entry 0 is the embedding output
        idx = sorted({(l + n if l < 0 else l) for l in layers})
        if not all(0 <= l < n for l in idx):
            raise ValueError(f"layer index out of range for {n} blocks")
        return [reduce(hs[l + 1]) for l in idx]

    def forward(self, ids: torch.Tensor, targets=None):
        raise NotImplementedError


class HyenaDNAWrapper(_ExternalWrapper):
    DEFAULT_HF_ID = "LongSafari/hyenadna-medium-450k-seqlen-hf"

    def __init__(self, config: dict):
        super().__init__(config)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = config.get("hf_model_id", self.DEFAULT_HF_ID)
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if config.get("random_init"):
            # Same architecture and tokenizer, freshly initialized weights: the
            # "does pretraining help or hurt" control, matched to the released checkpoint
            # in everything except training.
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            self.hf = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True).eval()
        else:
            self.hf = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True).eval()
        vmap = self._vocab_map(tok.get_vocab(), {})
        self.register_buffer("in_map", vmap, persistent=False)
        self.register_buffer("out_map", vmap, persistent=False)
        self.config["d_model"] = int(self.hf.config.d_model)

    def _hidden(self, ids, want_all: bool):
        out = self.hf.hyena(input_ids=self.in_map[ids], output_hidden_states=want_all)
        return out.hidden_states if want_all else out.last_hidden_state

    def forward(self, ids: torch.Tensor, targets=None):
        logits = self.hf(input_ids=self.in_map[ids]).logits
        return logits[..., self.out_map], None


def _mamba_transformers_shim():
    """mamba-ssm's `utils.generation` imports `GreedySearchDecoderOnlyOutput` /
    `SampleDecoderOnlyOutput`, which transformers renamed to `GenerateDecoderOnlyOutput` in
    v4.41. Caduceus' remote code imports mamba_ssm at module scope, so without this alias the
    checkpoint cannot be loaded at all under transformers>=5. Only generation-output *type
    aliases* are added — no behaviour is patched, and the Caduceus forward path never uses
    them."""
    from transformers import generation
    if not hasattr(generation, "GreedySearchDecoderOnlyOutput"):
        target = generation.GenerateDecoderOnlyOutput
        for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput",
                     "GreedySearchEncoderDecoderOutput", "SampleEncoderDecoderOutput"):
            if not hasattr(generation, name):
                setattr(generation, name, target)


def _caduceus_class(hf_id: str, name: str = "CaduceusForMaskedLM"):
    """Resolve (and repair) the checkpoint's remote model class.

    `CaduceusForMaskedLM`, not the bare backbone: the released weights are all prefixed
    `caduceus.`, so loading the backbone class alone silently randomly-initialises every
    parameter (transformers only warns).

    The released code was written against transformers 4.38 and overrides `tie_weights`
    without calling `super()` (and skips it entirely when the config unties the embeddings),
    so under transformers>=5 the loader trips on the `all_tied_weights_keys` mapping that
    `tie_weights` is now expected to populate. The patch gives every instance an empty mapping — correct here, because the plain
    `Caduceus` backbone (no LM head) ties nothing."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(f"modeling_caduceus.{name}", hf_id)
    if not getattr(cls, "_dna_kda_patched", False):
        original_init = cls.__init__

        def __init__(self, *a, **kw):
            original_init(self, *a, **kw)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        original_tie = cls.tie_weights

        def tie_weights(self, *a, **kw):
            # transformers>=5 calls tie_weights(missing_keys=..., recompute_mapping=...);
            # the released method takes no arguments and only mirrors the bidirectional
            # Mamba in/out projections, which is argument-independent.
            return original_tie(self)

        cls.__init__ = __init__
        cls.tie_weights = tie_weights
        cls._dna_kda_patched = True
    return cls


class CaduceusWrapper(_ExternalWrapper):
    """Caduceus-Ph / Caduceus-PS (kuleshov-group). Bidirectional Mamba masked LM."""

    DEFAULT_HF_ID = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    causal = False

    def __init__(self, config: dict):
        super().__init__(config)
        from transformers import AutoTokenizer

        _mamba_transformers_shim()
        hf_id = config.get("hf_model_id", self.DEFAULT_HF_ID)
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.hf = _caduceus_class(hf_id).from_pretrained(hf_id, trust_remote_code=True).eval()
        self.backbone = self.hf.caduceus
        # Caduceus' char tokenizer spells its specials <cls>/<pad>/<unk>-style, and has no BOS.
        aliases = {"[PAD]": ("<pad>", "[CLS]", "<cls>"), "[BOS]": ("[CLS]", "<cls>", "[BOS]"),
                   "[UNK]": ("<unk>", "[UNK]", "[MASK]", "<mask>")}
        self.register_buffer("in_map", self._vocab_map(tok.get_vocab(), aliases),
                             persistent=False)
        # Caduceus-PS wraps every layer in RCPS (reverse-complement parameter sharing), which
        # concatenates the forward and RC channel blocks — its hidden states are 2*d_model
        # wide. Read from the config, not from a probe forward: the released code's fused
        # add-norm is a Triton kernel and cannot run on CPU at all.
        self.config["d_model"] = int(self.hf.config.d_model) * (
            2 if getattr(self.hf.config, "rcps", False) else 1)

    @property
    def tok_emb(self):
        return self.backbone.backbone.embeddings.word_embeddings

    def _hidden(self, ids, want_all: bool):
        out = self.backbone(input_ids=self.in_map[ids], output_hidden_states=want_all)
        if want_all:
            hs = getattr(out, "hidden_states", None)
            if not hs:
                raise SkipBenchmark("Caduceus backbone returned no per-layer hidden states")
            return hs
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def forward(self, ids: torch.Tensor, targets=None):
        raise SkipBenchmark(
            "Caduceus is a bidirectional masked LM: causal next-nucleotide CE is neither its "
            "objective nor comparable to the causal arms. Run it on the probe benchmarks "
            "(species, lrb, downstream) instead.")


FAMILIES = {"hyenadna": HyenaDNAWrapper, "caduceus": CaduceusWrapper}


def build_model(config: dict):
    family = config.get("external_family")
    if family not in FAMILIES:
        raise SystemExit(f"config needs external_family in {sorted(FAMILIES)}, got {family!r}")
    return FAMILIES[family](config)
