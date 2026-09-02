# LIMITATIONS

Honest statement of what this system does badly and what the evaluation does
not cover. Anything not listed in RESULTS.md as a measured number is not a
claim.

## Evaluation data

- **Single-word printed images only.** The entire quantitative evaluation is
  APTI: 2,000 synthetic printed Arabic word crops. Consequences:
  - **WER is meaningless here** and is deliberately suppressed (it
    degenerates to exact-match on single words; the v2 project reported
    WER=1.01 on this data, which was an artifact).
  - **Segmentation and layout stages are untested by construction** — a word
    crop has nothing to segment. The RTL region-ordering logic is unit-tested
    on synthetic geometry only.
  - Slice reporting covers word length and digit presence; there is no
    document-type or scan-quality axis because the corpus has none.
- **No handwritten evaluation.** KHATT is registration-gated and absent.
  The two screened community TrOCR checkpoints are handwriting-trained, so
  their poor APTI numbers say nothing about their handwriting performance.
- **MSA only.** Normalization (v1.1.0) folds orthographic variants that are
  safe for MSA. Per the evaluation protocol, these rules are NOT validated
  for dialectal text and would collapse meaningful dialectal distinctions.
- **Ground truth is undiacritized single tokens**; diacritic-recognition
  quality is therefore unmeasured (scoring strips diacritics; output keeps
  them).

## Models

- **LayoutLMv3 is integrated but unevaluated.** No labeled Arabic
  field-extraction data exists in this project. The wrapper is smoke-tested
  (embeddings come back with the right shape, apply_ocr disabled so its
  non-Arabic internal OCR can never run silently). **No field-F1 is claimed
  anywhere.**
- **Qwen2-VL-2B stands in for "Qwen-VL"** (the original checkpoint is
  deprecated upstream and too large for the 16 GB target machine).
- **TrOCR has no official Arabic checkpoint.** Community checkpoints were
  screened and all underperformed EasyOCR on printed APTI (see RESULTS);
  TrOCR is therefore integrated but not part of the winning configuration.
  In-domain fine-tuning was NOT attempted (no training budget in scope).
- **Confidence signals**: EasyOCR's confidence is nearly uninformative on
  this data (r≈0.09 with correctness) and is never used for routing.
  Qwen2-VL's mean-token-logprob is moderately informative (r≈0.48) and is
  used, with a threshold chosen on dev.

## Page-level reading

- The `page` engine (line segmentation + per-region arbitration) has NO
  quantitative evaluation — no page-level Arabic ground truth exists in
  this project. Its line-crop confidence bar (0.70) was chosen
  qualitatively on a single Sudanese-dialect poetry screenshot; treat it
  as a demo default, not a measured result.
- Qwen2-VL reads full pages in visual (reversed) order; the bidi repair
  recovers characters per line but, when the model emits no newlines,
  whole-text reversal flips line ORDER. The page engine avoids this
  entirely by segmenting first.

## Method

- The arbitration policy (primary/fallback, τ=0.40, degeneration flag
  thresholds) was tuned on the 200-sample dev split. Golden is a regression
  set; the held-out split is scored once, after all decisions were frozen.
  Numbers on dev are optimistic by construction; cite held-out numbers.
- The degeneration flag v2 thresholds were chosen on dev (0 false alarms)
  but the *motivation* for revisiting them came from a failure observed in
  golden; this is disclosed in RESULTS and the held-out run arbitrates.
- Latency numbers are single-machine (Apple M2 Pro, MPS) medians; no
  batching, no throughput optimization was attempted.
- The self-correction (CamelBERT reranking) ships disabled unless its dev
  fix/break balance is net-positive with CI excluding zero — see RESULTS
  for the measured outcome.
