# LIMITATIONS

Honest statement of what this system does badly and what the evaluation does
not cover. Anything not listed in RESULTS.md as a measured number is not a
claim.

## Evaluation data

- **Provenance is not fully established.** The 2,000 images are the first
  rows of the Hugging Face dataset `mssqpi/Arabic-OCR-Dataset`, downloaded in
  May 2026 and stored under a folder named `apti`. That name is a legacy
  label; there is no evidence the images come from the APTI database. The
  source dataset was no longer publicly accessible on 30 September 2026, so
  its license and generation process could not be confirmed and the images
  are not redistributed. Reproduction is limited to rescoring the published
  run files.
- **Single-word printed images only.** The entire quantitative evaluation is
  these 2,000 synthetic printed Arabic word crops, 35 pixels high, with
  labels of 7 to 10 characters. Consequences:
  - **WER is meaningless here** and is deliberately suppressed (it
    degenerates to exact-match on single words; the v2 project reported
    WER=1.01 on this data, which was an artifact).
  - **Segmentation and layout stages are untested by construction** — a word
    crop has nothing to segment. The RTL region-ordering logic is unit-tested
    on synthetic geometry only.
  - Slice reporting covers word length and digit presence; there is no
    document-type or scan-quality axis because the corpus has none.
- **No handwritten evaluation.** KHATT is registration-gated and absent.
  At least one of the four screened community TrOCR checkpoints is
  handwriting-trained, so poor numbers on this data say nothing about
  handwriting performance.
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
  screened on ten images and all underperformed EasyOCR (see RESULTS);
  TrOCR is therefore integrated but not part of the winning configuration.
  In-domain fine-tuning was NOT attempted (no training budget in scope).
- **Confidence signals**: EasyOCR's confidence is weakly related to
  correctness on this data (r between 0.09 and 0.28 across splits) and is
  never used for routing; whether adding it would help was not tested.
  Qwen2-VL's confidence is moderately informative (r about 0.46 to 0.50) and
  is used, with a threshold chosen on dev.
- **One prompt, one checkpoint.** Only one Arabic prompt was tried and the
  Qwen2-VL revision was not pinned. The rate of refusals and prompt echoes
  may depend on both.

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

- Only the confidence threshold (τ=0.40) was tuned on the 200-sample dev
  split. The routing design and the degeneration flag were written from the
  50-image golden set, and the flag limits were lowered once after two golden
  refusals passed both tests. Golden and dev numbers for arbitration are
  therefore in-sample; cite held-out numbers. No decision used held-out.
- **The headline result is sensitive to the flag limits.** With the original
  limits (40 characters, 3 non-Arabic letters) the held-out error rate is
  0.2057 instead of 0.1710. The two versions differ on 15 held-out outputs,
  11 of them one identical 40-character refusal. Exact match is 32.6% and
  32.7% respectively.
- The arbitration numbers are computed by replaying the rule over stored
  per-engine outputs; the combined engine was not run end to end on the
  full split.
- EasyOCR is run with its default detector on word crops, and 260 of its
  1,750 held-out outputs contain a spurious space. With spaces removed from
  both systems the comparison is 0.2102 against 0.1587.
- Latency numbers are single-machine (Apple M2 Pro, MPS); the held-out
  EasyOCR run shared the machine with another job. No batching or
  throughput optimization was attempted.
- The self-correction (CamelBERT reranking) ships disabled unless its dev
  fix/break balance is net-positive with CI excluding zero — see RESULTS
  for the measured outcome.
