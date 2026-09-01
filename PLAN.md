# MO-OCR v3 — Architecture & Milestone Plan

Fresh implementation (no code carried over from mo-ocr-v2). Target: an Arabic
document-intelligence pipeline whose every reported number is reproducible,
built to the original four-model specification, evaluated under the
arabic-nlp-evaluation protocol.

## 1. Measured constraints (not assumptions)

| Constraint | Value | How known |
|---|---|---|
| Hardware | Apple M2 Pro, 16 GB unified, MPS (no CUDA) | `sysctl` |
| Qwen-VL | Qwen2-VL-2B-Instruct **fully cached** locally | HF cache inspection |
| TrOCR | No official Arabic checkpoint exists; community checkpoint must be probed live and pinned | HF cache + hub |
| CamelBERT | not cached — ~440 MB download | HF cache |
| LayoutLMv3 | not cached — ~500 MB download | HF cache |
| EasyOCR (ar) | weights cached; kept only as the *continuity baseline* engine | `~/.EasyOCR` |
| Data in hand | APTI: 2,000 printed Arabic word images + ground truth (from v2 data dir, copied — not the code) | disk |
| KHATT | **absent** (registration-gated). Handwritten slice is out of scope until data exists | disk |
| Field-extraction labels | **none exist.** LayoutLMv3 is integrated and smoke-tested, but no field-F1 will be claimed | disk |

## 2. Architecture

```
                          ┌────────────────────────────────────────────┐
 image ──► Preprocess ──► │ Recognizers (pluggable, common interface)  │
          (deskew, DPI,   │  • TrOCR-Arabic  (primary, seq-logprob)    │
           grayscale,     │  • Qwen2-VL-2B   (VLM reader, MPS)         │
           pad crops)     │  • EasyOCR-ar    (continuity baseline)     │
                          └───────────────┬────────────────────────────┘
                                          ▼
                     Arbitration (confidence-routed; signal VALIDATED on dev
                     split before use — corr(confidence, correctness) reported)
                                          ▼
                     CamelBERT post-correction — GATED (low-confidence only),
                     edit-distance-capped, reranking not rewriting;
                     fix/break counts reported every run
                                          ▼
                     LayoutLMv3 structuring (interface + smoke test only —
                     no labeled data, no claims)
                                          ▼
                     Output (OUTPUT normalization profile — diacritics kept)

 Everything above is scored by a model-agnostic harness that never imports
 any model module.
```

## 3. Requirement → component map

| Original requirement | Component | Milestone |
|---|---|---|
| Data pipeline | `moocr/data/` — manifest with per-file SHA-256, deterministic splits: golden 50 / dev 200 / held-out 1,750 | M1 |
| Versioned normalization | `moocr/normalization.py` — `NORM_VERSION`, separate SCORING vs OUTPUT profiles, every rule a flag | M1 |
| Dual raw/normalized CER+WER | `moocr/metrics.py` — corpus-level primary, macro secondary; WER suppressed on single-word sets (meaningless, per Phase A) | M1 |
| Bidi-direction check | `metrics.bidi_check` — run before any model-blame conclusion | M1 |
| Paired bootstrap CI on deltas | `harness/compare.py` — paired resampling, CI + sign-test p, seeded | M1 |
| Transparent failure accounting | failures scored 1.0 *and* counted separately; both numbers in every report | M1 |
| Held-out + golden splits | split assignment committed as manifest; golden runs as regression test | M1 |
| TrOCR | `models/trocr.py` — checkpoint chosen by live probe, pinned by revision hash in config | M2 |
| Qwen-VL | `models/qwen_vl.py` — Qwen2-VL-2B-Instruct, MPS, constrained transcription prompt | M2 |
| Baseline results per component | `results/baseline_<engine>_<split>.json` for each recognizer alone | M2 |
| Ablation error budget | `harness/error_budget.py` — oracle-substitution deltas summing to 100% | M3 |
| Fix/break for post-correction | logged per run by `compare.py`; corrector ships OFF until net-positive on dev | M3–M4 |
| Self-correction mechanism | `models/camelbert.py` + `fusion.py` — gated, capped, reranking | M4 |
| Per-slice reporting | slices: word length (≤4 / 5–8 / ≥9), digit-bearing, engine-agreement | M1, used from M2 |
| LayoutLMv3 | `models/layoutlmv3.py` — inference wrapper + synthetic smoke test | M5 |
| Preprocessing | `moocr/preprocess.py` — each op independently toggleable, measured one at a time | M4 |
| Tests per component | `tests/` — pure-python with fake recognizers; no downloads needed for CI | every milestone |
| Reproducibility | `scripts/reproduce.sh` regenerates every number; seeds fixed; deps pinned | M6 |
| RESULTS/METHOD/LIMITATIONS | written from measured artifacts only | M6 |

## 4. Milestones (each ends with tests green + a committed number or an explicit "no claim")

- **M0** Scaffold: package, pydantic config, logging w/ correlation ids, pinned deps.
- **M1** Harness before models: normalization v1, metrics, splits, compare, fake-engine end-to-end test. *Gate: unit tests green; harness validated against synthetic cases with known CER.*
- **M2** Recognizers + baselines: EasyOCR / TrOCR / Qwen2-VL each scored alone on golden+dev. *Gate: `baseline.json` committed.*
- **M3** Error budget: ablations, confidence-signal validation, engine agreement analysis. *Gate: budget table sums to 100%.*
- **M4** Accuracy work in budget order: preprocessing, arbitration, gated CamelBERT. One variable per change; keep-or-revert by paired CI on dev; final confirmation on held-out once.
- **M5** LayoutLMv3 integration (smoke only, stated limitation).
- **M6** Engineering hardening + write-up artifacts + reproduction script.

## 5. Standing assumptions (each flagged, none silent)

1. "Qwen-VL" is satisfied by **Qwen2-VL-2B-Instruct** (the locally runnable variant; the original Qwen-VL-Chat is deprecated upstream).
2. "TrOCR" is satisfied by the best **community Arabic TrOCR** checkpoint that loads and beats EasyOCR on the dev split; identity pinned in config after the M2 probe. If none beats EasyOCR, that negative result is reported and EasyOCR remains primary.
3. CamelBERT variant: `bert-base-arabic-camelbert-mix` (APTI is MSA; `-mix` is safe for MSA and robust if dialectal data arrives; the dialectal-normalization caveat of the protocol is respected by keeping normalization MSA-only and saying so).
4. Latency budget: none was given; latency is *reported* per stage, not optimized against.
5. Single-word printed data means: WER not reported, layout/segmentation stages exist but their error-budget share on this set is ~0 by construction — stated in LIMITATIONS, not hidden.
