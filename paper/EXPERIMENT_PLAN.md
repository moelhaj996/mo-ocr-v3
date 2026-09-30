# Experiment plan

What would be needed to turn the report's narrow result into supported general
claims. None of this has been run, except the two-point threshold check noted
under C1. Estimates use the per-image costs measured on the Apple M2 Pro used
so far, taking the mean per image (EasyOCR about 0.06 to 0.1 s per word image
on CPU, Qwen2-VL-2B 0.7 s on the GPU; both from `results/paper_stats.json`).
The CAMeLBERT figure of about 0.07 s per scored word comes from the console
output of the calibration run (1,290 s for 18,508 scored words) and is not
stored in a result file. A different machine changes the wall-clock numbers,
not the design.

Protocol that applies to every experiment below:

- Choose every threshold and prompt on a development split. Score the test
  split once, after the configuration is frozen, and say so in the write-up.
- Report CER on raw and normalized text, exact match, and paired bootstrap
  intervals on differences, exactly as the current harness does.
- Record per-sample outputs and timings in run files under `results/` so the
  numbers can be rescored later.
- Keep the disclosure that the current images are not confirmed APTI data
  and remain unlicensed for redistribution.

## Priority 1: data with traceable provenance (blocks any general claim)

**A1. Repeat the evaluation on a dataset with a documented license.**
Candidates to check, in order: APTI and KHATT (both obtainable through a
registration form; confirm the licence terms permit publication of derived
results and, ideally, of the per-image outputs), then any word- or line-level
Arabic OCR dataset on the Hugging Face Hub whose card states a licence.
Verify the licence text before downloading anything. Keep the same three-way
split design (pilot / development / test) with a fixed seed and a hashed
manifest.
Cost: about 25 minutes of Qwen2-VL time and 3 minutes of EasyOCR time per
2,000 word images at the measured means; a day of human time for access,
licence checking and manifest building.

**A2. Add an independent second dataset to test generalization.**
It should differ from the first in renderer or source: real scans or camera
images rather than clean synthetic renderings, and ideally text lines rather
than isolated words. Report the two datasets separately; do not pool them.
Cost: as A1 per 2,000 images, plus line-level ground truth if lines are used.
Line images will need the page-mode path, which is currently unevaluated, so
budget time for a line-segmentation check first.

## Priority 2: fairer baselines

**B1. Prompt and decoding sensitivity for Qwen2-VL.**
The 22% degenerate-output rate was measured with one Arabic prompt and greedy
decoding. Try at least: an English instruction, a minimal prompt of the form
"OCR:", a prompt that forbids explanations explicitly, beam search with 3 to 5
beams, and a lower token limit. Measure the degenerate rate and the routed
result for each on the development split; pick one configuration; score it
once on the test split.
Cost: about 6 Qwen2-VL runs on 200 development images (about 2.5 minutes
each at the measured mean) plus one test run (about 20 minutes for 1,750
images).

**B2. Another dedicated OCR baseline under the same harness.**
Tesseract 5 with the Arabic model and PaddleOCR's Arabic recognizer are both
freely licensed and run on CPU. Add each as a `Recognizer` in
`src/moocr/models/`, score it alone on the same splits, and test it as the
fallback engine in the routing rule. This shows whether the gain depends on
EasyOCR specifically.
Cost: an hour of engineering per engine; a few minutes of compute per split.

**B3. Pin the Qwen2-VL checkpoint revision** in `config.py` before any rerun,
so later results are comparable.

## Priority 3: robustness

**C1. Threshold sensitivity on the test split.** Sweep the confidence threshold
and both flag limits on the test split after the policy is frozen, and report
the whole curve as a post hoc analysis, so readers can see how sharp the
optimum is. Partly done: the report has the held-out result under both flag
versions at threshold 0.40, and a two-point check of the threshold (0.35 and
0.45, both within 0.005 of the reported figure). A full curve and the flag
limits remain.
Cost: none beyond A1; it is a replay over stored outputs.

**C2. Image quality.** Degrade the images in controlled steps (downscaling,
blur, additive noise, JPEG compression) and rerun both engines and the router
at each step. The question is whether the routing gain survives when both
engines get worse.
Cost: one Qwen2-VL run per degradation level per split; four levels on 200
development images is about 10 minutes.

**C3. Word length and mixed-script content.** The current data has no digits,
no Latin letters and no words shorter than 7 characters, so the degeneration
flag's non-Arabic-letter test has never been exercised on legitimate input. A
dataset with digits, dates and embedded Latin words is required; the flag
will need a different rule for such data.
Cost: depends entirely on finding data; the flag redesign is a day of work.

## Priority 4: runtime

**D1. Benchmark the combined engine live.** Run `get_engine("arbitration")`
end to end on the development split on a quiet machine, three repetitions,
warm caches, and record per-image wall-clock time including fallback
execution. Report medians and totals next to the per-engine figures, and
replace the reconstructed 7.8x estimate with the measured value.
Cost: about 10 minutes of compute; no new data.

**D2. Throughput.** Batch Qwen2-VL inference and report images per second
for both the language model and the router. Only after D1.

## Priority 5: the negative results

**E1. TrOCR.** The ten-image screen should be replaced by a run on the full
development split for any checkpoint that loads, and by a fine-tuning
attempt on in-domain data before any conclusion about TrOCR is drawn.
Cost: minutes for the screen; fine-tuning needs a labelled training set and
several GPU hours.

**E2. Corrector with context.** The reranking experiment used isolated words.
Repeat it on text lines, where the language model has context, before
concluding that masked-language-model reranking does not help Arabic OCR.
Cost: needs line-level data (A2); about 0.07 s per candidate scored.
