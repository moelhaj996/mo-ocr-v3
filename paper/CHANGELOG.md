# Paper change log

Substantive corrections to `paper.tex`, most recent first. Each entry says
what changed and what evidence supports it.

## 2026-09-30, review round 4

1. **WER explanation corrected again (Section 4.3).** Round 3 had replaced the
   wrong claim with another wrong one: that with single-token references WER
   equals the exact-mismatch rate plus an insertion term. A hypothesis that
   contains the reference word plus an extra word costs one insertion under
   the standard alignment but is an exact mismatch, so that formula double
   counts. The stored held-out predictions contain six such Qwen2-VL outputs
   and two arbitration outputs. The paper now states the standard definition
   (word substitutions plus deletions plus insertions over reference words),
   notes that insertions can push it above 1, and says it is not generally
   equivalent to the exact-mismatch rate. The same correction was made in
   README.md, METHOD.md, LIMITATIONS.md, the RESULTS.md generator, the
   metrics module docstring and the note the harness writes into result
   files. No WER or CER value changed: held-out WER is still 1.01, 3.80 and
   0.77 (1,775, 6,645 and 1,345 word edits over 1,750 reference words).
2. **Statistics script.** `scripts/paper_stats.py` now uses a new library
   function, `moocr.metrics.token_edit_breakdown`, which is exact only for
   single-token references and asserts that. Its output field
   `wrong_first_token` (a misnomer, since the alignment can match the
   reference anywhere in the hypothesis) is renamed
   `substitutions_and_deletions`, and the assumption is documented in the
   `definition` string of `results/paper_stats.json`. A new field counts the
   hypotheses that hold the reference token plus extra tokens (0, 6, 2).
3. **Regression tests.** Twelve tests added covering exact match, the
   reference word preceded or followed by an extra word, a substitution and
   an empty hypothesis, checked against the standard formula and against
   JiWER's word-level counts.

## 2026-09-30, review round 3

1. **WER justification corrected (Section 4.3).** The old text said single-word
   references make WER equivalent to exact match. That is wrong: hypotheses
   can contain several tokens, so insertions count and WER can exceed 1. The
   paper now presents CER and exact match as a reporting choice and gives the
   held-out WER for the record: 1.01 EasyOCR, 3.80 Qwen2-VL, 0.77 arbitration
   (`scripts/paper_stats.py`, block `wer_heldout`). The same wording was fixed
   in README.md, METHOD.md, LIMITATIONS.md, the RESULTS.md generator and the
   note the harness writes into result files. The committed run files were
   produced before that note changed, so their `scores.wer` text is
   superseded by `results/paper_stats.json`. The WER values are on normalized
   text; raw values are 1.05, 3.82 and 0.80. Insertions are most of Qwen2-VL's
   token edits and a small share of the other two systems'.
2. **Latency figure labelled as an estimate (Section 5.6).** The mean Qwen2-VL
   latency range now reads 698 to 780 ms (was rounded to 700), and the slow
   degenerate outputs are explained by their length (median 87 characters,
   1.37 s) rather than by the token limit, which the stored outputs do not
   support. The 7.8x cost of
   arbitration is reconstructed from per-engine times recorded while each
   engine ran alone (Qwen2-VL on every image plus EasyOCR on the 851 routed
   images, over EasyOCR on every image). The paper no longer reads as if the
   combined engine had been timed end to end, and cross-references the replay
   statement in Section 4.3.
3. **Normalization claims qualified (Sections 4.2 and 7).** Removed the categorical
   statement that all foldings would be wrong for dialectal text. The text now
   says normalization is task dependent and deliberately forgiving, that some
   merged forms distinguish real words in MSA too, and that this is why raw
   and normalized rates are both reported. The Limitations section now says
   the profile was chosen for MSA and not validated on dialectal text.
   LIMITATIONS.md matches.
4. **Terminology.** "Classical OCR engine" replaced by "dedicated OCR engine"
   in the title, introduction and Section 3.1, with a sentence noting that
   EasyOCR's detector and recognizer are neural networks.
5. **TrOCR screen conclusions limited (Section 6.1).** Removed the prediction
   that a larger sample would not change the decision. The screen is now
   described as a compatibility check on ten images that ranks nothing and
   supports no general claim about Arabic TrOCR models.
6. **Contribution sharpened (Section 1).** The contributions are stated as:
   complementary error patterns of two recognizers on this dataset, a simple
   routing policy evaluated on it, component ablations with threshold
   sensitivity, and transparent negative results. The text says routing
   between recognizers is not new and makes no claim about Arabic documents
   in general.

7. **Threshold sensitivity added to Table 3.** Post hoc held-out results at
   thresholds 0.35 and 0.45 (0.1760 and 0.1735) now appear beside the reported
   0.40 row, from `results/paper_stats.json`.

Unchanged and re-verified: the held-out result (0.2294 to 0.1710, interval
−0.0664 to −0.0505), the sensitivity disclosure (0.2057 with the original flag,
15 outputs differ, 11 of them one refusal string), all tables and figures.
`results/paper_stats.json` was regenerated and is identical to the committed
version apart from the new `wer_heldout` block.

## 2026-09-30, review round 2

- Flag-only development figures recomputed with the current flag limits
  (0.1992, interval −0.0654 to −0.0007); they had been computed with the
  earlier limits.
- Held-out ablation added (Table 3), including the result under the original
  flag limits and the 15-output sensitivity.
- Tuning history corrected: the routing design and both flag versions came
  from the 50-image pilot set; only the confidence threshold was tuned on the
  development split.
- Corrector candidate generator fixed (letters in several confusion families
  were swapped within one only); experiment rerun (Section 6.2).
- TrOCR screen rerun through the harness; the fourth checkpoint raised an
  error on every image and had been recorded as empty output.
- Data provenance corrected: the images are the first 2,000 rows of
  `mssqpi/Arabic-OCR-Dataset`; the APTI label was a folder name and is
  unsupported; the source is no longer publicly accessible.
- Degenerate-output categories recounted with disjoint rules; dot-confusion
  share, spurious-space effect and per-split confidence correlations added.
- All bibliography entries verified against publisher or index pages.
