"""Recompute a result file's aggregates from its stored per-sample predictions.

Used when scoring rules change (normalization version bump): predictions are
unchanged, so no model rerun is needed — only the derived numbers move. The
file's meta.norm_version is updated and the previous aggregates are kept
under 'superseded'.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moocr.harness.evaluate import _score_run  # noqa: E402
from moocr.config import Config  # noqa: E402


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg)
        data = json.loads(path.read_text(encoding="utf-8"))
        old = {
            "norm_version": data["meta"]["norm_version"],
            "scores": data["scores"],
        }
        rescored = _score_run(
            data["meta"]["engine"],
            data["meta"]["split"],
            data["meta"]["manifest"],
            data["per_sample"],
            data["failures"],
            Config(),
        )
        rescored["meta"] = {**data["meta"], "norm_version": rescored["meta"]["norm_version"],
                            "rescored_from": old["norm_version"]}
        rescored["superseded"] = old
        path.write_text(json.dumps(rescored, ensure_ascii=False, indent=2), encoding="utf-8")
        new_cer = rescored["scores"]["normalized"]["corpus_cer"]
        old_cer = old["scores"]["normalized"]["corpus_cer"]
        print(f"{path.name}: CER(norm) {old_cer:.4f} -> {new_cer:.4f} (norm {old['norm_version']} -> {rescored['meta']['norm_version']})")


if __name__ == "__main__":
    main()
