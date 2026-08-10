#!/usr/bin/env python3
"""Score paper-format prediction JSONL and optionally compare two systems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.scoring import corpus_metrics, extract_transcription, paired_nwer_bootstrap


def load_rows(path: Path) -> tuple[list[str], list[str]]:
    references: list[str] = []
    hypotheses: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            reference = row.get("reference", row.get("text"))
            if not isinstance(reference, str):
                raise ValueError(f"{path}:{line_number}: missing string reference/text")
            if isinstance(row.get("prediction"), str):
                hypothesis = row["prediction"]
            elif isinstance(row.get("transcript"), str):
                hypothesis = row["transcript"]
            else:
                hypothesis = extract_transcription(
                    row.get("raw_output", ""), row.get("output_contract", "plain")
                )
            references.append(reference)
            hypotheses.append(hypothesis)
    if not references:
        raise ValueError(f"{path}: no rows")
    return references, hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    references, hypotheses = load_rows(args.predictions)
    payload: dict[str, object] = {
        "system_a": str(args.predictions),
        "metrics_a": corpus_metrics(references, hypotheses),
    }
    if args.compare:
        compare_references, compare_hypotheses = load_rows(args.compare)
        if compare_references != references:
            raise ValueError("comparison references or row order differ")
        payload.update(
            {
                "system_b": str(args.compare),
                "metrics_b": corpus_metrics(compare_references, compare_hypotheses),
                "paired_nwer_bootstrap": paired_nwer_bootstrap(
                    references,
                    hypotheses,
                    compare_hypotheses,
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed,
                ),
            }
        )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
