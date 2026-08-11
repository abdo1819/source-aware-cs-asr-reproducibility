# Source-Aware Arabic–English Code-Switched ASR: Reproducibility Package

This is the minimal public companion for **“Source-Aware Evaluation of Synthetic Speech Augmentation and GRPO Output Contracts for Arabic–English Code-Switched ASR.”** It freezes the paper-facing configuration, transcript extraction and normalization rules, evaluation entry point, Gemini request protocol, and aggregate supplementary sensitivity results.

## Included

- `configs/paper_runs.yaml`: realized SFT and GRPO settings, selected checkpoints, and KL disclosure.
- `protocols/gemini_evaluation.json`: exact prompts, schema, request settings, and known endpoint limitations.
- `src/scoring.py`: transcript extraction, Arabic-aware normalization, English-token projection, corpus scoring, and paired utterance bootstrap.
- `scripts/score_predictions.py`: a runnable JSONL evaluator.
- `supplementary/manual_human_sensitivity_table.*`: the validated 347-utterance manually adjudicated human-only sensitivity table and its provenance record.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/score_predictions.py predictions.jsonl --output metrics.json
```

Each JSONL row must contain `reference` and either `prediction`, `transcript`, or `raw_output`. An optional `output_contract` can be `plain`, `structured_language_spans`, or `thinking`. Failed or empty outputs remain blank hypotheses and therefore contribute deletion errors.

To reproduce a paired nWER comparison against another prediction file:

```bash
python scripts/score_predictions.py system_a.jsonl \
  --compare system_b.jsonl \
  --bootstrap-replicates 100000 \
  --bootstrap-seed 42 \
  --output comparison.json
```

The files must contain the same rows in the same order and identical references.

## Scope and exclusions

This package is intentionally small. It does not redistribute third-party human audio, generated audio, model weights, LoRA checkpoints, commercial-API caches, or credentials. The manuscript identifies the external datasets and model checkpoints. The supplementary JSON records SHA-256 hashes of the private/local result inputs used to build the manual sensitivity table, enabling artifact identity checks by authorized holders.

The published comparisons are checkpoint-conditional. Principal training runs were not repeated across seeds, and the paired bootstrap does not estimate retraining variance. Commercial preview APIs may also change after the recorded request windows.

## License

Code in this package is released under the MIT License. Supplementary numerical outputs are supplied for research verification; third-party data and models retain their original licenses and terms.
