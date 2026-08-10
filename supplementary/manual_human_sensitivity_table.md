# Manually Adjudicated Human-Test Sensitivity Analysis

This table includes the **347** rows in the frozen 862-utterance human test split whose review status is exactly `keep`. The 515 `gemini_keep` rows are excluded, removing direct Gemini-based acceptance from this sensitivity subset.

## Minimal interpretation

The lowest nWER remains Gemini Pro direct/thinking-on (19.57%), followed by Gemini Flash structured/thinking-off (31.85%) and Plain GRPO (40.05%). Their corresponding full human-source nWER values are 12.23%, 13.30%, and 32.31%, respectively. Thus the manually retained subset is harder, especially for Flash, but the main ordering is not reversed. This is a selection-bias sensitivity check, not a deployment-equivalence or causal comparison.

| Rank | System | nWER (%) | nCER (%) | En-WER (%) | Coverage | Empty (%) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Gemini Pro (direct, thinking on) | 19.57 | 10.35 | 25.50 | 347/347 | 0.29 |
| 2 | Gemini Pro (structured, thinking on) | 27.95 | 17.61 | 55.57 | 346/347 | 0.29 |
| 3 | Gemini Flash (structured, thinking off) | 31.85 | 19.88 | 51.81 | 347/347 | 0.00 |
| 4 | Gemini Flash (direct, thinking on) | 33.05 | 22.40 | 45.23 | 347/347 | 0.29 |
| 5 | AraZN Whisper Small | 38.24 | 18.29 | 38.66 | 347/347 | 0.00 |
| 6 | Plain GRPO | 40.05 | 20.52 | 40.54 | 347/347 | 0.00 |
| 7 | AraZN Whisper Medium | 41.75 | 21.87 | 40.94 | 347/347 | 0.00 |
| 8 | Human+synthetic SFT | 46.18 | 28.75 | 51.81 | 347/347 | 0.00 |
| 9 | Human-only SFT | 46.70 | 27.82 | 45.10 | 347/347 | 0.00 |
| 10 | Structured GRPO | 48.90 | 32.54 | 55.70 | 347/347 | 2.88 |
| 11 | Base 4-bit Qwen3-ASR | 54.22 | 33.60 | 85.50 | 347/347 | 0.00 |
| 12 | Thinking-only GRPO | 57.52 | 37.65 | 85.37 | 347/347 | 2.59 |
| 13 | Whisper large-v3 | 66.89 | 46.50 | 109.13 | 347/347 | 0.00 |
| 14 | Gemini Flash (direct, thinking off) | 175.64 | 157.04 | 68.99 | 347/347 | 0.29 |

## Inclusion and scoring

- Inclusion: membership in `human_test_v1.jsonl` and an exact manual review status of `keep`.
- Exclusion: `gemini_keep` and every non-human-test row.
- Join: the final two components of each audio path, which is stable across saved workspace prefixes.
- Metrics: corpus-level nWER and nCER after `asr_utils.normalize_for_alignment`; En-WER after the same normalization and Latin-token projection.
- Empty or failed outputs: retained as blank hypotheses and counted as deletion errors.
- Inference: none; all hypotheses come from the existing JSON results or committed Gemini SQLite caches.
- Validation: rescoring all 862 human-test rows reproduces the manuscript nWER and En-WER for all 14 included systems at two decimal places.

## Availability notes

- Gemini Flash structured/thinking-on is not included because the manuscript omits its incomplete stored sweep.
- Gemini Pro zero-thinking conditions are unavailable because the endpoint rejected the requested zero thinking budget.
- Gemini Pro direct/thinking-on combines the original cache with successful direct retries.
- Gemini Pro structured/thinking-on combines its retry cache with successful 16,384-token retries; remaining failures stay blank.

## Exact stored inputs

- `qwen3asr_data/canonical_v1/human_test_v1.jsonl`
- `data_review/train_full/review_state.json`
- `analysis/arazn_whisper_medium_test_v1_full.json`
- `analysis/arazn_whisper_small_test_v1_full.json`
- `analysis/gemini_canonical_eval/gemini-3-flash-preview_test_v1.json`
- `analysis/gemini_canonical_eval/gemini-3.1-pro-preview_test_v1_retry_summary.json`
- `analysis/whisper_large_v3_test_v1_full.json`
- `experiments/06_grpo_from_sft700_norm_control/eval_test/eval_00_base_4bit_test_v1.json`
- `experiments/07_grpo_from_norm50_reward_fix/eval/eval_final_test_v1_post250.json`
- `experiments/10_grpo_structured_reduced_from_s08_ckpt900/eval/eval_10_grpo_structured_reduced_checkpoint-200_checkpoint-200.json`
- `experiments/24_grpo_thinking_joint_v4/eval/eval_24_grpo_thinking_joint_v4_checkpoint-190.json`
- `experiments/s02_sft_canonical_v1/eval/eval_s02_sft_canonical_v1_test_v1.json`
- `experiments/s05_sft_canonical_v1_human_only/eval/eval_test_v1.json`
- `generative_model_capability/cache/gemini_batch_cache.sqlite3`
- `generative_model_capability/cache/gemini_31_pro_direct_thinking_retry.sqlite3`
- `generative_model_capability/cache/gemini_31_pro_structured_thinking_retry.sqlite3`
- `generative_model_capability/cache/gemini_31_pro_structured_thinking_retry_5_high_tokens.sqlite3`

The CSV contains model-family and exact source-artifact columns. The JSON records input hashes, selection metadata, diagnostics, and unrounded values.
