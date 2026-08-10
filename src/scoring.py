"""Exact paper-facing transcript extraction, normalization, and scoring."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import jiwer
import numpy as np


_ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_ARABIC_ALEF_VARIANTS = re.compile(r"[إأآا]")
_OPTIONAL_FILLER_TOKENS = re.compile(r"\b(?:ا?م{2,}|اه+|او+ف+)\b")
_SPLIT_WAW_PREFIX = re.compile(r"(?<!ال )\bو\s+([A-Za-z\u0621-\u064A])")
_SPLIT_AL_ALREADY_DEF = re.compile(r"\bال\s+(ال[\u0621-\u064A]+)\b")
_SPLIT_AL_PREFIX = re.compile(r"\bال\s+([A-Za-z][A-Za-z0-9_-]*|[\u0621-\u064A]{2,})\b")
_SPLIT_MA_NEGATION = re.compile(r"\bما\s+([\u0621-\u064A]{2,}ش)\b")
_SPLIT_VERB_LANA = re.compile(r"\b([\u0621-\u064A]{3,})\s+لنا\b")
_SPLIT_AN_HUWA = re.compile(r"\bان\s+هو\b")
_SPLIT_WA_LA = re.compile(r"\bو\s+لا\b")
_OPTIONAL_WAW_TOKENS = re.compile(r"\bوبعد\b")
_EGYPTIAN_DEICTIC_DA = re.compile(r"\bدا\b")
_ARABIC_LATIN_BOUNDARY = re.compile(
    r"([\u0621-\u064A])([A-Za-z])|([A-Za-z])([\u0621-\u064A])"
)
_LATIN_TOKEN = re.compile(r"[a-z]")
_ARABIC_TOKEN = re.compile(r"[\u0600-\u06FF]")


def extract_transcription(raw: str, output_contract: str = "plain") -> str:
    """Extract the final transcript using the manuscript's contract rules."""
    candidate = (raw or "").strip()
    if candidate.startswith("{"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("transcript"), str):
            return parsed["transcript"].strip()
        match = re.search(r'"transcript"\s*:\s*"((?:[^"\\]|\\.)*)', candidate, re.DOTALL)
        if match:
            try:
                return json.loads(f'"{match.group(1)}"').strip()
            except json.JSONDecodeError:
                return match.group(1).strip()
    if "<asr_text>" in candidate:
        transcript = candidate.split("<asr_text>", 1)[1]
        if "</asr_text>" in transcript:
            transcript = transcript.split("</asr_text>", 1)[0]
        return transcript.strip()
    if output_contract in {"thinking", "structured_language_spans"}:
        return ""
    if re.match(r"^language\s+", candidate, re.IGNORECASE) or "<thinking>" in candidate:
        return ""
    return candidate


def _normalize_wer_variants(text: str) -> str:
    text = _OPTIONAL_FILLER_TOKENS.sub(" ", text)
    text = _EGYPTIAN_DEICTIC_DA.sub("ده", text)
    text = _SPLIT_AN_HUWA.sub("انه", text)
    text = _SPLIT_MA_NEGATION.sub(r"ما\1", text)
    text = _SPLIT_VERB_LANA.sub(r"\1لنا", text)
    text = _SPLIT_WA_LA.sub("ولا", text)
    text = _SPLIT_WAW_PREFIX.sub(r"و\1", text)
    text = _SPLIT_AL_ALREADY_DEF.sub(r"\1", text)
    text = _SPLIT_AL_PREFIX.sub(r"ال\1", text)
    return _OPTIONAL_WAW_TOKENS.sub("بعد", text)


def _separate_mixed_script_tokens(text: str) -> str:
    while True:
        updated = _ARABIC_LATIN_BOUNDARY.sub(r"\1\3 \2\4", text)
        if updated == text:
            return text
        text = updated


def normalize_for_alignment(text: str) -> str:
    """Apply the exact Arabic-aware normalizer used by the reported metrics."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("ـ", "")
    text = _ARABIC_ALEF_VARIANTS.sub("ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي").lower()
    text = _normalize_wer_variants(text)
    text = _separate_mixed_script_tokens(text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def extract_english_tokens(text: str) -> str:
    normalized = normalize_for_alignment(text)
    return " ".join(token for token in normalized.split() if _LATIN_TOKEN.search(token))


def token_language(token: str) -> str:
    if _LATIN_TOKEN.search(token):
        return "English"
    if _ARABIC_TOKEN.search(token):
        return "Arabic"
    return "Other"


def code_switch_diagnostics(text: str) -> dict[str, float | int]:
    languages = [token_language(token) for token in normalize_for_alignment(text).split()]
    languages = [language for language in languages if language != "Other"]
    arabic = languages.count("Arabic")
    english = languages.count("English")
    retained = arabic + english
    cmi = 0.0 if retained == 0 else 100.0 * (1.0 - max(arabic, english) / retained)
    switches = sum(a != b for a, b in zip(languages, languages[1:]))
    return {"arabic_tokens": arabic, "english_tokens": english, "cmi": cmi, "switches": switches}


@dataclass(frozen=True)
class EditCounts:
    errors: int
    reference_units: int


def _word_counts(reference: str, hypothesis: str) -> EditCounts:
    output = jiwer.process_words(reference, hypothesis)
    return EditCounts(output.substitutions + output.deletions + output.insertions, output.hits + output.substitutions + output.deletions)


def _char_counts(reference: str, hypothesis: str) -> EditCounts:
    output = jiwer.process_characters(reference, hypothesis)
    return EditCounts(output.substitutions + output.deletions + output.insertions, output.hits + output.substitutions + output.deletions)


def corpus_metrics(references: Iterable[str], hypotheses: Iterable[str]) -> dict[str, float | int | None]:
    references = list(references)
    hypotheses = list(hypotheses)
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and have equal length")
    normalized_references = [normalize_for_alignment(value) for value in references]
    normalized_hypotheses = [normalize_for_alignment(value) for value in hypotheses]
    if any(not value for value in normalized_references):
        raise ValueError("an empty normalized reference is not scoreable")
    english_pairs = [
        (extract_english_tokens(reference), extract_english_tokens(hypothesis))
        for reference, hypothesis in zip(references, hypotheses)
    ]
    english_pairs = [pair for pair in english_pairs if pair[0]]
    return {
        "n": len(references),
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
        "nwer": jiwer.wer(normalized_references, normalized_hypotheses),
        "ncer": jiwer.cer(normalized_references, normalized_hypotheses),
        "english_wer": None if not english_pairs else jiwer.wer([p[0] for p in english_pairs], [p[1] for p in english_pairs]),
        "n_english": len(english_pairs),
        "empty_rate": sum(not value for value in normalized_hypotheses) / len(hypotheses),
    }


def paired_nwer_bootstrap(
    references: list[str],
    hypotheses_a: list[str],
    hypotheses_b: list[str],
    *,
    replicates: int = 100_000,
    seed: int = 42,
    chunk_size: int = 500,
) -> dict[str, float | int]:
    """Paired utterance bootstrap for delta nWER = A - B."""
    if not (len(references) == len(hypotheses_a) == len(hypotheses_b)) or not references:
        raise ValueError("paired inputs must be non-empty and have equal length")
    normalized_references = [normalize_for_alignment(value) for value in references]
    normalized_a = [normalize_for_alignment(value) for value in hypotheses_a]
    normalized_b = [normalize_for_alignment(value) for value in hypotheses_b]
    if any(not value for value in normalized_references):
        raise ValueError("an empty normalized reference is not scoreable")
    counts_a = [_word_counts(r, h) for r, h in zip(normalized_references, normalized_a)]
    counts_b = [_word_counts(r, h) for r, h in zip(normalized_references, normalized_b)]
    errors_a = np.asarray([item.errors for item in counts_a], dtype=np.float64)
    errors_b = np.asarray([item.errors for item in counts_b], dtype=np.float64)
    denominators = np.asarray([item.reference_units for item in counts_a], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, len(references), size=(stop - start, len(references)))
        denominator = denominators[indices].sum(axis=1)
        deltas[start:stop] = errors_a[indices].sum(axis=1) / denominator - errors_b[indices].sum(axis=1) / denominator
    tail = min(np.count_nonzero(deltas <= 0), np.count_nonzero(deltas >= 0))
    p_value = min(1.0, 2.0 * (tail + 1) / (replicates + 1))
    return {
        "replicates": replicates,
        "seed": seed,
        "delta_nwer_a_minus_b": float(errors_a.sum() / denominators.sum() - errors_b.sum() / denominators.sum()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "two_sided_p_value": float(p_value),
    }
