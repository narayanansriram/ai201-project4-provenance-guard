import re
import math


def _sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


def _words(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def sentence_length_variance_score(text: str) -> float:
    """High variance → human-like → low score. Returns 0.0–1.0 (AI likelihood)."""
    sents = _sentences(text)
    if len(sents) < 3:
        return 0.5  # not enough data
    lengths = [len(_words(s)) for s in sents]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    # std dev of 0 → score 1.0 (perfectly uniform = AI)
    # std dev of 15+ → score 0.0 (very variable = human)
    score = max(0.0, 1.0 - (std / 15.0))
    return round(score, 4)


def type_token_ratio_score(text: str) -> float:
    """High TTR → human-like → low score. Returns 0.0–1.0 (AI likelihood)."""
    words = _words(text)
    if len(words) < 10:
        return 0.5
    # Use 100-word sliding window for longer texts
    if len(words) > 300:
        window = words[:100]
    else:
        window = words
    ttr = len(set(window)) / len(window)
    # TTR ranges ~0.4 (repetitive/AI) to ~0.9 (varied/human)
    # Map to AI score: low TTR → high score
    score = max(0.0, min(1.0, (0.9 - ttr) / 0.5))
    return round(score, 4)


def punctuation_density_score(text: str) -> float:
    """High comma/semicolon/em-dash density → AI-like → high score. Returns 0.0–1.0."""
    words = _words(text)
    if not words:
        return 0.5
    commas = text.count(",")
    semicolons = text.count(";")
    em_dashes = text.count("—") + text.count("--")
    density = (commas + semicolons + em_dashes) / len(words) * 100
    # Human baseline ~5–8 per 100 words; AI tends 10+
    score = max(0.0, min(1.0, (density - 5.0) / 10.0))
    return round(score, 4)


def detect_with_stylometrics(text: str) -> dict:
    """Returns {"score": float, "variance_score": float, "ttr_score": float, "punct_score": float}."""
    v = sentence_length_variance_score(text)
    t = type_token_ratio_score(text)
    p = punctuation_density_score(text)
    # Weights from planning.md: 0.40 / 0.40 / 0.20
    combined = round((0.40 * v) + (0.40 * t) + (0.20 * p), 4)
    return {
        "score": combined,
        "variance_score": v,
        "ttr_score": t,
        "punct_score": p,
    }
