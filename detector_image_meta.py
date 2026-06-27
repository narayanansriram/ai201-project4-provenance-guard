import re
import math


def _words(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def _sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


def sentence_length_variance_score(text: str) -> float:
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.5
    lengths = [len(_words(s)) for s in sents]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    score = max(0.0, 1.0 - (std / 8.0))  # tighter scale for short captions
    return round(score, 4)


def type_token_ratio_score(text: str) -> float:
    words = _words(text)
    if len(words) < 5:
        return 0.5
    ttr = len(set(words)) / len(words)
    score = max(0.0, min(1.0, (0.9 - ttr) / 0.5))
    return round(score, 4)


def svo_pattern_score(text: str) -> float:
    """
    AI image descriptions overuse 'A [noun] [verb]-ing in/on/with/at a [noun]' patterns.
    Count matches and normalize.
    """
    pattern = re.compile(
        r'\b(a|an|the)\s+\w+\s+\w+ing\s+(in|on|with|at|near|under|over|beside|through)\b',
        re.IGNORECASE
    )
    matches = len(pattern.findall(text))
    words = _words(text)
    if not words:
        return 0.0
    # Normalize: 1 match per 10 words → score 1.0
    density = matches / max(len(words), 1) * 10
    return round(min(1.0, density), 4)


def detect_image_description(text: str) -> dict:
    """
    Modified stylometric signal for image descriptions.
    Replaces punctuation_density with svo_pattern_score.
    Weights: sentence_variance 0.40, ttr 0.40, svo 0.20
    """
    v = sentence_length_variance_score(text)
    t = type_token_ratio_score(text)
    s = svo_pattern_score(text)
    combined = round((0.40 * v) + (0.40 * t) + (0.20 * s), 4)
    return {
        "score": combined,
        "variance_score": v,
        "ttr_score": t,
        "svo_score": s,
    }
