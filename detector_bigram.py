import re


def _words(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def detect_with_bigrams(text: str) -> dict:
    """
    Measures bigram (word-pair) repetition rate.
    AI writing reuses collocations from its training distribution;
    human writing uses more varied collocations.

    repetition_rate = (total_bigrams - unique_bigrams) / total_bigrams
    Mapped linearly: 0.0 → score 0.0, 0.30+ → score 1.0 (clamped).

    Returns {"score": float, "repetition_rate": float, "total_bigrams": int, "unique_bigrams": int}
    """
    words = _words(text)
    if len(words) < 4:
        return {"score": 0.5, "repetition_rate": 0.0, "total_bigrams": 0, "unique_bigrams": 0}

    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    total = len(bigrams)
    unique = len(set(bigrams))
    repetition_rate = (total - unique) / total

    # Linear scale: 0.0 → 0.0, 0.08 → 1.0 (clamped).
    # 0.30 was too high for single paragraphs (65 bigrams rarely repeat 30%).
    # 0.08 calibrated on paragraph-length texts where AI hits ~5–8% repetition.
    score = min(1.0, repetition_rate / 0.08)
    score = round(score, 4)

    return {
        "score": score,
        "repetition_rate": round(repetition_rate, 4),
        "total_bigrams": total,
        "unique_bigrams": unique,
    }
