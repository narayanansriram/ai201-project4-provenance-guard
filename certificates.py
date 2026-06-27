from datetime import datetime, timezone
from detector_stylo import detect_with_stylometrics

# In-memory store: content_id → certificate dict
_store: dict = {}


def _stylo_similarity(text_a: str, text_b: str) -> tuple:
    """
    Compares TTR and sentence length variance between two texts.
    Returns (is_similar: bool, reasoning: str).
    Both metrics must be within ±0.15 to pass.
    """
    a = detect_with_stylometrics(text_a)
    b = detect_with_stylometrics(text_b)

    ttr_diff = abs(a["ttr_score"] - b["ttr_score"])
    var_diff = abs(a["variance_score"] - b["variance_score"])

    if ttr_diff <= 0.15 and var_diff <= 0.15:
        return True, f"Stylometric profiles are similar (TTR diff: {ttr_diff:.3f}, variance diff: {var_diff:.3f})"
    parts = []
    if ttr_diff > 0.15:
        parts.append(f"vocabulary diversity differs too much (TTR diff: {ttr_diff:.3f})")
    if var_diff > 0.15:
        parts.append(f"sentence structure differs too much (variance diff: {var_diff:.3f})")
    return False, "Texts appear to be from different authors — " + "; ".join(parts)


def issue_certificate(content_id: str, original_confidence: float,
                      original_text: str, sample_text: str,
                      sample_confidence: float) -> dict:
    """
    Issues a provenance certificate if:
    - original_confidence < 0.40 (original submission scores human)
    - sample_confidence < 0.40 (verification sample scores human)
    - Both texts are stylometrically similar (same-author check)

    Returns the certificate dict (issued=True or False).
    """
    if original_confidence >= 0.40:
        cert = {
            "issued": False,
            "content_id": content_id,
            "reason": f"Original submission confidence too high ({original_confidence:.3f} ≥ 0.40 threshold).",
        }
        _store[content_id] = cert
        return cert

    if sample_confidence >= 0.40:
        cert = {
            "issued": False,
            "content_id": content_id,
            "reason": f"Verification sample confidence too high ({sample_confidence:.3f} ≥ 0.40 threshold).",
        }
        _store[content_id] = cert
        return cert

    similar, similarity_reasoning = _stylo_similarity(original_text, sample_text)
    if not similar:
        cert = {
            "issued": False,
            "content_id": content_id,
            "reason": similarity_reasoning,
        }
        _store[content_id] = cert
        return cert

    cert = {
        "issued": True,
        "content_id": content_id,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "display_label": "✦ Verified Human — Creator has passed an additional verification step.",
        "reasoning": similarity_reasoning,
    }
    _store[content_id] = cert
    return cert


def get_certificate(content_id: str) -> dict | None:
    return _store.get(content_id)
