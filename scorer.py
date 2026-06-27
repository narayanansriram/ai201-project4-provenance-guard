def combine_scores(llm_score: float, stylo_score: float, bigram_score: float = None,
                   content_type: str = "text") -> dict:
    """
    Two-signal mode (bigram_score=None): LLM 60%, stylometric 40%.
    Three-signal ensemble mode: LLM 45%, stylometric 30%, bigram 25% + voting layer.

    Thresholds (text): < 0.35 → likely_human, 0.35–0.65 → uncertain, > 0.65 → likely_ai.
    Thresholds (image_description): < 0.30 → likely_human, 0.30–0.70 → uncertain, > 0.70 → likely_ai.
    Image description thresholds are more conservative — short texts produce noisier scores.

    Voting layer (ensemble only):
      3/3 AI votes                          → likely_ai
      2/3 AI votes + confidence > high_t    → likely_ai
      2/3 AI votes + confidence <= high_t   → uncertain
      0–1/3 AI votes                        → likely_human
    """
    low_t = 0.30 if content_type == "image_description" else 0.35
    high_t = 0.70 if content_type == "image_description" else 0.65

    if bigram_score is None:
        confidence = round((0.60 * llm_score) + (0.40 * stylo_score), 4)
        if confidence < low_t:
            attribution = "likely_human"
        elif confidence > high_t:
            attribution = "likely_ai"
        else:
            attribution = "uncertain"
        return {
            "confidence": confidence,
            "attribution": attribution,
            "llm_score": round(llm_score, 4),
            "stylo_score": round(stylo_score, 4),
        }

    # Ensemble mode
    confidence = round((0.45 * llm_score) + (0.30 * stylo_score) + (0.25 * bigram_score), 4)

    votes_ai = sum([
        1 if llm_score > 0.50 else 0,
        1 if stylo_score > 0.50 else 0,
        1 if bigram_score > 0.50 else 0,
    ])

    if votes_ai == 3:
        attribution = "likely_ai"
    elif votes_ai == 2 and confidence > high_t:
        attribution = "likely_ai"
    elif votes_ai == 2:
        attribution = "uncertain"
    else:
        # 0–1 votes: fall back to confidence thresholds rather than hardcoding human.
        # Matters for image_description where short texts often yield only 1 signal vote
        # even on clearly AI content (stylometric and bigram can't work on 1 sentence).
        if confidence > high_t:
            attribution = "likely_ai"
        elif confidence >= low_t:
            attribution = "uncertain"
        else:
            attribution = "likely_human"

    return {
        "confidence": confidence,
        "attribution": attribution,
        "llm_score": round(llm_score, 4),
        "stylo_score": round(stylo_score, 4),
        "bigram_score": round(bigram_score, 4),
        "votes_ai": votes_ai,
    }
