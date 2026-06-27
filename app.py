import uuid
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import auditor
import storage
from detector_llm import detect_with_llm
from detector_stylo import detect_with_stylometrics
from detector_bigram import detect_with_bigrams
from detector_image_meta import detect_image_description
from scorer import combine_scores
from labeler import get_label
import certificates
from analytics import compute_analytics

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    creator_id = body.get("creator_id", "").strip()

    if not text:
        return jsonify({"error": "text field is required"}), 400
    if not creator_id:
        return jsonify({"error": "creator_id field is required"}), 400

    content_id = str(uuid.uuid4())
    content_type = body.get("content_type", "text").strip()
    if content_type not in ("text", "image_description"):
        content_type = "text"

    llm_result = detect_with_llm(text, prompt_type=content_type)
    bigram_result = detect_with_bigrams(text)

    if content_type == "image_description":
        stylo_result = detect_image_description(text)
        stylo_detail_key = {
            "variance_score": stylo_result["variance_score"],
            "ttr_score": stylo_result["ttr_score"],
            "svo_score": stylo_result["svo_score"],
        }
    else:
        stylo_result = detect_with_stylometrics(text)
        stylo_detail_key = {
            "variance_score": stylo_result["variance_score"],
            "ttr_score": stylo_result["ttr_score"],
            "punct_score": stylo_result["punct_score"],
        }

    scored = combine_scores(llm_result["score"], stylo_result["score"],
                            bigram_result["score"], content_type=content_type)
    label = get_label(scored["confidence"], scored["attribution"])

    record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": scored["attribution"],
        "confidence": scored["confidence"],
        "llm_score": scored["llm_score"],
        "stylo_score": scored["stylo_score"],
        "bigram_score": scored["bigram_score"],
        "votes_ai": scored["votes_ai"],
        "stylo_detail": stylo_detail_key,
        "bigram_detail": {
            "repetition_rate": bigram_result["repetition_rate"],
            "total_bigrams": bigram_result["total_bigrams"],
            "unique_bigrams": bigram_result["unique_bigrams"],
        },
        "label": label,
        "status": "classified",
        "provenance_certificate": None,
    }

    storage.save(content_id, record)
    auditor.log_entry({
        "type": "submission",
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": scored["attribution"],
        "confidence": scored["confidence"],
        "llm_score": scored["llm_score"],
        "stylo_score": scored["stylo_score"],
        "bigram_score": scored["bigram_score"],
        "votes_ai": scored["votes_ai"],
        "status": "classified",
    })

    return jsonify(record), 200


@app.route("/verify", methods=["POST"])
def verify():
    body = request.get_json(silent=True) or {}
    content_id = body.get("content_id", "").strip()
    verification_sample = body.get("verification_sample", "").strip()

    if not content_id:
        return jsonify({"error": "content_id is required"}), 400
    if not verification_sample:
        return jsonify({"error": "verification_sample is required"}), 400

    record = storage.get(content_id)
    if record is None:
        return jsonify({"error": "content_id not found"}), 404

    existing = certificates.get_certificate(content_id)
    if existing and existing.get("issued"):
        return jsonify({
            "certificate_issued": True,
            "content_id": content_id,
            "display_label": existing["display_label"],
            "reasoning": "Certificate already issued.",
        }), 200

    # Run full pipeline on the verification sample
    llm_result = detect_with_llm(verification_sample)
    stylo_result = detect_with_stylometrics(verification_sample)
    bigram_result = detect_with_bigrams(verification_sample)
    sample_scored = combine_scores(llm_result["score"], stylo_result["score"], bigram_result["score"])

    original_text = body.get("original_text", "")
    cert = certificates.issue_certificate(
        content_id=content_id,
        original_confidence=record["confidence"],
        original_text=original_text,
        sample_text=verification_sample,
        sample_confidence=sample_scored["confidence"],
    )

    if cert["issued"]:
        record["provenance_certificate"] = cert
        storage.save(content_id, record)
        auditor.log_entry({
            "type": "certificate",
            "content_id": content_id,
            "creator_id": record.get("creator_id"),
            "issued": True,
            "display_label": cert["display_label"],
        })

    return jsonify({
        "certificate_issued": cert["issued"],
        "content_id": content_id,
        "display_label": cert.get("display_label"),
        "reasoning": cert.get("reasoning") or cert.get("reason"),
    }), 200 if cert["issued"] else 422


@app.route("/appeal", methods=["POST"])
def appeal():
    body = request.get_json(silent=True) or {}
    content_id = body.get("content_id", "").strip()
    creator_reasoning = body.get("creator_reasoning", "").strip()

    if not content_id:
        return jsonify({"error": "content_id is required"}), 400
    if len(creator_reasoning) < 20:
        return jsonify({"error": "creator_reasoning must be at least 20 characters"}), 400

    record = storage.get(content_id)
    if record is None:
        return jsonify({"error": "content_id not found"}), 404

    storage.update_status(content_id, "under_review")

    auditor.log_entry({
        "type": "appeal",
        "content_id": content_id,
        "creator_id": record.get("creator_id"),
        "creator_reasoning": creator_reasoning,
        "original_attribution": record.get("attribution"),
        "original_confidence": record.get("confidence"),
        "original_llm_score": record.get("llm_score"),
        "original_stylo_score": record.get("stylo_score"),
        "status": "under_review",
    })

    return jsonify({
        "appeal_received": True,
        "content_id": content_id,
        "status": "under_review",
        "message": (
            "Your appeal has been received and will be reviewed. "
            "No automated re-classification will occur — a human reviewer will assess your submission."
        ),
    }), 200


@app.route("/log", methods=["GET"])
def log():
    n = request.args.get("n", 20, type=int)
    return jsonify({"entries": auditor.get_log(n)}), 200


@app.route("/analytics", methods=["GET"])
def analytics():
    return jsonify(compute_analytics()), 200


if __name__ == "__main__":
    app.run(debug=True)
