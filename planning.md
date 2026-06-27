# Provenance Guard — planning.md

> Written before any implementation code. Updated before any stretch features.

---

## System Narrative

A piece of text enters the system via `POST /submit` along with a `creator_id`. The
Flask app validates the request, then passes the raw text to two independent detection
functions running in sequence: `detector_llm.py` sends the text to Groq's LLM and gets
back a score between 0.0 and 1.0 representing how AI-like the writing appears
semantically. `detector_stylo.py` computes three statistical heuristics on the text
locally and produces a second 0.0–1.0 score representing how structurally AI-like the
writing appears. `scorer.py` combines those two scores into a single weighted confidence
score. `labeler.py` maps that score to one of three transparency label variants.
`auditor.py` writes a structured JSON entry capturing everything — timestamp, IDs, both
signal scores, the combined score, the attribution result, and the label — to an append-only
log file. The endpoint returns all of this to the caller as JSON.

A creator who disputes their classification hits `POST /appeal` with their `content_id`
and written reasoning. The app validates the content_id exists, updates its status in
`storage.py` from `"classified"` to `"under_review"`, and appends an appeal entry to
the audit log. No automated re-classification occurs; a human reviewer would use
`GET /log` to inspect the original decision and the creator's reasoning side by side.

---

## Architecture

```
SUBMISSION FLOW
===============

POST /submit
{ text, creator_id }
        │
        ▼
   Validate input
   Generate content_id (uuid4)
        │
        ├─────────────────────────────┐
        ▼                             ▼
 detector_llm.py             detector_stylo.py
 Groq LLM (llama-3.3-70b)   Pure Python heuristics
 Prompt → structured score   sentence_variance()
 Returns: llm_score 0.0–1.0  type_token_ratio()
                              punctuation_density()
                              Returns: stylo_score 0.0–1.0
        │                             │
        └─────────────────────────────┘
                      │
                      ▼
                 scorer.py
        confidence = (0.60 × llm_score)
                   + (0.40 × stylo_score)
                      │
                      ▼
                 labeler.py
        confidence → label text (one of 3 variants)
                      │
             ┌────────┴─────────┐
             ▼                  ▼
        auditor.py          JSON response
     Append log entry    { content_id,
     to audit.jsonl        attribution,
                           confidence,
                           llm_score,
                           stylo_score,
                           label,
                           status }


APPEAL FLOW
===========

POST /appeal
{ content_id, creator_reasoning }
        │
        ▼
   Validate content_id exists in storage.py
        │
        ▼
   storage.py
   status: "classified" → "under_review"
        │
        ▼
   auditor.py
   Append appeal entry to audit.jsonl
   { type: "appeal", content_id,
     creator_reasoning, timestamp,
     original_confidence, original_attribution }
        │
        ▼
   JSON response
   { appeal_received: true,
     content_id,
     status: "under_review" }
```

---

## 1. Detection Signals

### Signal 1 — LLM-based classification (Groq)

**What it measures:** Semantic and stylistic coherence holistically. The LLM assesses
whether the text reads like AI output: characteristic hedging phrases ("it is important
to note that"), over-balanced structures, formulaic transitions, uniform register, and
the kind of confident-but-hollow elaboration common in AI writing. This is a holistic
judgment that no set of rules can fully replicate.

**Output format:** A single float between 0.0 (clearly human) and 1.0 (clearly AI).
The Groq prompt will ask the model to respond with only a JSON object
`{ "score": <float>, "reasoning": "<one sentence>" }` so we can parse it reliably.

**Why this signal:** It captures what stylometrics cannot — the *meaning* of the text,
not just its surface statistics. A human who writes in a formal register will still use
idiosyncratic word choices, digress, and contradict themselves. An LLM won't.

**Blind spots:**
- Heavily edited AI output looks more human to an LLM reviewer
- Very short texts (< 80 words) give the LLM insufficient signal; scores will cluster
  near 0.5
- ESL writers with formal syntax may trigger some AI-like pattern recognition
- The LLM may reflect its own biases about what "human" writing looks like

---

### Signal 2 — Stylometric heuristics (pure Python)

**What it measures:** Statistical properties of the text's structure that differ
measurably between human and AI writing. AI text tends to be more uniform; human writing
is more variable and idiosyncratic.

**Three sub-metrics:**

**2a. Sentence length variance**
Compute the standard deviation of sentence lengths (in words) across the text.
AI text tends to produce sentences in a narrow band (12–20 words). Human writing
fluctuates more — short punchy sentences mix with long complex ones.
- High variance → more human-like → lower AI sub-score
- Normalize: map std dev of 0–15+ words to a 0.0–1.0 AI sub-score using a sigmoid-like
  inverse (high std dev → score near 0, low std dev → score near 1)

**2b. Type-token ratio (TTR)**
`unique_words / total_words` computed over the full text (or a 100-word sliding window
for texts over 300 words, to correct for length effects). AI text tends to repeat
vocabulary patterns from its training distribution. Human writing, even casual writing,
uses more idiosyncratic word choices.
- High TTR → more human-like → lower AI sub-score
- Normalize: TTR typically ranges 0.4–0.9 in practice; map to 0.0–1.0 AI sub-score
  (high TTR → score near 0)

**2c. Punctuation density**
AI text overuses commas and em-dashes in predictable patterns; it also tends to use
semicolons more than casual human writers but fewer sentence fragments and ellipses.
Compute: `(commas + semicolons + em-dashes) / word_count × 100`.
- Compare to a human baseline range; above-baseline density → higher AI sub-score

**Combined stylometric score:**
```
stylo_score = (0.40 × sentence_variance_score)
            + (0.40 × ttr_score)
            + (0.20 × punctuation_score)
```

**Output format:** A single float 0.0–1.0.

**Why this signal:** It is genuinely independent from Signal 1 — structural, not
semantic. Two signals that capture different dimensions of the same phenomenon are more
informative than two signals that both look at semantics.

**Blind spots:**
- Formal human writing (academic papers, legal briefs) has low TTR and low sentence
  variance — our heuristics will over-score these as AI
- Poetry that uses anaphora or repetition (intentional low TTR) will score high
- Very short texts (< 5 sentences) produce unreliable variance estimates
- AI text that uses bullet points or numbered lists will have artificially short, uniform
  "sentences" depending on how we tokenize

---

### Combining the signals

```python
confidence = (0.60 × llm_score) + (0.40 × stylo_score)
```

The LLM signal receives higher weight (60%) because it captures semantic patterns that
no statistical heuristic can. The stylometric signal acts as a corrective: if the LLM
scores something as borderline but the structural analysis strongly agrees, the combined
score moves off the fence. If they strongly disagree, the score stays in the uncertain
zone — which is the honest answer.

---

## 2. Uncertainty Representation

**What does a confidence score of 0.6 mean?**
It means both signals leaned AI but neither strongly. The text has some AI-like
properties but also some human-like ones. We should not label this as AI — that risks
a false positive on a human writer. We should label it uncertain and give the creator
a path to appeal.

**Thresholds:**

```
0.00 ──────────────────────────────────────────── 1.00
      [   HUMAN   ]   [   UNCERTAIN   ]   [   AI   ]
      0.00     0.35   0.35         0.65   0.65    1.00
```

- **< 0.35** → `likely_human` — low enough that we're confident labeling it human-written
- **0.35 – 0.65** → `uncertain` — signal is insufficient to make a claim either way
- **> 0.65** → `likely_ai` — high enough across both signals to apply the AI label

**Why these thresholds?**
The assignment notes that a false positive (labeling a human as AI) is worse than a
false negative. The thresholds reflect this asymmetry: the uncertain zone is not centered
at 0.5. A score of 0.5 falls in the uncertain range and will never trigger an AI label.
To get an AI label, both signals need to point clearly in the same direction, producing
a combined score above 0.65.

**How to validate scores are meaningful:**
Run the four test inputs from the spec. The clearly-AI paragraph should score above
0.75. The casual ramen review should score below 0.25. The two borderline cases should
fall between 0.35 and 0.65. If any of these don't hold, investigate both signal scores
separately to find which is misbehaving before adjusting weights.

---

## 3. Transparency Label Design

All three variants must appear verbatim in the final README. These are the exact strings
`labeler.py` will return.

---

**High-confidence AI label** (confidence > 0.65):

```
⚠️ Likely AI-generated

Our system detected strong indicators that this content may have been produced
with an AI writing tool. Confidence: {confidence_pct}% AI.

This label reflects automated analysis and may not be accurate. If you wrote
this yourself, you can submit an appeal below — we'll review it and update
this classification.
```

---

**Uncertain label** (confidence 0.35 – 0.65):

```
🔍 Authorship unclear

We weren't able to confidently determine whether this content was written by
a human or generated by an AI tool. Our signals disagreed or were too weak
to make a reliable call.

No action will be taken based on this result. You can still submit an appeal
if you'd like to add context about how this was written.
```

---

**High-confidence human label** (confidence < 0.35):

```
✅ Appears human-written

Our analysis found no strong indicators of AI generation in this content.
Confidence: {confidence_pct}% human-authored.
```

---

**Design notes:**
- The AI label explicitly mentions appeals because that's the path to correct a false
  positive — it must be visible
- The uncertain label says "no action will be taken" because a platform should not
  penalize a creator for an uncertain verdict
- The human label is brief — there is nothing the reader needs to do
- `{confidence_pct}` is `round((1 - confidence) * 100)` for the human label and
  `round(confidence * 100)` for the AI label, shown as a whole number

---

## 4. Appeals Workflow

**Who can appeal:** Any creator who has a `content_id` from a `/submit` response.
No authentication is implemented; in a real system this would require auth.

**What they provide:**
- `content_id` (required) — the UUID from their submission
- `creator_reasoning` (required, min 20 characters) — their explanation of why the
  classification is wrong. The spec says to "capture the creator's reasoning" — we
  enforce minimum length so this field is useful, not just a checkbox.

**What the system does when an appeal is received:**
1. Look up `content_id` in `storage.py`. If not found, return 404.
2. Update the stored record: `status: "classified"` → `status: "under_review"`.
3. Write an appeal entry to `audit.jsonl`:
   ```json
   {
     "type": "appeal",
     "content_id": "...",
     "creator_id": "...",
     "timestamp": "...",
     "creator_reasoning": "...",
     "original_attribution": "likely_ai",
     "original_confidence": 0.78,
     "original_llm_score": 0.81,
     "original_stylo_score": 0.72,
     "status": "under_review"
   }
   ```
4. Return a confirmation response:
   ```json
   {
     "appeal_received": true,
     "content_id": "...",
     "status": "under_review",
     "message": "Your appeal has been received and will be reviewed. No automated
                 re-classification will occur — a human reviewer will assess your
                 submission."
   }
   ```

**What a human reviewer sees (via GET /log):** The original submission entry (with both
signal scores, the combined confidence, and the label applied) followed by the appeal
entry (with the creator's reasoning). Everything needed to make a judgment is in the log.

**Automated re-classification:** Not implemented. The spec does not require it. Appeals
are a human-review process.

---

## 5. Anticipated Edge Cases

**Edge case 1: Non-native English speaker with formal syntax**
A writer whose first language is not English may produce text that is grammatically
correct but syntactically uniform — shorter sentences, simpler vocabulary, consistent
structure. This will produce a high stylometric AI score (low TTR, low sentence
variance) even though the content is entirely original. The LLM signal may also
misfire if the formal register triggers AI-pattern recognition. This is the most likely
source of false positives on a creative writing platform with international users.
**Mitigation in design:** The 0.65 AI threshold and the wide uncertain zone mean this
type of writer will most likely land in `uncertain` rather than `likely_ai`. The appeal
workflow exists specifically for this case.

**Edge case 2: Short texts under 100 words**
A short poem, a haiku, or a brief caption gives the stylometric signal almost no data.
Sentence length variance computed over 3 sentences is nearly meaningless. TTR on 50
words is unreliable. The LLM signal is similarly weakened because there is little to
assess holistically. Scores for very short texts will cluster near 0.5, producing
`uncertain` labels almost always — which is the honest answer. We should document this
in known limitations.

**Edge case 3: Poetry using intentional repetition**
Anaphora, refrains, and repeated structural phrases are legitimate poetic devices that
will devastate the TTR and sentence-variance scores. A poet who writes "I have dreamed
of this. I have dreamed of the water. I have dreamed and I am tired." will score high
on the stylometric AI signal through no fault of their own. The LLM signal may partially
counteract this if it recognizes the poetic register, but it is not guaranteed.

**Edge case 4: Lightly human-edited AI output**
A creator who generates a draft with AI and then rewrites significant portions may
produce text that is genuinely hybrid. Our system has no concept of "partial AI" — it
produces a single score. Heavy editing may push the score into the uncertain range,
which is the most honest label we can apply. This is not a failure mode; it reflects
the genuine ambiguity of the content.

**Edge case 5: Academic or legal writing**
Dense formal prose with long sentences and specialist vocabulary tends to score
unexpectedly high on stylometric signals because it resembles the training distribution
of LLMs. A law review article or scientific paper written entirely by a human may
trigger a borderline `likely_ai` label. The 0.65 threshold should catch most of these
(the LLM signal will recognize academic writing as human), but edge cases will exist.

---

## Stretch Features

> Per the assignment: update this section before starting each stretch feature.
> Document completed features in README (what you built + how it works), not just here.

---

### S1 — Ensemble Detection (3rd signal + voting)

**What it adds:** A third independent detection signal and a documented weighting/voting
approach that replaces the two-signal weighted average in `scorer.py`.

**Third signal — Bigram repetition rate (pure Python)**

What it measures: How often word pairs (bigrams) repeat across the text. AI writing
tends to reuse specific two-word combinations because its outputs are shaped by training
distribution patterns (e.g., "it is," "in the," "as well," "it is important"). Human
writing uses more varied collocations, especially in creative or personal content.

How to compute:
```
bigrams = list of all consecutive word pairs in the text (lowercased)
repetition_rate = (total_bigrams - unique_bigrams) / total_bigrams
```
A high repetition rate → more AI-like → higher `bigram_score` (0.0–1.0).
Normalize: map 0.0–0.3+ repetition rate to a 0.0–1.0 AI score using a linear scale
(0.0 → 0.0, 0.30+ → 1.0, clamp at 1.0).

Why this is genuinely independent: Signal 1 is semantic. Signal 2 is structural
(sentence-level). Signal 3 is lexical/collocational — it operates at the word-pair
level and measures a different dimension of uniformity.

Blind spots: Very short texts will have almost no repeated bigrams regardless of origin.
Texts that use a lot of proper nouns or specialized vocabulary (place names, technical
terms) will score artificially low on repetition even if AI-generated.

**Updated scoring (ensemble):**
```
confidence = (0.45 × llm_score)
           + (0.30 × stylo_score)
           + (0.25 × bigram_score)
```
LLM weight drops from 0.60 to 0.45; stylo drops from 0.40 to 0.30; bigram adds 0.25.
Rationale: LLM still gets the most weight (holistic judgment), but the two structural
signals together now carry 55% to counterbalance potential LLM hallucination.

**Voting layer (for attribution only — not used for confidence score):**
Each signal casts a vote: "AI" if its score > 0.50, "Human" otherwise.
- 3/3 votes AI → attribution: `likely_ai`
- 2/3 votes AI + weighted confidence > 0.65 → attribution: `likely_ai`
- 2/3 votes AI + weighted confidence ≤ 0.65 → attribution: `uncertain`
- 0–1/3 votes AI → attribution: `likely_human`

The confidence score still uses the weighted formula; the voting layer adds a
cross-check so a single outlier signal can't dominate the attribution.

**Files to create/modify:**
- Create `detector_bigram.py` — `detect_with_bigrams(text: str) -> dict`
- Modify `scorer.py` — update `combine_scores()` to accept 3 inputs, apply new
  weights, add voting logic
- Modify `auditor.py` — add `bigram_score` field to log entries
- Modify `app.py` — call `detect_with_bigrams()` in the submit route

**Updated architecture (submission flow with 3 signals):**
```
POST /submit
        │
        ├─────────────────┬─────────────────┐
        ▼                 ▼                 ▼
 detector_llm.py  detector_stylo.py  detector_bigram.py
 llm_score        stylo_score        bigram_score
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
                     scorer.py
             ensemble weighting + voting
             → confidence, attribution
```

**AI Tool Plan for S1:**
- Provide: Signal 1, Signal 2, and S1 sections of this doc + updated architecture
- Ask for: `detect_with_bigrams()` in `detector_bigram.py` + updated `combine_scores()`
  in `scorer.py` with 3-signal weighting and voting logic
- Verify: Run all 4 test inputs. Compare attribution results between old 2-signal
  system and new 3-signal system. The borderline cases are the most interesting — does
  the 3rd signal break ties differently? Check that `bigram_score` appears in `/log` output.

---

### S2 — Provenance Certificate

**What it adds:** A "verified human" credential that a creator can earn through an
additional verification step. Once earned, it is displayed on their content and visible
in the submission response.

**Verification step design:**
The creator submits a `POST /verify` request with their `content_id` and a
`verification_sample` — a second piece of their own writing that was not submitted
through the main pipeline. The system runs both the original submission and the sample
through the full detection pipeline. If both score below 0.40 (well into human territory)
*and* the two pieces show stylometric similarity (they were written by the same person),
a provenance certificate is issued.

The stylometric similarity check compares the TTR and sentence length variance of both
texts. If they're within a ±0.15 range, they are likely from the same writer.
This is not cryptographic proof — it's a best-effort heuristic, and the label says so.

**Certificate display:**
The `POST /submit` response and `GET /log` entries gain a new field:
```json
"provenance_certificate": {
  "issued": true,
  "issued_at": "2025-05-01T10:22:00Z",
  "display_label": "✦ Verified Human — Creator has passed an additional verification step."
}
```
If no certificate exists: `"provenance_certificate": null`

**New endpoint:**
```
POST /verify
{ content_id, verification_sample }
→ { certificate_issued: bool, content_id, display_label, reasoning }
```

**Failure cases to handle:**
- `content_id` not found → 404
- `content_id` already has a certificate → return existing certificate, don't reissue
- Either text scores above 0.40 → deny, return `{ certificate_issued: false, reason: "..." }`
- Stylometric similarity check fails → deny, return reasoning

**Files to create/modify:**
- Create `certificates.py` — `issue_certificate()`, `get_certificate()`, in-memory store
- Modify `storage.py` — add certificate field to content records
- Modify `app.py` — add `POST /verify` route
- Modify `auditor.py` — log certificate issuance events

**AI Tool Plan for S2:**
- Provide: S2 section of this doc + architecture diagram + detection signals section
- Ask for: `POST /verify` route + `issue_certificate()` + stylometric similarity check
  as a standalone function in `certificates.py`
- Verify: Submit two clearly human texts from the same author (write them yourself).
  Confirm certificate is issued. Then try with one human + one AI text. Confirm denial.
  Check that `GET /log` shows the certificate event.

---

### S3 — Analytics Dashboard

**What it adds:** A `GET /analytics` endpoint (and optional HTML view) showing
detection patterns across all submissions.

**Three metrics to display:**

1. **Detection distribution** — of all submissions, what % landed in each category:
   `likely_ai` / `uncertain` / `likely_human`. Computed by reading `audit.jsonl` and
   counting attribution values across all `"type": "submission"` entries.

2. **Appeal rate** — `total appeals / total submissions × 100`. Shows how often
   creators contest classifications. A high appeal rate is a signal the system may
   be miscalibrated.

3. **Average confidence score over time** — group submissions by day, compute mean
   confidence per day. A rising average could indicate an adversarial pattern (more
   AI content being submitted) or a calibration drift. Returned as a list of
   `{ date, avg_confidence, count }` objects.

**Endpoint response:**
```json
GET /analytics
{
  "total_submissions": 42,
  "detection_distribution": {
    "likely_ai": { "count": 18, "pct": 42.9 },
    "uncertain":  { "count": 11, "pct": 26.2 },
    "likely_human": { "count": 13, "pct": 30.9 }
  },
  "appeal_rate": { "total_appeals": 5, "pct_of_submissions": 11.9 },
  "confidence_over_time": [
    { "date": "2025-05-01", "avg_confidence": 0.61, "count": 7 },
    { "date": "2025-05-02", "avg_confidence": 0.54, "count": 12 }
  ]
}
```

**Optional HTML view:** Add a `GET /dashboard` route that returns a minimal HTML page
rendering the same data as a human-readable summary. No JavaScript framework needed —
plain HTML with inline style is fine. This is a bonus on top of the JSON endpoint;
the JSON endpoint is the deliverable.

**Files to create/modify:**
- Create `analytics.py` — `compute_analytics() -> dict` reads `audit.jsonl` and
  computes all three metrics
- Modify `app.py` — add `GET /analytics` route (and optionally `GET /dashboard`)

**AI Tool Plan for S3:**
- Provide: S3 section of this doc + audit log entry format from section 4
- Ask for: `compute_analytics()` in `analytics.py` that reads `audit.jsonl` and
  returns the documented JSON structure, plus the `GET /analytics` route in `app.py`
- Verify: Generate 10+ submissions spanning all 3 attribution categories, then at least
  2 appeals. Hit `GET /analytics` and confirm: distribution sums to 100%, appeal rate
  matches manual count, time-series has one entry per day that had submissions.

---

### S4 — Multi-Modal Support (image description metadata)

**What it adds:** Extends `POST /submit` to handle a second content type —
structured metadata describing an image — in addition to plain text. This lets the
system classify whether an *image description* was written by a human or generated
by an AI captioning tool.

**How it works:**
The request body gains an optional `content_type` field: `"text"` (default) or
`"image_description"`. If `content_type` is `"image_description"`, the `text` field
contains a description of an image (alt text, caption, or EXIF-style metadata string).

The detection pipeline adjusts:
- Signal 1 (LLM) gets a modified prompt tuned for image descriptions — asking whether
  the description reads as AI-generated alt text vs. a human caption. The prompt
  includes examples of each.
- Signal 2 (stylometrics) is adjusted: sentence length variance and TTR are still
  computed, but the thresholds are recalibrated because image descriptions are inherently
  shorter and more uniform than prose. The `punctuation_density` sub-metric is dropped
  (not meaningful for captions) and replaced with a **subject-verb-object pattern score**:
  AI image descriptions overuse the pattern "A [noun] [verbing] in/on a [noun]."
- Signal 3 (bigrams, if S1 is implemented) applies unchanged.

The response gains a `content_type` field so the caller knows which pipeline ran.
The audit log records `content_type` on every entry.

**Threshold recalibration for image descriptions:**
Image descriptions are short. Scores will be less reliable. The thresholds shift:
- `likely_human`: confidence < 0.30 (tighter — we need stronger evidence for short text)
- `uncertain`: 0.30 – 0.70
- `likely_ai`: confidence > 0.70 (higher bar — short text produces noisier scores)

These more conservative thresholds mean most image descriptions will land in `uncertain`
unless the signals agree strongly. That's the honest behavior for a noisy input type.

**Files to create/modify:**
- Create `detector_image_meta.py` — `detect_image_description(text: str) -> dict`
  with the modified stylometric approach (SVO pattern score replaces punctuation density)
- Modify `detector_llm.py` — add `prompt_type` parameter to switch between text and
  image description prompts
- Modify `scorer.py` — add `content_type` parameter; use recalibrated thresholds
  when `content_type == "image_description"`
- Modify `app.py` — read `content_type` from request body, route to correct detectors
- Modify `auditor.py` — log `content_type` field

**AI Tool Plan for S4:**
- Provide: S4 section of this doc + Signal 1 and Signal 2 sections (to show existing
  patterns to extend) + architecture diagram
- Ask for: `detect_image_description()` in `detector_image_meta.py` + updated
  `detector_llm.py` with `prompt_type` parameter + updated `scorer.py` with
  content_type branching
- Verify: Submit 3 plain-text inputs (confirm `content_type: "text"` in response).
  Submit 2 clearly AI-generated image descriptions (e.g., "A serene mountain landscape
  with snow-capped peaks reflecting in a crystal-clear lake at sunset.") and 2 clearly
  human captions (e.g., "dad at the beach, he hated that hat"). Confirm different
  content types produce different log entries. Confirm the recalibrated thresholds
  produce more `uncertain` results for short image descriptions than plain text would.

---

## AI Tool Plan

### Milestone 3 — Submission endpoint + Signal 1

**Spec sections to provide:**
- This file's "System Narrative" section
- The "Detection Signals → Signal 1" section
- The Architecture diagram (submission flow only)

**What to ask the AI tool to generate:**
1. Flask app skeleton in `app.py`: `POST /submit` route stub (accepts JSON body, returns
   hardcoded response), `GET /log` route stub, Flask-Limiter setup with `storage_uri="memory://"`,
   and `.env` loading via python-dotenv
2. `detect_with_llm(text: str) -> dict` in `detector_llm.py`: sends text to Groq
   `llama-3.3-70b-versatile` with a prompt that returns `{ "score": float, "reasoning": str }`,
   parses the response, returns the score as a float 0.0–1.0
3. `log_entry(entry: dict)` in `auditor.py`: appends a JSON-serialized dict to
   `logs/audit.jsonl`, creates the file if it doesn't exist, and `get_log(n=20) -> list`
   reads the last N entries

**How to verify before wiring in:**
- Call `detect_with_llm()` directly in a test script with the clearly-AI paragraph and
  the ramen review from the spec. Confirm it returns a float, not a string or dict.
- Confirm the float is above 0.6 for the AI paragraph and below 0.4 for the ramen review.
- Hit `POST /submit` with curl; confirm the response includes `content_id` as a UUID string.
- Hit `GET /log`; confirm the response is valid JSON with an `entries` array.

---

### Milestone 4 — Signal 2 + confidence scoring

**Spec sections to provide:**
- The "Detection Signals → Signal 2" section (all three sub-metrics with formulas)
- The "Uncertainty Representation" section (thresholds and weighting formula)
- The Architecture diagram (submission flow, scorer box)

**What to ask the AI tool to generate:**
1. `detect_with_stylometrics(text: str) -> dict` in `detector_stylo.py`: implements
   `sentence_length_variance_score()`, `type_token_ratio_score()`, and
   `punctuation_density_score()` as separate helper functions, then combines them with
   the documented 0.40/0.40/0.20 weights into a single `stylo_score` float 0.0–1.0
2. `combine_scores(llm_score: float, stylo_score: float) -> dict` in `scorer.py`:
   applies the 60/40 weighting formula, returns `{ confidence, attribution, llm_score, stylo_score }`
   where `attribution` is one of `"likely_ai"`, `"uncertain"`, `"likely_human"` based
   on the defined thresholds

**How to verify:**
- Run all 4 test inputs from the spec. Print `llm_score`, `stylo_score`, and combined
  `confidence` for each. Expected ranges:
  - Clearly AI paragraph: confidence > 0.75
  - Ramen review: confidence < 0.25
  - Academic paragraph: confidence 0.35–0.65 (borderline)
  - Lightly edited AI: confidence 0.35–0.65 (borderline)
- If either borderline case scores above 0.65, investigate which signal is misfiring
  by comparing their individual scores.

---

### Milestone 5 — Production layer

**Spec sections to provide:**
- The "Transparency Label Design" section (all 3 variants verbatim)
- The "Appeals Workflow" section (fields, status change, log format)
- The Architecture diagram (both flows)

**What to ask the AI tool to generate:**
1. `get_label(confidence: float, attribution: str) -> str` in `labeler.py`: maps the
   confidence score and attribution string to the exact label text defined in this document,
   substituting `{confidence_pct}` with the computed percentage
2. `POST /appeal` route in `app.py`: validates `content_id` and `creator_reasoning`,
   calls `storage.update_status()`, calls `auditor.log_appeal()`, returns confirmation JSON
3. `update_status(content_id: str, status: str)` in `storage.py`: in-memory dict
   (or SQLite) mapping content_id to its current status
4. Flask-Limiter decorator on `POST /submit`: `@limiter.limit("10 per minute;100 per day")`

**How to verify:**
- Submit 3 inputs: one that should score above 0.65, one below 0.35, one in between.
  Confirm all three label variants appear in the responses — not just one.
- Run `POST /appeal` with a real `content_id` from a previous submit. Confirm `GET /log`
  shows the entry with `"status": "under_review"` and `creator_reasoning` populated.
- Run the 12-request rate limit test from the spec. Confirm requests 11 and 12 return 429.
- Inspect `audit.jsonl` directly. Confirm it has 3+ entries, each valid JSON, covering
  at least one submission and one appeal.

---

### Stretch S1 — Ensemble detection (3rd signal)

> Update this section before implementing. Full design in the Stretch Features section above.

- Update planning.md: add `detector_bigram.py` to architecture diagram
- Implement, verify, document in README under "Stretch Features: Ensemble Detection"

---

### Stretch S2 — Provenance certificate

> Update this section before implementing. Full design in the Stretch Features section above.

- Update planning.md: add `POST /verify` to architecture diagram and endpoint list
- Implement, verify, document in README under "Stretch Features: Provenance Certificate"

---

### Stretch S3 — Analytics dashboard

> Update this section before implementing. Full design in the Stretch Features section above.

- Update planning.md: add `GET /analytics` and `analytics.py` to architecture
- Implement, verify, document in README under "Stretch Features: Analytics Dashboard"

---

### Stretch S4 — Multi-modal support

> Update this section before implementing. Full design in the Stretch Features section above.

- Update planning.md: add `content_type` branching to architecture diagram
- Implement, verify, document in README under "Stretch Features: Multi-Modal Support"