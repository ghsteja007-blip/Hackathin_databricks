from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_SPELL_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 8.0


def _model_for(env_name: str, fallback: str = DEFAULT_OPENAI_MODEL) -> str:
    return os.getenv(env_name) or os.getenv("OPENAI_MODEL") or fallback


def _openai_timeout_seconds() -> float:
    try:
        return max(2.0, float(os.getenv("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_OPENAI_TIMEOUT_SECONDS


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=_openai_timeout_seconds(), max_retries=0)


NEED_SYNONYMS = {
    "dialysis": ["dialysis", "hemodialysis", "nephrology", "kidney", "renal"],
    "kidney": ["kidney", "renal", "nephrology", "dialysis"],
    "emergency": ["emergency", "24/7", "24 hrs", "icu", "ambulance", "trauma", "critical care"],
    "surgery": ["surgery", "surgical", "general surgery", "operation", "operating theater", "laparoscopic"],
    "emergency surgery": ["emergency", "surgery", "trauma", "icu", "ambulance", "general surgery"],
    "maternity": ["maternity", "obstetrics", "gynecology", "delivery", "neonatal", "nicu"],
    "pregnancy": ["pregnancy", "obstetrics", "gynecology", "delivery", "maternal", "neonatal"],
    "pediatric": ["pediatric", "paediatric", "children", "neonatal", "nicu"],
    "heart": ["heart", "cardiology", "cardiac", "ecg", "echo", "cath"],
    "cardiac": ["cardiac", "cardiology", "heart", "ecg", "echo", "cath"],
    "cancer": ["cancer", "oncology", "chemotherapy", "radiotherapy"],
    "orthopedic": ["orthopedic", "orthopaedic", "bone", "joint", "trauma", "fracture"],
    "eye": ["eye", "ophthalmology", "ophthalmic", "vision"],
    "dental": ["dental", "dentistry", "dentist", "root canal", "implant"],
    "diabetes": ["diabetes", "endocrinology", "diabetic"],
    "blood": ["blood bank", "blood", "pathology", "laboratory"],
}


@dataclass
class ParsedReferralQuery:
    raw_query: str
    care_need: str
    location: str
    urgency: str = "routine"
    required_terms: list[str] = field(default_factory=list)
    source: str = "fallback"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "care_need": self.care_need,
            "location": self.location,
            "urgency": self.urgency,
            "required_terms": self.required_terms,
            "source": self.source,
            "notes": self.notes,
        }


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen = set()
    output = []
    for term in terms:
        cleaned = re.sub(r"\s+", " ", str(term).strip().lower())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def expand_need_terms(care_need: str, initial_terms: list[str] | None = None) -> list[str]:
    text = care_need.lower()
    terms = list(initial_terms or [])
    terms.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9/+.-]{2,}", text))

    for key, synonyms in NEED_SYNONYMS.items():
        if key in text:
            terms.extend(synonyms)

    if "emergency" in text and "surgery" in text:
        terms.extend(NEED_SYNONYMS["emergency surgery"])

    return _dedupe_terms(terms)


def _fallback_parse(raw_query: str) -> ParsedReferralQuery:
    query = re.sub(r"\s+", " ", raw_query.strip())
    match = re.search(r"\b(?:near|around|in|at)\s+(.+)$", query, flags=re.IGNORECASE)

    if match:
        care_need = query[: match.start()].strip(" ,.;")
        location = match.group(1).strip(" ,.;?")
    else:
        care_need = query
        location = ""

    care_need = re.sub(r"^(find|show|search|need|looking for)\s+", "", care_need, flags=re.IGNORECASE).strip()
    urgency = "emergency" if re.search(r"\b(emergency|urgent|trauma|critical)\b", query, re.I) else "routine"

    return ParsedReferralQuery(
        raw_query=raw_query,
        care_need=care_need or query,
        location=location,
        urgency=urgency,
        required_terms=expand_need_terms(care_need or query),
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _parse_with_openai(raw_query: str, fallback: ParsedReferralQuery) -> ParsedReferralQuery | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        client = _openai_client(api_key)
    except Exception:
        fallback.notes = "OpenAI SDK is not installed; used fallback parsing."
        return None

    system_prompt = (
        "Extract referral search intent from the user's text. "
        "Return only JSON with keys: care_need, location, urgency, required_terms. "
        "required_terms should be short facility-record match terms, not diagnosis advice. "
        "Do not invent facility names."
    )

    response = client.responses.create(
        model=_model_for("OPENAI_PARSE_MODEL"),
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": raw_query},
        ],
    )

    payload = _extract_json(getattr(response, "output_text", "") or "")
    care_need = str(payload.get("care_need") or fallback.care_need).strip()
    location = str(payload.get("location") or fallback.location).strip()
    urgency = str(payload.get("urgency") or fallback.urgency).strip().lower()
    terms = payload.get("required_terms") or []
    if not isinstance(terms, list):
        terms = []

    return ParsedReferralQuery(
        raw_query=raw_query,
        care_need=care_need,
        location=location,
        urgency="emergency" if "emergency" in urgency or "urgent" in urgency else "routine",
        required_terms=expand_need_terms(care_need, [str(term) for term in terms]),
        source="openai",
    )


def web_resolve_india_location(location: str) -> dict[str, Any] | None:
    if os.getenv("ENABLE_WEB_RESOLUTION", "false").lower() not in {"1", "true", "yes", "on"}:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not location.strip():
        return None

    try:
        client = _openai_client(api_key)
    except Exception:
        return None

    prompt = (
        "Resolve this Indian place name for referral search. Return only JSON with keys: "
        "label, district, state, latitude, longitude, confidence, evidence_urls. "
        "Use a district or city centroid when an exact facility location is not requested. "
        "latitude and longitude must be decimal numbers within India. "
        f"Place: {location}"
    )

    response = client.responses.create(
        model=_model_for("OPENAI_SEARCH_MODEL"),
        tools=[{"type": "web_search", "search_context_size": "low"}],
        input=[
            {
                "role": "developer",
                "content": (
                    "You are a careful India geography resolver. Prefer official or encyclopedic sources. "
                    "Do not return coordinates unless they are plausible for India."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    try:
        payload = _extract_json(getattr(response, "output_text", "") or "")
        latitude = float(payload.get("latitude"))
        longitude = float(payload.get("longitude"))
    except Exception:
        return None

    if not (6 <= latitude <= 38 and 68 <= longitude <= 98):
        return None

    urls = payload.get("evidence_urls") or []
    if not isinstance(urls, list):
        urls = []

    return {
        "label": str(payload.get("label") or location),
        "district": str(payload.get("district") or ""),
        "state": str(payload.get("state") or ""),
        "latitude": latitude,
        "longitude": longitude,
        "confidence": str(payload.get("confidence") or "unknown"),
        "evidence_urls": [str(url) for url in urls if url],
    }


def spell_check_location(location_text: str) -> str | None:
    """
    Ask OpenAI to correct a misspelled Indian city / district / state name.

    Returns the corrected place name if a misspelling is detected, or None if
    the input already looks correct or the API is unavailable.

    Uses the same OPENAI_API_KEY and OPENAI_MODEL env vars as the rest of the app,
    both sourced from Databricks secrets via app.yaml.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not location_text.strip():
        return None

    try:
        client = _openai_client(api_key)
    except Exception:
        return None

    system = (
        "You are an India geography spell-checker. "
        "You know all Indian cities, districts, states, towns, and pin codes. "
        "Only correct obvious misspellings of real Indian place names. "
        "Never change a name that is already correct or that you are not confident about."
    )
    user_msg = (
        f'The user typed this Indian place name: "{location_text.strip()}"\n\n'
        "If it is misspelled, reply with ONLY the correctly spelled name "
        "(nothing else - no punctuation, no explanation, no quotes).\n"
        "If it is already correct, or if you are not sure, reply with exactly: CORRECT"
    )

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_SPELL_MODEL", DEFAULT_SPELL_MODEL),
            input=[
                {"role": "developer", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        result = (getattr(response, "output_text", "") or "").strip()
        # Treat empty, "CORRECT", or unchanged input as no-suggestion
        if not result or result.upper() == "CORRECT":
            return None
        if result.lower() == location_text.strip().lower():
            return None
        return result
    except Exception:
        return None


def parse_referral_query(raw_query: str) -> ParsedReferralQuery:
    fallback = _fallback_parse(raw_query)
    try:
        parsed = _parse_with_openai(raw_query, fallback)
        return parsed or fallback
    except Exception as exc:
        fallback.notes = f"OpenAI parsing failed; used fallback parsing. {exc}"
        return fallback


def _facility_for_prompt(item: dict[str, Any]) -> dict[str, Any]:
    evidence_terms: list[str] = []
    for evidence in item.get("evidence", [])[:4]:
        evidence_terms.extend(str(term) for term in evidence.get("terms", [])[:5])

    return {
        "name": item.get("name"),
        "distance_km": item.get("distance_km"),
        "score": item.get("score"),
        "base_score": item.get("base_score"),
        "public_score_delta": item.get("public_score_delta"),
        "public_signal": item.get("public_signal"),
        "facility_type": item.get("facility_type"),
        "operator_type": item.get("operator_type"),
        "city_state": item.get("city_state"),
        "phone": item.get("phone"),
        "email": item.get("email"),
        "website": item.get("website"),
        "source_urls": item.get("source_urls", [])[:3],
        "matching_terms": list(dict.fromkeys(evidence_terms))[:12],
        "missing_or_suspicious": item.get("missing_or_suspicious", [])[:8],
    }


def _public_signal_facility(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "name": item.get("name"),
        "city_state": item.get("city_state"),
        "facility_type": item.get("facility_type"),
        "phone": item.get("phone"),
        "website": item.get("website"),
        "source_urls": item.get("source_urls", [])[:2],
        "distance_km": item.get("distance_km"),
        "base_score": item.get("score"),
        "raw_score": item.get("raw_score"),
    }


def _clean_public_signal(payload: dict[str, Any]) -> dict[str, Any]:
    rating = payload.get("google_rating")
    try:
        rating = float(rating) if rating not in (None, "") else None
    except (TypeError, ValueError):
        rating = None
    if rating is not None and not (0 <= rating <= 5):
        rating = None

    review_count = payload.get("google_review_count")
    try:
        review_count = int(float(review_count)) if review_count not in (None, "") else None
    except (TypeError, ValueError):
        review_count = None

    themes = payload.get("review_themes") or []
    if not isinstance(themes, list):
        themes = []

    source_urls = payload.get("source_urls") or []
    if not isinstance(source_urls, list):
        source_urls = []

    notes = payload.get("notes") or ""
    if isinstance(notes, list):
        notes = "; ".join(str(item) for item in notes[:3])

    return {
        "candidate_id": str(payload.get("candidate_id") or ""),
        "google_rating": rating,
        "google_review_count": review_count,
        "rating_source": str(payload.get("rating_source") or "Google Maps"),
        "rating_url": str(payload.get("rating_url") or payload.get("google_maps_url") or ""),
        "review_themes": [str(item).strip() for item in themes[:5] if str(item).strip()][:5],
        "confidence": str(payload.get("confidence") or "unknown").lower(),
        "source_urls": [str(url) for url in source_urls[:4] if url],
        "notes": str(notes).strip(),
    }


def _public_rating_delta(signal: dict[str, Any]) -> float:
    rating = signal.get("google_rating")
    count = signal.get("google_review_count") or 0
    if rating is None:
        return 0.0

    # Public reputation is a meaningful 0-10 modifier, but still not a replacement for referral evidence.
    confidence = min(1.0, max(0.35, count / 250 if count else 0.45))
    delta = ((float(rating) - 3.5) / 1.5) * 2.0 * confidence
    return round(max(-1.5, min(2.0, delta)), 2)


def enrich_candidate_public_signals(
    candidates: list[dict[str, Any]],
    care_need: str,
    location_label: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Fetch public rating/review signals for the already-ranked shortlist in one web-search call.

    These signals are used as a small score modifier and shown separately from
    facility-record evidence. They are not treated as clinical evidence.
    """
    if os.getenv("ENABLE_PUBLIC_REVIEW_ENRICHMENT", "true").lower() not in {"1", "true", "yes", "on"}:
        return candidates, None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not candidates:
        return candidates, None

    try:
        limit = max(1, min(10, int(os.getenv("PUBLIC_REVIEW_ENRICHMENT_LIMIT", "8"))))
    except ValueError:
        limit = 8

    facilities = [_public_signal_facility(item) for item in candidates[:limit]]
    if not facilities:
        return candidates, None

    try:
        client = _openai_client(api_key)
    except Exception:
        return candidates, "Public rating lookup skipped: OpenAI SDK unavailable."

    system = (
        "You enrich a healthcare referral shortlist with public Google reputation signals. "
        "Use web search to look up each listed Indian facility. The google_rating field must be the overall Google Maps / Google Business Profile rating, not an inferred rating and not another site's rating. "
        "If the overall Google rating is not confidently visible, set google_rating and google_review_count to null and confidence to not_found. "
        "Return only compact JSON. Do not include medical advice. Do not quote long reviews; summarize review themes in your own words."
    )
    prompt = (
        "Care need: "
        f"{care_need or 'unspecified'}\n"
        "Search location: "
        f"{location_label or 'unspecified'}\n\n"
        "Candidate facilities JSON:\n"
        f"{json.dumps(facilities, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON exactly in this shape:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "candidate_id": "same candidate_id",\n'
        '      "google_rating": 4.2,\n'
        '      "google_review_count": 123,\n'
        '      "rating_source": "Google Maps",\n'
        '      "rating_url": "Google Maps or Google Business Profile URL if available",\n'
        '      "review_themes": ["short paraphrased theme", "short paraphrased theme"],\n'
        '      "confidence": "high|medium|low|not_found",\n'
        '      "source_urls": ["supporting URL"],\n'
        '      "notes": "short uncertainty note"\n'
        "    }\n"
        "  ],\n"
        '  "note": "overall lookup note"\n'
        "}\n"
        "If a facility cannot be confidently matched to a Google listing, set confidence to not_found and leave Google rating fields null. Do not substitute ratings from Practo, Justdial, Facebook, hospital websites, or other sources into google_rating."
    )

    try:
        response = client.responses.create(
            model=_model_for("OPENAI_REVIEW_MODEL", _model_for("OPENAI_SEARCH_MODEL")),
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice="auto",
            input=[
                {"role": "developer", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        payload = _extract_json(getattr(response, "output_text", "") or "")
    except Exception as exc:
        return candidates, f"Public rating lookup skipped: {exc}"

    items = payload.get("items") or []
    if not isinstance(items, list):
        return candidates, "Public rating lookup returned no structured items."

    signals = {
        signal["candidate_id"]: signal
        for signal in (_clean_public_signal(item) for item in items if isinstance(item, dict))
        if signal.get("candidate_id")
    }

    enriched = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        signal = signals.get(candidate_id)
        if not signal:
            enriched.append(candidate)
            continue
        base_score = float(candidate.get("score") or 0)
        delta = _public_rating_delta(signal)
        updated = {
            **candidate,
            "base_score": round(base_score, 1),
            "public_signal": signal,
            "public_score_delta": delta,
            "score": round(max(0.0, min(10.0, base_score + delta)), 1),
        }
        enriched.append(updated)

    enriched.sort(key=lambda item: (-float(item.get("score") or 0), item.get("distance_km", 10**9), item.get("name")))
    note = payload.get("note") or "Public ratings checked with one LLM web-search call."
    return enriched, str(note)


def ask_shortlist_copilot(question: str, shortlist: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Answer coordinator questions about the saved shortlist.

    Uses the OpenAI Responses API with the built-in web_search tool when the
    model decides current web context is useful. If web search is unavailable
    for the configured model/account, it retries without tools so the chat still
    works from shortlist evidence.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "answer": "OpenAI is not configured. Check that the app secret resource named `secret` is attached as OPENAI_API_KEY.",
            "used_search": False,
            "error": "missing_api_key",
        }
    if not question.strip():
        return {"answer": "Ask me a question about the saved facilities.", "used_search": False, "error": None}
    if not shortlist:
        return {"answer": "Save at least one facility first, then I can compare and investigate the shortlist.", "used_search": False, "error": None}

    try:
        client = _openai_client(api_key)
    except Exception:
        return {
            "answer": "The OpenAI SDK is not installed in this app environment.",
            "used_search": False,
            "error": "missing_openai_sdk",
        }

    facilities = [_facility_for_prompt(item) for item in shortlist[:8]]
    system = (
        "You are Referral Copilot for a care coordinator in India. "
        "Use the saved shortlist evidence first. Use web search only when the user asks for current, external, or missing details such as websites, phone verification, public reputation, hours, directions, or recent information. "
        "Do not provide diagnosis or treatment instructions. Be explicit about uncertainty and tell the user what must be verified before referral."
    )
    prompt = (
        "Saved facility shortlist JSON:\n"
        f"{json.dumps(facilities, ensure_ascii=False, indent=2)}\n\n"
        "Coordinator question:\n"
        f"{question.strip()}\n\n"
        "Answer with concise bullets. Include a short recommendation when the evidence supports it. "
        "If web search was useful, mention what you verified from the web; otherwise say you used the saved evidence."
    )

    model = _model_for("OPENAI_CHAT_MODEL")

    try:
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice="auto",
            input=[
                {"role": "developer", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        used_search = any(getattr(item, "type", "") == "web_search_call" for item in getattr(response, "output", []))
        return {
            "answer": (getattr(response, "output_text", "") or "").strip() or "I could not produce an answer.",
            "used_search": used_search,
            "error": None,
        }
    except Exception as exc:
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return {
                "answer": (getattr(response, "output_text", "") or "").strip()
                + "\n\nNote: web search was not available for this request, so I used only the saved shortlist evidence.",
                "used_search": False,
                "error": str(exc),
            }
        except Exception as fallback_exc:
            return {
                "answer": f"OpenAI chat failed: {fallback_exc}",
                "used_search": False,
                "error": str(fallback_exc),
            }
