from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_SPELL_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 8.0
DEFAULT_PUBLIC_SIGNAL_CACHE_SECONDS = 6 * 60 * 60


_PUBLIC_SIGNAL_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]], str | None]] = {}
_TRANSLATION_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


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


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    lowered = text.lower()
    return "429" in lowered or "rate_limit" in lowered or "rate limit" in lowered


def _rate_limit_retry_delay_seconds(exc: Exception) -> float | None:
    if not _looks_like_rate_limit_error(exc):
        return None

    text = str(exc)
    try:
        max_wait = float(os.getenv("OPENAI_RATE_LIMIT_RETRY_SECONDS", "22"))
    except ValueError:
        max_wait = 22.0
    if max_wait <= 0:
        return None

    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", text, flags=re.IGNORECASE)
    requested_wait = float(match.group(1)) + 1.0 if match else 20.0
    return min(max_wait, requested_wait)


def _is_rate_limit_error(exc: Exception) -> bool:
    return _looks_like_rate_limit_error(exc)


def _friendly_openai_unavailable_message(exc: Exception, feature: str) -> str:
    if _is_rate_limit_error(exc):
        return f"{feature} is temporarily rate-limited by OpenAI; showing the dataset evidence for now. Try Search again in about 20 seconds."
    return f"{feature} is unavailable right now; showing the dataset evidence for now."


def _responses_create_with_retry(client: Any, **kwargs: Any) -> Any:
    try:
        return client.responses.create(**kwargs)
    except Exception as exc:
        delay = _rate_limit_retry_delay_seconds(exc)
        if delay is None:
            raise
        time.sleep(delay)
        return client.responses.create(**kwargs)


def _public_signal_cache_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("PUBLIC_REVIEW_CACHE_TTL_SECONDS", DEFAULT_PUBLIC_SIGNAL_CACHE_SECONDS)))
    except ValueError:
        return float(DEFAULT_PUBLIC_SIGNAL_CACHE_SECONDS)


def _public_signal_cache_key(care_need: str, location_label: str, facilities: list[dict[str, Any]]) -> str:
    fingerprint = {
        "care_need": re.sub(r"\s+", " ", (care_need or "").strip().lower()),
        "location": re.sub(r"\s+", " ", (location_label or "").strip().lower()),
        "facilities": [
            {
                "candidate_id": item.get("candidate_id"),
                "name": item.get("name"),
                "city_state": item.get("city_state"),
            }
            for item in facilities
        ],
    }
    payload = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached_public_signals(cache_key: str) -> tuple[dict[str, dict[str, Any]], str | None] | None:
    ttl = _public_signal_cache_ttl_seconds()
    if ttl <= 0:
        return None

    cached = _PUBLIC_SIGNAL_CACHE.get(cache_key)
    if not cached:
        return None

    cached_at, signals, note = cached
    if (time.time() - cached_at) > ttl:
        _PUBLIC_SIGNAL_CACHE.pop(cache_key, None)
        return None
    return signals, note


def _set_cached_public_signals(cache_key: str, signals: dict[str, dict[str, Any]], note: str | None) -> None:
    ttl = _public_signal_cache_ttl_seconds()
    if ttl <= 0:
        return

    try:
        max_entries = max(4, int(os.getenv("PUBLIC_REVIEW_CACHE_MAX_ENTRIES", "64")))
    except ValueError:
        max_entries = 64

    _PUBLIC_SIGNAL_CACHE[cache_key] = (time.time(), signals, note)
    while len(_PUBLIC_SIGNAL_CACHE) > max_entries:
        oldest_key = min(_PUBLIC_SIGNAL_CACHE, key=lambda key: _PUBLIC_SIGNAL_CACHE[key][0])
        _PUBLIC_SIGNAL_CACHE.pop(oldest_key, None)


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

    response = _responses_create_with_retry(
        client,
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

    try:
        response = _responses_create_with_retry(
            client,
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
    except Exception:
        return None

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
    except Exception:
        fallback.notes = "OpenAI parsing failed; used fallback parsing."
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


def _is_http_url(url: Any) -> bool:
    text = str(url or "").strip()
    return text.startswith("http://") or text.startswith("https://")


def _dedupe_urls(urls: list[Any]) -> list[str]:
    cleaned = []
    for url in urls:
        text = str(url or "").strip()
        if _is_http_url(text) and text not in cleaned:
            cleaned.append(text)
    return cleaned


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
        "rating_source": str(payload.get("rating_source") or "public web"),
        "rating_url": str(payload.get("rating_url") or ""),
        "official_website_url": str(payload.get("official_website_url") or ""),
        "review_themes": [str(item).strip() for item in themes[:5] if str(item).strip()][:5],
        "confidence": str(payload.get("confidence") or "unknown").lower(),
        "source_urls": _dedupe_urls(source_urls[:4]),
        "notes": str(notes).strip(),
    }


def _public_rating_delta(signal: dict[str, Any]) -> float:
    rating = signal.get("google_rating")
    count = signal.get("google_review_count") or 0
    if rating is None:
        return 0.0

    # Keep public reputation as a small 0-10 modifier, not a replacement for referral evidence.
    confidence = min(1.0, max(0.35, count / 250 if count else 0.45))
    delta = (float(rating) - 3.8) * 0.75 * confidence
    return round(max(-0.6, min(0.9, delta)), 2)


def _apply_public_signals(
    candidates: list[dict[str, Any]],
    signals: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        signal = signals.get(candidate_id)
        if not signal:
            enriched.append(candidate)
            continue

        base_score = max(0.0, min(10.0, float(candidate.get("score") or 0)))
        delta = _public_rating_delta(signal)
        official_website = signal.get("official_website_url") if _is_http_url(signal.get("official_website_url")) else ""
        merged_source_urls = _dedupe_urls(
            [signal.get("rating_url")]
            + list(candidate.get("source_urls") or [])
            + [signal.get("official_website_url")]
            + list(signal.get("source_urls") or [])
        )
        enriched.append(
            {
                **candidate,
                "base_score": round(base_score, 1),
                "public_signal": signal,
                "public_score_delta": delta,
                "score": round(max(0.0, min(10.0, base_score + delta)), 1),
                "website": official_website or candidate.get("website") or "",
                "source_urls": merged_source_urls,
            }
        )

    enriched.sort(key=lambda item: (-float(item.get("score") or 0), item.get("distance_km", 10**9), item.get("name")))
    return enriched


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

    cache_key = _public_signal_cache_key(care_need, location_label, facilities)
    cached = _get_cached_public_signals(cache_key)
    if cached:
        cached_signals, cached_note = cached
        return _apply_public_signals(candidates, cached_signals), (
            cached_note or "Public ratings reused from recent lookup."
        )

    try:
        client = _openai_client(api_key)
    except Exception:
        return candidates, "Public rating lookup skipped: OpenAI SDK unavailable."

    system = (
        "You enrich a healthcare referral shortlist with public web reputation signals and official website links. "
        "Use web search to look up each listed Indian facility. Prefer Google Maps / Google Business Profile rating signals when visible in search results; "
        "if not visible, use another clearly public rating source and mark the source. "
        "Also find the hospital's own official website URL when one is confidently available; do not use rating directories, social pages, or review pages as official_website_url. "
        "Return only compact JSON. Do not include medical advice. Do not quote long reviews; summarize themes in your own words."
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
        '      "rating_source": "Google Maps or another public source",\n'
        '      "rating_url": "best public URL if available",\n'
        '      "official_website_url": "hospital official website URL if available",\n'
        '      "review_themes": ["short paraphrased theme", "short paraphrased theme"],\n'
        '      "confidence": "high|medium|low|not_found",\n'
        '      "source_urls": ["supporting URL"],\n'
        '      "notes": "short uncertainty note"\n'
        "    }\n"
        "  ],\n"
        '  "note": "overall lookup note"\n'
        "}\n"
        "If a facility cannot be confidently matched, set confidence to not_found and leave rating fields null."
    )

    try:
        response = _responses_create_with_retry(
            client,
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
        return candidates, _friendly_openai_unavailable_message(exc, "Public rating lookup")

    items = payload.get("items") or []
    if not isinstance(items, list):
        return candidates, "Public rating lookup returned no structured items."

    signals = {
        signal["candidate_id"]: signal
        for signal in (_clean_public_signal(item) for item in items if isinstance(item, dict))
        if signal.get("candidate_id")
    }

    note = payload.get("note") or "Public ratings and official website links checked with one LLM web-search call."
    _set_cached_public_signals(cache_key, signals, str(note))
    return _apply_public_signals(candidates, signals), str(note)


def _translation_cache_key(language: str, parsed: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    payload = {
        "language": language,
        "care_need": parsed.get("care_need"),
        "terms": parsed.get("required_terms", [])[:10],
        "candidates": [
            {
                "candidate_id": item.get("candidate_id"),
                "facility_type": item.get("facility_type"),
                "operator_type": item.get("operator_type"),
                "evidence": [
                    {
                        "field": evidence.get("field"),
                        "terms": evidence.get("terms", [])[:6],
                        "snippet": evidence.get("snippet", ""),
                    }
                    for evidence in item.get("evidence", [])[:6]
                ],
                "missing_or_suspicious": item.get("missing_or_suspicious", [])[:8],
                "public_signal": {
                    "review_themes": (item.get("public_signal") or {}).get("review_themes", [])[:4],
                    "notes": (item.get("public_signal") or {}).get("notes", ""),
                },
            }
            for item in candidates[:25]
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _translation_cache_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("RESULT_TRANSLATION_CACHE_TTL_SECONDS", "21600")))
    except ValueError:
        return 21600.0


def _translation_payload(parsed: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parsed": {
            "care_need": parsed.get("care_need"),
            "required_terms": parsed.get("required_terms", [])[:10],
        },
        "candidates": [
            {
                "candidate_id": item.get("candidate_id"),
                "facility_type": item.get("facility_type"),
                "operator_type": item.get("operator_type"),
                "evidence": [
                    {
                        "field": evidence.get("field"),
                        "terms": evidence.get("terms", [])[:6],
                        "snippet": evidence.get("snippet", ""),
                    }
                    for evidence in item.get("evidence", [])[:6]
                ],
                "missing_or_suspicious": item.get("missing_or_suspicious", [])[:8],
                "public_signal": {
                    "review_themes": (item.get("public_signal") or {}).get("review_themes", [])[:4],
                    "notes": (item.get("public_signal") or {}).get("notes", ""),
                },
            }
            for item in candidates[:25]
        ],
    }


def _apply_translation_payload(
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
    translated: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated_candidates = copy.deepcopy(candidates)
    updated_parsed = copy.deepcopy(parsed)

    translated_parsed = translated.get("parsed") or {}
    if translated_parsed.get("care_need"):
        updated_parsed["care_need"] = str(translated_parsed["care_need"])
    if isinstance(translated_parsed.get("required_terms"), list):
        updated_parsed["required_terms"] = [str(term) for term in translated_parsed["required_terms"][:10] if str(term).strip()]

    by_id = {
        str(item.get("candidate_id") or ""): item
        for item in translated.get("candidates", [])
        if isinstance(item, dict)
    }

    for candidate in updated_candidates:
        patch = by_id.get(str(candidate.get("candidate_id") or ""))
        if not patch:
            continue

        if patch.get("facility_type"):
            candidate["facility_type"] = str(patch["facility_type"])
        if patch.get("operator_type"):
            candidate["operator_type"] = str(patch["operator_type"])

        evidence_patch = patch.get("evidence") or []
        if isinstance(evidence_patch, list):
            for idx, evidence in enumerate((candidate.get("evidence") or [])[: len(evidence_patch)]):
                translated_evidence = evidence_patch[idx] if isinstance(evidence_patch[idx], dict) else {}
                if isinstance(translated_evidence.get("terms"), list):
                    evidence["terms"] = [str(term) for term in translated_evidence["terms"][:6] if str(term).strip()]
                if translated_evidence.get("snippet"):
                    evidence["snippet"] = str(translated_evidence["snippet"])

        if isinstance(patch.get("missing_or_suspicious"), list):
            candidate["missing_or_suspicious"] = [
                str(item)
                for item in patch["missing_or_suspicious"][:8]
                if str(item).strip()
            ]

        signal = candidate.get("public_signal") or {}
        signal_patch = patch.get("public_signal") or {}
        if isinstance(signal_patch.get("review_themes"), list):
            signal["review_themes"] = [
                str(item)
                for item in signal_patch["review_themes"][:4]
                if str(item).strip()
            ]
        if signal_patch.get("notes"):
            signal["notes"] = str(signal_patch["notes"])
        if signal:
            candidate["public_signal"] = signal

    return updated_candidates, updated_parsed


def translate_referral_results(
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
    language: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    language = (language or "English").strip()
    if not language or language.lower() == "english" or not candidates:
        return candidates, parsed, None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return candidates, parsed, "Translation skipped: OpenAI is not configured."

    cache_key = _translation_cache_key(language, parsed, candidates)
    cached = _TRANSLATION_CACHE.get(cache_key)
    ttl = _translation_cache_ttl_seconds()
    if cached and ttl > 0 and (time.time() - cached[0]) <= ttl:
        cached_candidates, cached_parsed = _apply_translation_payload(candidates, parsed, cached[1])
        return cached_candidates, cached_parsed, None

    try:
        client = _openai_client(api_key)
    except Exception:
        return candidates, parsed, "Translation skipped: OpenAI SDK unavailable."

    system = (
        "You translate healthcare referral result text for Indian care coordinators. "
        "Translate only user-facing explanatory text into the requested language. "
        "Do not translate hospital/facility names, phone numbers, emails, URLs, pin codes, coordinates, IDs, or proper nouns. "
        "Keep medical terms accurate; transliterate only when that is more natural for the target language. "
        "Return compact JSON only."
    )
    payload = _translation_payload(parsed, candidates)
    prompt = (
        f"Target language: {language}\n\n"
        "Translate this JSON and return the same shape. Preserve candidate_id values exactly.\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        response = _responses_create_with_retry(
            client,
            model=_model_for("OPENAI_TRANSLATION_MODEL", DEFAULT_SPELL_MODEL),
            input=[
                {"role": "developer", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        translated = _extract_json(getattr(response, "output_text", "") or "")
        if ttl > 0:
            _TRANSLATION_CACHE[cache_key] = (time.time(), translated)
            while len(_TRANSLATION_CACHE) > 64:
                oldest_key = min(_TRANSLATION_CACHE, key=lambda key: _TRANSLATION_CACHE[key][0])
                _TRANSLATION_CACHE.pop(oldest_key, None)
        translated_candidates, translated_parsed = _apply_translation_payload(candidates, parsed, translated)
        return translated_candidates, translated_parsed, None
    except Exception as exc:
        return candidates, parsed, _friendly_openai_unavailable_message(exc, "Result translation")


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
        response = _responses_create_with_retry(
            client,
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
            response = _responses_create_with_retry(
                client,
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
            message = (
                "Copilot is temporarily rate-limited by OpenAI. Please try again in about 20 seconds."
                if _is_rate_limit_error(fallback_exc)
                else "Copilot could not reach OpenAI right now. Please try again shortly."
            )
            return {
                "answer": message,
                "used_search": False,
                "error": str(fallback_exc),
            }
