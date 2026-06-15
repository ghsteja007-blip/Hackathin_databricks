from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


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
        from openai import OpenAI
    except Exception:
        fallback.notes = "OpenAI SDK is not installed; used fallback parsing."
        return None

    system_prompt = (
        "Extract referral search intent from the user's text. "
        "Return only JSON with keys: care_need, location, urgency, required_terms. "
        "required_terms should be short facility-record match terms, not diagnosis advice. "
        "Do not invent facility names."
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
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
        from openai import OpenAI
    except Exception:
        return None

    prompt = (
        "Resolve this Indian place name for referral search. Return only JSON with keys: "
        "label, district, state, latitude, longitude, confidence, evidence_urls. "
        "Use a district or city centroid when an exact facility location is not requested. "
        "latitude and longitude must be decimal numbers within India. "
        f"Place: {location}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
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


def parse_referral_query(raw_query: str) -> ParsedReferralQuery:
    fallback = _fallback_parse(raw_query)
    try:
        parsed = _parse_with_openai(raw_query, fallback)
        return parsed or fallback
    except Exception as exc:
        fallback.notes = f"OpenAI parsing failed; used fallback parsing. {exc}"
        return fallback
