from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_SPELL_MODEL = "gpt-4o-mini"


def _model_for(env_name: str, fallback: str = DEFAULT_OPENAI_MODEL) -> str:
    return os.getenv(env_name) or os.getenv("OPENAI_MODEL") or fallback


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
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
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
        from openai import OpenAI
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

    client = OpenAI(api_key=api_key)
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
