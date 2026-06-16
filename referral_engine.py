from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd

from openai_helpers import ParsedReferralQuery


INDIA_LAT_RANGE = (6.0, 38.0)
INDIA_LON_RANGE = (68.0, 98.0)

TEXT_FIELDS = [
    "name",
    "facilityTypeId",
    "operatorTypeId",
    "description",
    "specialties",
    "procedure",
    "equipment",
    "capability",
]

FIELD_WEIGHTS = {
    "specialties": 10,
    "procedure": 9,
    "equipment": 8,
    "capability": 8,
    "description": 5,
    "name": 3,
    "facilityTypeId": 3,
    "operatorTypeId": 1,
}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9/+.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_jsonish_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item not in (None, "")]
            if isinstance(parsed, dict):
                return [json.dumps(parsed)]
        except Exception:
            return [text]

    return [text]


def first_jsonish_value(value: Any) -> str:
    values = parse_jsonish_list(value)
    return values[0] if values else ""


def valid_india_coord(latitude: Any, longitude: Any) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except Exception:
        return False
    return INDIA_LAT_RANGE[0] <= lat <= INDIA_LAT_RANGE[1] and INDIA_LON_RANGE[0] <= lon <= INDIA_LON_RANGE[1]


def repair_swapped_coord(latitude: Any, longitude: Any) -> tuple[float | None, float | None, str | None]:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except Exception:
        return None, None, "coordinates missing"

    if valid_india_coord(lat, lon):
        return lat, lon, None

    if INDIA_LON_RANGE[0] <= lat <= INDIA_LON_RANGE[1] and INDIA_LAT_RANGE[0] <= lon <= INDIA_LAT_RANGE[1]:
        return lon, lat, "latitude/longitude appeared swapped"

    return lat, lon, "coordinates outside India bounds"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _mean_valid_coords(df: pd.DataFrame) -> tuple[float | None, float | None]:
    coords = []
    for _, row in df.iterrows():
        lat, lon, warning = repair_swapped_coord(row.get("latitude"), row.get("longitude"))
        if lat is not None and lon is not None and valid_india_coord(lat, lon):
            coords.append((lat, lon))
    if not coords:
        return None, None
    return sum(lat for lat, _ in coords) / len(coords), sum(lon for _, lon in coords) / len(coords)


def _contains_location(df: pd.DataFrame, location: str, columns: list[str]) -> pd.Series:
    needle = normalize_text(location)
    mask = pd.Series(False, index=df.index)
    if not needle:
        return mask
    pattern = rf"(?:^|\s){re.escape(needle)}(?:\s|$)"
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].map(normalize_text).str.contains(pattern, na=False, regex=True)
    return mask


def resolve_location(location: str, facilities: pd.DataFrame, pincodes: pd.DataFrame) -> dict[str, Any]:
    location = (location or "").strip()
    warnings: list[str] = []
    if not location:
        return {"label": "", "latitude": None, "longitude": None, "method": "missing", "match_count": 0, "warnings": []}

    pin_match = re.search(r"\b\d{6}\b", location)
    if pin_match and "pincode" in pincodes.columns:
        pin = pin_match.group(0)
        matches = pincodes[pincodes["pincode"].astype(str).str.extract(r"(\d{6})")[0] == pin]
        lat, lon = _mean_valid_coords(matches)
        if lat and lon:
            return {
                "label": pin,
                "latitude": lat,
                "longitude": lon,
                "method": "pincode",
                "match_count": int(len(matches)),
                "warnings": warnings,
            }
        warnings.append("pincode matched but had no valid coordinates")

    facility_columns = [
        "address_city",
        "address_stateOrRegion",
        "address_zipOrPostcode",
        "address_line1",
        "address_line2",
        "address_line3",
    ]
    facility_mask = _contains_location(facilities, location, facility_columns)
    facility_matches = facilities[facility_mask] if len(facilities) else pd.DataFrame()
    lat, lon = _mean_valid_coords(facility_matches)
    if lat and lon:
        return {
            "label": location,
            "latitude": lat,
            "longitude": lon,
            "method": "facility address centroid",
            "match_count": int(len(facility_matches)),
            "warnings": warnings,
        }

    pin_mask = _contains_location(pincodes, location, ["officename", "district", "statename", "regionname"])
    pin_matches = pincodes[pin_mask] if len(pincodes) else pd.DataFrame()
    lat, lon = _mean_valid_coords(pin_matches)
    if lat and lon:
        return {
            "label": location,
            "latitude": lat,
            "longitude": lon,
            "method": "post office directory",
            "match_count": int(len(pin_matches)),
            "warnings": warnings,
        }

    try:
        from openai_helpers import web_resolve_india_location

        web_location = web_resolve_india_location(location)
    except Exception as exc:
        web_location = None
        warnings.append(f"web location resolver failed: {exc}")

    if web_location:
        source_note = "web evidence: " + ", ".join(web_location.get("evidence_urls", [])[:2])
        return {
            "label": web_location.get("label") or location,
            "latitude": web_location.get("latitude"),
            "longitude": web_location.get("longitude"),
            "method": f"OpenAI web search ({web_location.get('confidence', 'unknown')} confidence)",
            "match_count": 1,
            "warnings": warnings + ([source_note] if web_location.get("evidence_urls") else []),
        }

    return {
        "label": location,
        "latitude": None,
        "longitude": None,
        "method": "unresolved",
        "match_count": 0,
        "warnings": warnings + ["no matching pincode, district, city, or facility address"],
    }


def _field_text(row: pd.Series, field: str) -> str:
    if field in {"specialties", "procedure", "equipment", "capability", "source_urls", "websites", "phone_numbers"}:
        return " ".join(parse_jsonish_list(row.get(field)))
    return str(row.get(field) or "")


def _field_snippet(text: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def _matching_evidence(row: pd.Series, terms: list[str]) -> tuple[list[dict[str, Any]], float]:
    evidence: list[dict[str, Any]] = []
    score = 0.0

    normalized_terms = [normalize_text(term) for term in terms if normalize_text(term)]
    for field in TEXT_FIELDS:
        raw_text = _field_text(row, field)
        normalized = normalize_text(raw_text)
        matched_terms = []

        for term in normalized_terms:
            if term and re.search(rf"\b{re.escape(term)}\b", normalized):
                matched_terms.append(term)

        if matched_terms:
            unique_terms = sorted(set(matched_terms))
            evidence.append(
                {
                    "field": field,
                    "terms": unique_terms,
                    "snippet": _field_snippet(raw_text),
                }
            )
            score += FIELD_WEIGHTS.get(field, 1) * len(unique_terms)

    return evidence, score


def _missing_or_suspicious(row: pd.Series, evidence: list[dict[str, Any]], parsed_need: ParsedReferralQuery) -> list[str]:
    items: list[str] = []

    if not evidence:
        items.append("no direct need match in facility text")
    if not row.get("officialPhone") and not row.get("phone_numbers"):
        items.append("phone missing")
    if not row.get("officialWebsite") and not row.get("websites"):
        items.append("website missing")
    if not row.get("source_urls"):
        items.append("source URLs missing")
    if not row.get("operatorTypeId") or str(row.get("operatorTypeId")).lower() == "nan":
        items.append("operator type missing")
    if not row.get("specialties"):
        items.append("specialties missing")

    need_text = normalize_text(parsed_need.care_need)
    combined = normalize_text(" ".join(_field_text(row, field) for field in TEXT_FIELDS))
    if "dialysis" in need_text and "dialysis" not in combined and "nephrology" not in combined:
        items.append("dialysis evidence absent")
    if "emergency" in need_text and not any(term in combined for term in ["emergency", "24/7", "24 hrs", "ambulance", "icu", "trauma"]):
        items.append("emergency readiness unclear")
    if "surgery" in need_text and not any(term in combined for term in ["surgery", "surgical", "operation", "operating theater"]):
        items.append("surgery evidence unclear")

    return items


def _city_state(row: pd.Series) -> str:
    bits = []
    for col in ["address_city", "address_stateOrRegion", "address_zipOrPostcode"]:
        value = row.get(col)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            text = str(value).strip()
            if text and text.lower() != "nan":
                bits.append(text)
    return ", ".join(bits)


def _distance_score(distance_km: float, radius_km: float) -> float:
    if distance_km <= 5:
        return 70
    return max(5, 70 * (1 - (distance_km / max(radius_km, 1))))


def _score_0_to_10(raw_score: float) -> float:
    return round(max(0.0, min(10.0, raw_score / 12.0)), 1)


def _source_urls(row: pd.Series) -> list[str]:
    urls = []
    urls.extend(parse_jsonish_list(row.get("source_urls")))
    urls.extend(parse_jsonish_list(row.get("websites")))
    cleaned = []
    for url in urls:
        url = str(url).strip()
        if url and url.lower() != "none" and url not in cleaned:
            cleaned.append(url)
    return cleaned[:5]


def rank_facilities(
    facilities: pd.DataFrame,
    location: dict[str, Any],
    parsed_need: ParsedReferralQuery,
    radius_km: float = 250,
    limit: int = 8,
) -> list[dict[str, Any]]:
    origin_lat = float(location["latitude"])
    origin_lon = float(location["longitude"])
    terms = parsed_need.required_terms or [parsed_need.care_need]
    candidates: list[dict[str, Any]] = []

    for row_index, row in facilities.iterrows():
        name = row.get("name")
        if not name or str(name).lower() == "nan":
            continue

        lat, lon, coord_warning = repair_swapped_coord(row.get("latitude"), row.get("longitude"))
        if lat is None or lon is None or not valid_india_coord(lat, lon):
            continue

        distance_km = haversine_km(origin_lat, origin_lon, lat, lon)
        if distance_km > radius_km:
            continue

        evidence, evidence_score = _matching_evidence(row, terms)
        suspicious = _missing_or_suspicious(row, evidence, parsed_need)
        if coord_warning:
            suspicious.append(coord_warning)

        facility_type = str(row.get("facilityTypeId") or "").lower()
        facility_bonus = 0
        if "hospital" in facility_type:
            facility_bonus += 8
        if "clinic" in facility_type:
            facility_bonus += 3

        emergency_bonus = 0
        if parsed_need.urgency == "emergency":
            combined = normalize_text(" ".join(_field_text(row, field) for field in TEXT_FIELDS))
            emergency_bonus += 10 if any(term in combined for term in ["emergency", "24/7", "24 hrs", "icu", "trauma"]) else -10

        missing_penalty = min(18, len(suspicious) * 2.5)
        raw_score = evidence_score + _distance_score(distance_km, radius_km) + facility_bonus + emergency_bonus - missing_penalty

        candidate_id = str(row.get("unique_id") or f"{row_index}-{name}")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "name": str(name),
                "score": _score_0_to_10(raw_score),
                "raw_score": round(raw_score, 2),
                "distance_km": round(distance_km, 1),
                "facility_type": str(row.get("facilityTypeId") or "facility"),
                "operator_type": str(row.get("operatorTypeId") or "unknown"),
                "city_state": _city_state(row),
                "phone": str(row.get("officialPhone") or first_jsonish_value(row.get("phone_numbers")) or ""),
                "email": str(row.get("email") or ""),
                "website": str(row.get("officialWebsite") or first_jsonish_value(row.get("websites")) or ""),
                "latitude": lat,
                "longitude": lon,
                "evidence": evidence,
                "missing_or_suspicious": suspicious,
                "source_urls": _source_urls(row),
            }
        )

    candidates.sort(key=lambda item: (-item["raw_score"], item["distance_km"], item["name"]))
    return candidates[:limit]
