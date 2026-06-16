import json
import math
import os
import re
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
import plotly.graph_objects as go

try:
    import dash_leaflet as dl
except Exception:
    dl = None

from data_access import load_datasets
from openai_helpers import (
    ask_shortlist_copilot,
    enrich_candidate_public_signals,
    parse_referral_query,
    spell_check_location,
)
from referral_engine import rank_facilities, resolve_location


APP_TITLE = "Referral Copilot"
MEDICAL_DISPLAY_TERMS = [
    "dialysis",
    "hemodialysis",
    "renal failure treatment",
    "kidney transplant",
    "nephrology",
    "urology",
    "internal medicine",
    "emergency services",
    "trauma care",
    "icu",
    "intensive care",
    "ct scan",
    "echocardiography",
    "pathology laboratory",
    "dialysis machines",
    "cardiology",
    "cardiothoracic surgery",
    "gastroenterology",
    "medical oncology",
    "neurosurgery",
    "pediatric surgery",
    "pediatrics",
    "neonatology",
    "perinatal medicine",
    "pediatric critical care",
    "pediatric cardiology",
    "pediatric gastroenterology",
    "gynecology and obstetrics",
    "orthopedic surgery",
    "ophthalmology",
    "otolaryngology",
    "plastic surgery",
    "cosmetic dentistry",
    "preventive health checkup",
    "video laryngoscopy",
]
NOISY_EVIDENCE_MARKERS = [
    "hexahealth",
    "listed as",
    "top hospital",
    "best hospital",
    "book appointment",
    "public feedback",
    "rating",
    "reviews",
    "near me",
]
CHAT_PROMPTS = [
    {
        "id": "compare",
        "label": "Compare saved",
        "prompt": "Compare the saved facilities and recommend the strongest referral option. Call out tradeoffs and what to verify.",
    },
    {
        "id": "verify",
        "label": "Verify contacts",
        "prompt": "Use web search if needed to check current contact details, website, and public information for the saved facilities.",
    },
    {
        "id": "risks",
        "label": "Evidence gaps",
        "prompt": "What evidence is missing or suspicious across the saved shortlist, and what should a coordinator verify before referral?",
    },
    {
        "id": "next",
        "label": "Next steps",
        "prompt": "Create a concise coordinator handoff plan for the saved shortlist, including calls to make and questions to ask.",
    },
]

app = Dash(__name__, title=APP_TITLE, suppress_callback_exceptions=True)
server = app.server


@server.get("/health")
def health_check():
    return {"status": "ok", "app": APP_TITLE}


# Location extraction helper

def _extract_location_hint(query: str) -> str:
    """
    Pull the location word(s) that follow a preposition in a free-text query.
    Returns at most the first two words after the preposition so incidental
    words like 'hospital' or 'clinic' don't pollute the spell-check input.
    """
    m = re.search(r"\b(?:near|around|in|at)\s+(\S+(?:\s+\S+)?)", query, re.IGNORECASE)
    if m:
        words = m.group(1).strip().split()[:2]
        return " ".join(words)
    # Fallback: last word of the query
    words = query.strip().split()
    return words[-1] if words else ""


# Reusable UI helpers

def chip(text: str, class_name: str = "chip") -> html.Span:
    return html.Span(text, className=class_name)


def render_empty_state() -> html.Div:
    return html.Div(
        className="empty-state",
        children=[
            html.H2("Start a referral search"),
            html.P(
                "Pick a search mode on the left, describe what you need, and click Search."
            ),
        ],
    )


def _humanize_evidence_text(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[_|;/]+", ", ", cleaned)
    cleaned = re.sub(r"\b([A-Za-z][A-Za-z0-9+.-]*)(?:\s+\1\b){1,}", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned


def _evidence_is_noisy(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in NOISY_EVIDENCE_MARKERS)


def _display_label(text: str) -> str:
    words = text.split()
    short_words = {"icu", "ct", "ivf", "iui"}
    return " ".join(word.upper() if word.lower() in short_words else word.capitalize() for word in words)


def _clean_evidence_snippet(field: str, snippet: str, terms: list[str]) -> str:
    text = _humanize_evidence_text(snippet)
    if not text:
        return ""

    lowered = text.lower()
    concepts: list[str] = []
    for term in terms:
        normalized = _humanize_evidence_text(term).lower()
        if normalized and normalized not in concepts:
            concepts.append(normalized)

    for term in MEDICAL_DISPLAY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered, flags=re.IGNORECASE) and term not in concepts:
            concepts.append(term)

    if field in {"specialties", "procedure", "equipment", "capability"} and concepts:
        return "Record mentions: " + ", ".join(_display_label(term) for term in concepts[:10]) + "."

    if _evidence_is_noisy(text):
        return "Record matched this field, but the source text looks directory-style; verify directly."

    sentences = [part.strip(" ,;") for part in re.split(r"(?<=[.!?])\s+|\s{2,}", text) if part.strip(" ,;")]
    filtered = [part for part in sentences if not _evidence_is_noisy(part)]
    clean = " ".join(filtered or sentences or [text])
    return clean[:177].rstrip(" ,.;") + "..." if len(clean) > 180 else clean


def render_evidence(items: list[dict[str, Any]]) -> list[html.Div]:
    if not items:
        return [html.Div("No direct matching evidence found in the facility record.", className="muted")]

    rendered = []
    for item in items[:6]:
        field = item.get("field", "evidence")
        label = field.replace("_", " ").title()
        terms_list = [_display_label(_humanize_evidence_text(term).lower()) for term in item.get("terms", [])[:5]]
        terms = ", ".join(dict.fromkeys(terms_list))
        snippet = _clean_evidence_snippet(field, item.get("snippet") or "", item.get("terms", []))
        rendered.append(
            html.Div(
                className="evidence-row",
                children=[
                    html.Span(label, className="evidence-field"),
                    html.Span(terms, className="evidence-terms"),
                    html.P(snippet, className="evidence-snippet") if snippet else None,
                ],
            )
        )
    return rendered


def _rating_text(signal: dict[str, Any] | None) -> str:
    if not signal:
        return ""
    rating = signal.get("google_rating")
    count = signal.get("google_review_count")
    if rating is None:
        return ""
    count_text = f" ({int(count):,} reviews)" if count else ""
    source = signal.get("rating_source") or "public rating"
    return f"{source}: {float(rating):.1f}/5{count_text}"


def _is_http_url(url: Any) -> bool:
    text = str(url or "").strip()
    return text.startswith("http://") or text.startswith("https://")


def _first_http_url(values: list[Any] | None) -> str:
    for value in values or []:
        text = str(value or "").strip()
        if _is_http_url(text):
            return text
    return ""


def render_public_signal(candidate: dict[str, Any]) -> html.Div:
    signal = candidate.get("public_signal") or {}
    if not signal:
        return html.Div(
            className="public-signal public-signal-empty",
            children=[
                html.Div("Public review signal", className="section-label"),
                chip("not checked or unavailable", "chip chip-soft"),
            ],
        )

    rating = signal.get("google_rating")
    count = signal.get("google_review_count")
    confidence = signal.get("confidence") or "unknown"
    themes = signal.get("review_themes") or []
    delta = candidate.get("public_score_delta")
    url = signal.get("rating_url") if _is_http_url(signal.get("rating_url")) else _first_http_url(signal.get("source_urls"))

    chips = []
    if rating is not None:
        chips.append(chip(f"rating {float(rating):.1f}/5", "chip chip-ok"))
    if count:
        chips.append(chip(f"{int(count):,} public reviews", "chip chip-soft"))
    if delta not in (None, ""):
        sign = "+" if float(delta) >= 0 else ""
        chips.append(chip(f"{sign}{float(delta):.1f}/10 adjustment", "chip chip-soft"))
    chips.append(chip(f"confidence: {confidence}", "chip chip-soft"))

    return html.Div(
        className="public-signal",
        children=[
            html.Div("Public review signal", className="section-label"),
            html.Div(className="public-signal-chips", children=chips),
            html.Div(
                className="public-themes",
                children=[chip(theme, "chip chip-soft") for theme in themes[:4]]
                or [chip(signal.get("notes") or "No review themes found", "chip chip-warning")],
            ),
            html.A(
                "Open public source",
                href=url,
                target="_blank",
                rel="noreferrer",
                className="chip chip-link public-source-link",
            )
            if url
            else None,
        ],
    )


def _env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_points(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for candidate in candidates:
        lat = _safe_float(candidate.get("latitude"))
        lon = _safe_float(candidate.get("longitude"))
        if lat is None or lon is None:
            continue
        if not (6 <= lat <= 38 and 68 <= lon <= 98):
            continue
        points.append({**candidate, "_lat": lat, "_lon": lon})
    return points


def _score_color(score: Any) -> str:
    value = _score_value(score)
    if value >= 8:
        return "#0f766e"
    if value >= 6.5:
        return "#2563eb"
    if value >= 4.5:
        return "#ca8a04"
    return "#64748b"


def _score_radius(score: Any) -> int:
    value = _score_value(score)
    return int(max(8, min(22, 8 + value * 1.2)))


def _score_value(score: Any) -> float:
    value = _safe_float(score)
    if value is None:
        return 0.0
    return max(0.0, min(10.0, value))


def _score_text(score: Any) -> str:
    value = _safe_float(score)
    return "n/a" if value is None else f"{_score_value(value):.1f}/10"


def _map_bounds(lats: list[float], lons: list[float]) -> list[list[float]]:
    if not lats or not lons:
        return [[6.5, 68.0], [37.5, 97.5]]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_pad = max(0.15, (max_lat - min_lat) * 0.22)
    lon_pad = max(0.15, (max_lon - min_lon) * 0.22)
    return [[min_lat - lat_pad, min_lon - lon_pad], [max_lat + lat_pad, max_lon + lon_pad]]


def _leaflet_zoom(span: float) -> int:
    if span < 0.25:
        return 12
    if span < 0.5:
        return 11
    if span < 1.0:
        return 10
    if span < 2.5:
        return 9
    if span < 5:
        return 8
    if span < 10:
        return 7
    if span < 20:
        return 6
    return 5


def render_plotly_candidate_map(location: dict[str, Any], candidates: list[dict[str, Any]]) -> dcc.Graph:
    lats = [candidate.get("latitude") for candidate in candidates]
    lons = [candidate.get("longitude") for candidate in candidates]
    names = [candidate.get("name") for candidate in candidates]
    scores = [_score_value(candidate.get("score")) for candidate in candidates]
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]

    hover_text = []
    for candidate in candidates:
        evidence_terms = []
        for item in candidate.get("evidence", [])[:3]:
            evidence_terms.extend(item.get("terms", [])[:3])
        evidence_label = ", ".join(dict.fromkeys(evidence_terms)) or "No direct evidence"
        rating_label = _rating_text(candidate.get("public_signal"))
        base_score = _safe_float(candidate.get("base_score"))
        hover_text.append(
            "<br>".join(
                [line for line in [
                    f"<b>{candidate.get('name') or 'Unnamed facility'}</b>",
                    f"Distance: {candidate.get('distance_km', 0):.1f} km",
                    f"Score: {_score_text(candidate.get('score'))}",
                    f"Base score: {_score_text(base_score)}" if base_score is not None else "",
                    rating_label,
                    f"Type: {candidate.get('facility_type') or 'facility'}",
                    f"Evidence: {evidence_label}",
                ] if line]
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            text=names,
            hovertext=hover_text,
            hoverinfo="text",
            marker={
                "size": [max(10, min(28, 10 + score * 1.7)) for score in scores],
                "color": scores,
                "cmin": 0,
                "cmax": 10,
                "colorscale": [[0, "#94a3b8"], [0.45, "#f59e0b"], [1, "#0f766e"]],
                "line": {"width": 1, "color": "#ffffff"},
                "colorbar": {"title": "Score /10", "thickness": 12},
            },
            name="Candidate facilities",
            customdata=candidate_ids,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[location.get("latitude")],
            lon=[location.get("longitude")],
            mode="markers+text",
            text=["Search origin"],
            textposition="bottom center",
            hovertext=[f"<b>{location.get('label') or 'Search origin'}</b><br>{location.get('method') or ''}"],
            hoverinfo="text",
            marker={"size": 16, "color": "#b91c1c", "symbol": "star", "line": {"width": 1, "color": "#ffffff"}},
            name="Search origin",
        )
    )

    all_lats = [float(value) for value in lats + [location.get("latitude")] if value is not None]
    all_lons = [float(value) for value in lons + [location.get("longitude")] if value is not None]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    span = max(max(all_lats) - min(all_lats), max(all_lons) - min(all_lons), 1.0)
    projection_scale = max(2.5, min(18, 15 / span))

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        geo={
            "scope": "asia",
            "center": {"lat": center_lat, "lon": center_lon},
            "projection": {"type": "mercator", "scale": projection_scale},
            "showland": True,
            "landcolor": "#f8fafc",
            "showocean": True,
            "oceancolor": "#e0f2fe",
            "showlakes": True,
            "lakecolor": "#e0f2fe",
            "showcountries": True,
            "countrycolor": "#cbd5e1",
            "showsubunits": True,
            "subunitcolor": "#e2e8f0",
            "fitbounds": "locations",
        },
    )

    return dcc.Graph(
        id="candidate-map",
        figure=fig,
        config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
        className="candidate-map",
    )


def render_leaflet_candidate_map(location: dict[str, Any], candidates: list[dict[str, Any]]) -> Any:
    if dl is None:
        return render_plotly_candidate_map(location, candidates)

    points = _candidate_points(candidates)
    origin_lat = _safe_float(location.get("latitude"))
    origin_lon = _safe_float(location.get("longitude"))
    if origin_lat is None or origin_lon is None:
        origin_lat = points[0]["_lat"] if points else 22.9734
        origin_lon = points[0]["_lon"] if points else 78.6569

    lats = [point["_lat"] for point in points] + [origin_lat]
    lons = [point["_lon"] for point in points] + [origin_lon]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.05)
    bounds = _map_bounds(lats, lons)

    tile_url = os.getenv("LEAFLET_TILE_URL", "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")
    tile_layer = dl.TileLayer(
        url=tile_url,
        attribution="&copy; OpenStreetMap contributors",
        maxZoom=18,
    )

    origin_marker = dl.Marker(
        id="leaflet-origin-marker",
        position=[origin_lat, origin_lon],
        title=location.get("label") or "Search origin",
        children=[
            dl.Tooltip(location.get("label") or "Search origin", sticky=True),
            dl.Popup(
                html.Div(
                    className="leaflet-popup-content",
                    children=[
                        html.Strong(location.get("label") or "Search origin"),
                        html.Div(location.get("method") or "Resolved search origin", className="muted"),
                    ],
                )
            ),
        ],
    )

    line_layers = []
    marker_layers = []
    for idx, candidate in enumerate(points, start=1):
        candidate_id = candidate.get("candidate_id") or str(idx)
        name = candidate.get("name") or "Unnamed facility"
        distance = candidate.get("distance_km", 0)
        score = candidate.get("score", 0)
        evidence_terms = []
        for item in candidate.get("evidence", [])[:3]:
            evidence_terms.extend(item.get("terms", [])[:3])
        evidence_label = ", ".join(dict.fromkeys(evidence_terms)) or "No direct evidence"
        rating_label = _rating_text(candidate.get("public_signal"))
        color = _score_color(score)

        line_layers.append(
            dl.Polyline(
                positions=[[origin_lat, origin_lon], [candidate["_lat"], candidate["_lon"]]],
                color=color,
                weight=2,
                opacity=0.35,
                interactive=False,
            )
        )
        marker_layers.append(
            dl.CircleMarker(
                id={"type": "leaflet-facility-marker", "index": candidate_id},
                center=[candidate["_lat"], candidate["_lon"]],
                radius=_score_radius(score),
                color="#ffffff",
                weight=2,
                fillColor=color,
                fillOpacity=0.92,
                children=[
                    dl.Tooltip(
                        str(idx),
                        permanent=True,
                        direction="center",
                        opacity=1,
                        className="leaflet-marker-number",
                    ),
                    dl.Popup(
                        html.Div(
                            className="leaflet-popup-content",
                            children=[
                                html.Div(f"#{idx}", className="rank leaflet-rank"),
                                html.Strong(name),
                                html.Div(f"{float(distance or 0):.1f} km away - score {_score_text(score)}"),
                                html.Div(rating_label, className="leaflet-rating") if rating_label else None,
                                html.Div(candidate.get("facility_type") or "facility", className="muted"),
                                html.Div(f"Evidence: {evidence_label}", className="leaflet-evidence"),
                            ],
                        )
                    ),
                ],
            )
        )

    children = [
        tile_layer,
        dl.LayerGroup(line_layers),
        dl.LayerGroup([origin_marker]),
        dl.LayerGroup(marker_layers),
        dl.ScaleControl(position="bottomleft"),
    ]

    return dl.Map(
        id="candidate-map",
        children=children,
        center=center,
        zoom=_leaflet_zoom(span),
        bounds=bounds,
        scrollWheelZoom=True,
        className="candidate-map leaflet-candidate-map",
        style={"height": "520px", "width": "100%"},
    )


def render_candidate_map(location: dict[str, Any], candidates: list[dict[str, Any]]) -> Any:
    if _env_enabled("ENABLE_LEAFLET_MAP", "true"):
        return render_leaflet_candidate_map(location, candidates)
    return render_plotly_candidate_map(location, candidates)


def render_map_selection(candidate: dict[str, Any] | None = None) -> html.Div:
    if not candidate:
        return html.Div("Click a facility marker to inspect details.", className="muted")

    evidence_terms: list[str] = []
    for item in candidate.get("evidence", [])[:3]:
        evidence_terms.extend(item.get("terms", [])[:4])
    evidence_terms = list(dict.fromkeys(evidence_terms))
    rating_label = _rating_text(candidate.get("public_signal"))

    return html.Div(
        className="map-selection-card",
        children=[
            html.Div("Selected facility", className="section-label"),
            html.H3(candidate.get("name") or "Unnamed facility"),
            html.Div(
                className="candidate-meta",
                children=[
                    chip(f"{candidate.get('distance_km', 0):.1f} km"),
                    chip(f"score {_score_text(candidate.get('score'))}"),
                    chip(rating_label, "chip chip-ok") if rating_label else None,
                    chip(candidate.get("facility_type") or "facility"),
                ],
            ),
            html.Div(
                className="warning-list",
                children=[chip(term, "chip chip-soft") for term in evidence_terms[:6]]
                or [chip("No direct evidence terms", "chip chip-warning")],
            ),
        ],
    )


def render_candidate(candidate: dict[str, Any], rank: int) -> html.Article:
    suspicious = candidate.get("missing_or_suspicious") or []
    evidence = candidate.get("evidence") or []
    signal = candidate.get("public_signal") or {}
    rating_url = signal.get("rating_url") if _is_http_url(signal.get("rating_url")) else ""
    source_url = rating_url or _first_http_url(candidate.get("source_urls") or [])
    website_url = str(candidate.get("website") or "").strip()
    source_label = "Public source" if rating_url else "Source"
    has_public_signal = bool(signal)
    base_score = _safe_float(candidate.get("base_score"))
    contact_bits = []

    if candidate.get("phone"):
        contact_bits.append(chip(candidate["phone"], "chip chip-soft"))
    if candidate.get("email"):
        contact_bits.append(chip(candidate["email"], "chip chip-soft"))
    if _is_http_url(website_url):
        contact_bits.append(
            html.A("Website", href=website_url, target="_blank", rel="noreferrer", className="chip chip-link")
        )
    else:
        contact_bits.append(chip("Website unavailable", "chip chip-soft chip-disabled"))
    if source_url:
        contact_bits.append(
            html.A(source_label, href=source_url, target="_blank", rel="noreferrer", className="chip chip-link")
        )

    return html.Article(
        className="candidate-card",
        children=[
            html.Div(
                className="candidate-topline",
                children=[
                    html.Div([html.Div(f"#{rank}", className="rank"), html.H3(candidate.get("name") or "Unnamed facility")]),
                    html.Div(
                        className="score-stack",
                        children=[
                            html.Strong(f"{_score_value(candidate.get('score')):.1f}"),
                            html.Span("/10 adjusted" if has_public_signal else "/10 score"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="candidate-meta",
                children=[
                    chip(f"{candidate.get('distance_km', 0):.1f} km"),
                    chip(f"base {_score_text(base_score)}", "chip chip-soft") if base_score is not None else None,
                    chip(_rating_text(candidate.get("public_signal")), "chip chip-ok")
                    if _rating_text(candidate.get("public_signal"))
                    else None,
                    chip(candidate.get("facility_type") or "facility"),
                    chip(candidate.get("operator_type") or "operator unknown"),
                    chip(candidate.get("city_state") or "location unknown"),
                ],
            ),
            html.Div(className="section-label", children="Matching evidence"),
            html.Div(className="evidence-list", children=render_evidence(evidence)),
            render_public_signal(candidate) if candidate.get("public_signal") else None,
            html.Div(className="section-label warning-label", children="Missing or suspicious evidence"),
            html.Div(
                className="warning-list",
                children=[chip(item, "chip chip-warning") for item in suspicious[:8]]
                or [chip("No obvious evidence gaps detected", "chip chip-ok")],
            ),
            html.Div(className="candidate-contact", children=contact_bits),
            html.Button(
                "Save",
                id={"type": "save-candidate", "index": candidate["candidate_id"]},
                className="secondary-button",
            ),
        ],
    )


def render_results(
    parsed: dict[str, Any],
    location: dict[str, Any],
    candidates: list[dict[str, Any]],
    data_notes: list[str],
) -> html.Div:
    if not candidates:
        return html.Div(
            className="empty-state",
            children=[
                html.H2("No candidates found"),
                html.P("Try a wider radius or a more general care need."),
            ],
        )

    parsed_terms = parsed.get("required_terms") or []
    location_note = (
        f"{location.get('label')} via {location.get('method')} ({location.get('match_count', 0)} matches)"
    )
    nearest = min(candidates, key=lambda candidate: candidate.get("distance_km", 10**9))
    highest = max(candidates, key=lambda candidate: candidate.get("score", -10**9))

    return html.Div(
        className="results-shell",
        children=[
            html.Div(
                className="search-summary",
                children=[
                    html.Div([html.Span("Need", className="summary-label"), html.Strong(parsed.get("care_need") or "unknown")]),
                    html.Div([html.Span("Location", className="summary-label"), html.Strong(location_note)]),
                    html.Div([html.Span("Terms", className="summary-label"), html.Strong(", ".join(parsed_terms[:8]) or "none")]),
                ],
            ),
            html.Div(
                className="data-notes",
                children=[chip(note, "chip chip-soft") for note in data_notes[:4] + location.get("warnings", [])[:4]],
            ),
            html.Div(
                className="map-panel",
                children=[
                    html.Div(
                        className="map-header",
                        children=[
                            html.H2("Referral Map"),
                            html.Span("Leaflet view: drag, zoom, open popups, and click markers for evidence."),
                        ],
                    ),
                    render_candidate_map(location, candidates),
                    html.Div(
                        className="map-inspector",
                        children=[
                            html.Div(
                                className="map-stats",
                                children=[
                                    chip(f"{len(candidates)} mapped facilities", "chip chip-soft"),
                                    chip(
                                        f"nearest: {nearest.get('name')} ({nearest.get('distance_km', 0):.1f} km)",
                                        "chip chip-soft",
                                    ),
                                    chip(
                                        f"top score: {highest.get('name')} ({_score_text(highest.get('score'))})",
                                        "chip chip-soft",
                                    ),
                                ],
                            ),
                            html.Div(id="map-selection-panel", children=render_map_selection(candidates[0])),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="candidate-list",
                children=[render_candidate(candidate, idx + 1) for idx, candidate in enumerate(candidates)],
            ),
        ],
    )


def render_shortlist(shortlist: list[dict[str, Any]] | None) -> html.Div:
    shortlist = shortlist or []
    if not shortlist:
        return html.Div("No saved facilities yet.", className="muted")

    def item_summary(item: dict[str, Any]) -> str:
        base_score = _safe_float(item.get("base_score"))
        parts = [
            f"{item.get('distance_km', 0):.1f} km",
            f"score {_score_text(item.get('score'))}",
            f"base {_score_text(base_score)}" if base_score is not None else "",
            _rating_text(item.get("public_signal")),
        ]
        return " - ".join(part for part in parts if part)

    return html.Div(
        className="shortlist-items",
        children=[
            html.Div(
                className="shortlist-item",
                children=[
                    html.Strong(item.get("name") or "Unnamed facility"),
                    html.Span(item_summary(item)),
                ],
            )
            for item in shortlist
        ],
    )


def render_chat_history(history: list[dict[str, Any]] | None) -> html.Div:
    history = history or []
    if not history:
        return html.Div(
            className="chat-empty",
            children=[
                html.Strong("Shortlist copilot"),
                html.Span("Save hospitals, then ask about tradeoffs, gaps, contact checks, or what to verify next."),
            ],
        )

    return html.Div(
        className="chat-thread-inner",
        children=[
            html.Div(
                className=f"chat-message chat-message-{message.get('role', 'assistant')}",
                children=[
                    html.Div(
                        className="chat-meta",
                        children=[
                            html.Span("You" if message.get("role") == "user" else "Copilot"),
                            chip("web checked", "chip chip-ok") if message.get("used_search") else None,
                        ],
                    ),
                    dcc.Markdown(message.get("content") or "", className="chat-markdown", link_target="_blank"),
                ],
            )
            for message in history[-8:]
        ],
    )


# Layout
app.layout = html.Div(
    id="app-root",
    className="app-shell",
    children=[
        dcc.Store(id="candidate-store", data=[]),
        dcc.Store(id="shortlist-store", data=[]),
        dcc.Store(id="geolocation-store", data=None),
        dcc.Store(id="search-mode-store", data="freetext"),
        dcc.Store(id="spell-suggestion-store", data=None),
        dcc.Store(id="theme-store", data="light"),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Download(id="shortlist-download"),
        html.Div(id="chat-scroll-anchor", style={"display": "none"}),
        html.Header(
            className="app-header",
            children=[
                html.Div([html.H1(APP_TITLE), html.P("Evidence-attached care facility shortlists.")]),
                html.Div(
                    className="header-actions",
                    children=[
                        html.Button("Dark", id="theme-toggle", className="theme-toggle", n_clicks=0),
                        html.Div(className="status-pill", children=os.getenv("DATABRICKS_SCHEMA", "local data")),
                    ],
                ),
            ],
        ),
        html.Main(
            className="workspace",
            children=[
                # Left sidebar: search panel
                html.Section(
                    className="search-panel",
                    children=[
                        # Panel heading
                        html.Div(
                            className="panel-heading",
                            children=[
                                html.H2("Find a facility"),
                                html.P("Choose a search mode, describe what you need, then hit Search."),
                            ],
                        ),
                        # Mode tab strip
                        html.Div(
                            className="mode-tabs",
                            children=[
                                html.Button(
                                    [html.Span("Free text")],
                                    id="mode-tab-freetext",
                                    className="mode-tab mode-tab--active",
                                    n_clicks=0,
                                ),
                                html.Button(
                                    [html.Span("Pin code")],
                                    id="mode-tab-pincode",
                                    className="mode-tab",
                                    n_clicks=0,
                                ),
                                html.Button(
                                    [html.Span("My location")],
                                    id="mode-tab-location",
                                    className="mode-tab",
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        # Free-text panel
                        html.Div(
                            id="panel-freetext",
                            className="mode-panel",
                            children=[
                                html.Label(
                                    "Care need + location",
                                    htmlFor="query-input",
                                    className="input-label",
                                ),
                                html.P(
                                    'e.g. "dialysis near Jaipur" or "emergency surgery near 380007"',
                                    className="input-hint",
                                ),
                                dcc.Input(
                                    id="query-input",
                                    type="text",
                                    value="dialysis near Jaipur",
                                    placeholder="dialysis near Jaipur",
                                    debounce=True,
                                    className="query-input",
                                ),
                            ],
                        ),
                        # Pin code panel
                        html.Div(
                            id="panel-pincode",
                            className="mode-panel mode-panel--hidden",
                            children=[
                                html.Label(
                                    "Care need",
                                    htmlFor="care-need-pincode",
                                    className="input-label",
                                ),
                                dcc.Input(
                                    id="care-need-pincode",
                                    type="text",
                                    placeholder="e.g. dialysis, emergency surgery",
                                    debounce=True,
                                    className="query-input",
                                ),
                                html.Label(
                                    "Pin code",
                                    htmlFor="pincode-input",
                                    className="input-label",
                                    style={"marginTop": "14px"},
                                ),
                                html.P("Your 6-digit postal pin code", className="input-hint"),
                                dcc.Input(
                                    id="pincode-input",
                                    type="text",
                                    placeholder="302001",
                                    maxLength=6,
                                    debounce=True,
                                    className="pincode-input",
                                ),
                            ],
                        ),
                        # My-location panel
                        html.Div(
                            id="panel-location",
                            className="mode-panel mode-panel--hidden",
                            children=[
                                html.Label(
                                    "Care need",
                                    htmlFor="care-need-location",
                                    className="input-label",
                                ),
                                dcc.Input(
                                    id="care-need-location",
                                    type="text",
                                    placeholder="e.g. dialysis, emergency surgery",
                                    debounce=True,
                                    className="query-input",
                                ),
                                html.Div(
                                    className="locate-row",
                                    children=[
                                        html.Button(
                                            "Use My Location",
                                            id="locate-button",
                                            className="locate-button",
                                            title="Use your device's GPS to set the search origin",
                                        ),
                                        html.Span(id="gps-status", className="gps-status"),
                                    ],
                                ),
                            ],
                        ),
                        # Spell-correction banner
                        html.Div(
                            id="spell-banner",
                            className="spell-banner spell-banner--hidden",
                            children=[
                                html.Span(id="spell-banner-text", className="spell-banner-text"),
                                html.Button(
                                    "Use this",
                                    id="apply-suggestion-btn",
                                    className="spell-apply-btn",
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        # Radius / results controls
                        html.Div(
                            className="controls-row",
                            children=[
                                html.Div([
                                    html.Label("Search radius km", htmlFor="radius-input"),
                                    dcc.Dropdown(
                                        id="radius-input",
                                        value=250,
                                        options=[
                                            {"label": "25 km", "value": 25},
                                            {"label": "50 km", "value": 50},
                                            {"label": "100 km", "value": 100},
                                            {"label": "250 km", "value": 250},
                                            {"label": "500 km", "value": 500},
                                            {"label": "1000 km", "value": 1000},
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        className="select-input",
                                    ),
                                ]),
                                html.Div([
                                    html.Label("Max results", htmlFor="limit-input"),
                                    dcc.Dropdown(
                                        id="limit-input",
                                        value=8,
                                        options=[
                                            {"label": "5", "value": 5},
                                            {"label": "8", "value": 8},
                                            {"label": "10", "value": 10},
                                            {"label": "15", "value": 15},
                                            {"label": "20", "value": 20},
                                            {"label": "25", "value": 25},
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        className="select-input",
                                    ),
                                ]),
                            ],
                        ),
                        html.Button("Search", id="search-button", className="primary-button"),
                        dcc.Loading(
                            id="search-status-loading",
                            type="circle",
                            color="#0f766e",
                            className="search-loading-shell",
                            children=html.Div(id="search-status", className="search-status"),
                        ),
                    ],
                ),
                # Center: results panel
                dcc.Loading(
                    id="results-loading",
                    type="circle",
                    color="#0f766e",
                    className="results-loading-shell",
                    children=html.Section(id="results-panel", className="results-panel", children=render_empty_state()),
                ),
                # Right sidebar: shortlist
                html.Aside(
                    className="shortlist-panel",
                    children=[
                        html.Div(
                            className="shortlist-header",
                            children=[
                                html.H2("Shortlist"),
                                html.Button("Clear", id="clear-shortlist", className="ghost-button"),
                            ],
                        ),
                        html.Div(id="shortlist-panel-body", className="shortlist-panel-body", children=render_shortlist([])),
                        html.Button("Download CSV", id="download-shortlist", className="secondary-button full-width"),
                        html.Div(
                            className="chat-panel",
                            children=[
                                html.Div(
                                    className="chat-heading",
                                    children=[
                                        html.H2("Ask Copilot"),
                                        html.P("Saved facilities become context. Web search runs when current details matter."),
                                    ],
                                ),
                                html.Div(
                                    className="chat-presets",
                                    children=[
                                        html.Button(
                                            item["label"],
                                            id={"type": "chat-preset", "index": item["id"]},
                                            className="chat-preset",
                                            n_clicks=0,
                                        )
                                        for item in CHAT_PROMPTS
                                    ],
                                ),
                                html.Div(
                                    className="chat-thread-shell",
                                    children=dcc.Loading(
                                        id="chat-loading",
                                        type="dot",
                                        color="#0f766e",
                                        children=html.Div(
                                            id="chat-history",
                                            className="chat-thread",
                                            children=render_chat_history([]),
                                        ),
                                    ),
                                ),
                                html.Div(
                                    className="copilot-composer",
                                    children=[
                                        dcc.Textarea(
                                            id="chat-input",
                                            className="chat-input",
                                            placeholder="Compare my shortlist, check evidence gaps, or look up current details...",
                                            value="",
                                        ),
                                        html.Div(
                                            className="copilot-composer-row",
                                            children=[
                                                html.Div(id="chat-status", className="chat-status"),
                                                html.Button(
                                                    "Ask Copilot",
                                                    id="chat-submit",
                                                    className="primary-button chat-submit",
                                                    n_clicks=0,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Footer(
            className="app-footer",
            children="Referral support only. Verify availability, clinical fit, insurance, and emergency status before sending a patient.",
        ),
    ],
)


app.clientside_callback(
    """
    function(children) {
        window.setTimeout(function() {
            const thread = document.getElementById("chat-history");
            if (thread) {
                thread.scrollTop = thread.scrollHeight;
            }
        }, 50);
        return "";
    }
    """,
    Output("chat-scroll-anchor", "children"),
    Input("chat-history", "children"),
)


# Tab switching and theme callbacks
@app.callback(
    Output("theme-store", "data"),
    Output("app-root", "className"),
    Output("theme-toggle", "children"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
)
def toggle_theme(n_clicks: int | None, current_theme: str | None):
    if not n_clicks:
        theme = current_theme or "light"
    else:
        theme = "dark" if (current_theme or "light") == "light" else "light"
    return theme, f"app-shell theme-{theme}", "Light" if theme == "dark" else "Dark"


app.clientside_callback(
    """
    function(ft_clicks, pin_clicks, loc_clicks) {
        const ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        const id = ctx.triggered[0].prop_id.split('.')[0];
        let mode = 'freetext';
        if (id === 'mode-tab-pincode')  mode = 'pincode';
        if (id === 'mode-tab-location') mode = 'location';

        const base = 'mode-panel';
        const hide = base + ' mode-panel--hidden';
        return [
            mode,
            mode === 'freetext'  ? base : hide,
            mode === 'pincode'   ? base : hide,
            mode === 'location'  ? base : hide,
            mode === 'freetext'  ? 'mode-tab mode-tab--active' : 'mode-tab',
            mode === 'pincode'   ? 'mode-tab mode-tab--active' : 'mode-tab',
            mode === 'location'  ? 'mode-tab mode-tab--active' : 'mode-tab',
        ];
    }
    """,
    Output("search-mode-store", "data"),
    Output("panel-freetext", "className"),
    Output("panel-pincode", "className"),
    Output("panel-location", "className"),
    Output("mode-tab-freetext", "className"),
    Output("mode-tab-pincode", "className"),
    Output("mode-tab-location", "className"),
    Input("mode-tab-freetext", "n_clicks"),
    Input("mode-tab-pincode", "n_clicks"),
    Input("mode-tab-location", "n_clicks"),
    prevent_initial_call=True,
)


# Spell-correction banner
@app.callback(
    Output("spell-banner", "className"),
    Output("spell-banner-text", "children"),
    Output("spell-suggestion-store", "data"),
    Input("query-input", "value"),
    State("search-mode-store", "data"),
    prevent_initial_call=True,
)
def check_spelling(query: str, mode: str):
    hidden = "spell-banner spell-banner--hidden"
    # Only active in free-text mode
    if mode not in (None, "freetext") or not query:
        return hidden, "", None
    location_text = _extract_location_hint(query)
    if not location_text or len(location_text) < 3:
        return hidden, "", None
    # Delegate to OpenAI with the same API key used throughout the app.
    suggestion = spell_check_location(location_text)
    if not suggestion:
        return hidden, "", None
    return (
        "spell-banner",
        f'Did you mean "{suggestion}"?',
        {"suggestion": suggestion, "original": location_text},
    )


# Apply spell suggestion into the query input
app.clientside_callback(
    """
    function(n_clicks, data, current_value) {
        if (!n_clicks || !data || !data.suggestion || !data.original) {
            return window.dash_clientside.no_update;
        }
        if (!current_value) return window.dash_clientside.no_update;
        // Escape special regex chars in the original typo before replacing
        var escaped = data.original.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        var re = new RegExp(escaped, 'gi');
        return current_value.replace(re, data.suggestion);
    }
    """,
    Output("query-input", "value"),
    Input("apply-suggestion-btn", "n_clicks"),
    State("spell-suggestion-store", "data"),
    State("query-input", "value"),
    prevent_initial_call=True,
)


# GPS: fetch coordinates
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        if (!navigator.geolocation) {
            return {error: 'Geolocation is not supported by this browser.'};
        }
        return new Promise(function(resolve) {
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    resolve({
                        latitude:  pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy:  pos.coords.accuracy
                    });
                },
                function(err) { resolve({error: err.message}); },
                {timeout: 10000, maximumAge: 60000, enableHighAccuracy: true}
            );
        });
    }
    """,
    Output("geolocation-store", "data"),
    Input("locate-button", "n_clicks"),
    prevent_initial_call=True,
)


# GPS: update status label
app.clientside_callback(
    """
    function(data) {
        if (!data) return '';
        if (data.error) return 'Location error: ' + data.error;
        var acc = data.accuracy ? ' (+/-' + Math.round(data.accuracy) + 'm)' : '';
        return 'Location acquired' + acc;
    }
    """,
    Output("gps-status", "children"),
    Input("geolocation-store", "data"),
    prevent_initial_call=True,
)


# Main search
@app.callback(
    Output("results-panel", "children"),
    Output("candidate-store", "data"),
    Output("search-status", "children"),
    Input("search-button", "n_clicks"),
    State("search-mode-store", "data"),
    State("query-input", "value"),
    State("care-need-pincode", "value"),
    State("pincode-input", "value"),
    State("care-need-location", "value"),
    State("radius-input", "value"),
    State("limit-input", "value"),
    State("geolocation-store", "data"),
    prevent_initial_call=True,
)
def run_search(
    n_clicks: int,
    mode: str,
    query: str,
    care_need_pin: str,
    pincode: str,
    care_need_loc: str,
    radius_km: int,
    limit: int,
    geolocation_data: dict | None,
):
    mode = mode or "freetext"

    try:
        datasets, data_notes = load_datasets()
        correction_note = ""
        try:
            radius_value = float(radius_km if radius_km not in (None, "") else 250)
        except (TypeError, ValueError):
            radius_value = 250.0
        radius_value = max(1.0, min(1000.0, radius_value))

        try:
            limit_value = int(limit if limit not in (None, "") else 8)
        except (TypeError, ValueError):
            limit_value = 8
        limit_value = max(1, min(25, limit_value))

        # Resolve parsed need and location per mode.
        if mode == "freetext":
            raw_query = (query or "").strip()
            if not raw_query:
                return render_empty_state(), [], "Enter a care need and location."

            location_hint = _extract_location_hint(raw_query)
            if location_hint and len(location_hint) >= 3:
                suggestion = spell_check_location(location_hint)
                if suggestion:
                    corrected_query = re.sub(re.escape(location_hint), suggestion, raw_query, count=1, flags=re.IGNORECASE)
                    if corrected_query != raw_query:
                        raw_query = corrected_query
                        correction_note = f'Autocorrected location to "{suggestion}".'
                        data_notes = data_notes + [f"LLM location autocorrection applied: {suggestion}"]

            parsed = parse_referral_query(raw_query)
            location = resolve_location(parsed.location, datasets.facilities, datasets.pincodes)
            if (not correction_note) and (not location.get("latitude") or not location.get("longitude")) and parsed.location:
                suggestion = spell_check_location(parsed.location)
                if suggestion:
                    corrected_query = re.sub(re.escape(parsed.location), suggestion, raw_query, flags=re.IGNORECASE)
                    corrected_parsed = parse_referral_query(corrected_query)
                    corrected_location = resolve_location(
                        corrected_parsed.location,
                        datasets.facilities,
                        datasets.pincodes,
                    )
                    if corrected_location.get("latitude") and corrected_location.get("longitude"):
                        parsed = corrected_parsed
                        location = corrected_location
                        correction_note = f'Autocorrected location to "{suggestion}".'
                        data_notes = data_notes + [f"LLM location autocorrection applied: {suggestion}"]

        elif mode == "pincode":
            care_need = (care_need_pin or "").strip()
            pin = re.sub(r"\D", "", (pincode or ""))
            if not care_need:
                return render_empty_state(), [], "Enter a care need."
            if len(pin) != 6:
                return (
                    html.Div(
                        className="empty-state",
                        children=[
                            html.H2("Invalid pin code"),
                            html.P("Please enter a valid 6-digit pin code, e.g. 302001."),
                        ],
                    ),
                    [],
                    "Enter a 6-digit pin code.",
                )
            parsed = parse_referral_query(f"{care_need} near {pin}")
            location = resolve_location(pin, datasets.facilities, datasets.pincodes)

        elif mode == "location":
            care_need = (care_need_loc or "").strip()
            if not care_need:
                return render_empty_state(), [], "Enter a care need."
            gps = geolocation_data or {}
            if not gps.get("latitude") or gps.get("error"):
                return (
                    html.Div(
                        className="empty-state",
                        children=[
                            html.H2("No location yet"),
                            html.P("Click 'Use My Location' and allow the browser to access your GPS."),
                        ],
                    ),
                    [],
                    "Tap 'Use My Location' first.",
                )
            parsed = parse_referral_query(care_need)
            accuracy = gps.get("accuracy")
            acc_note = f"+/-{int(accuracy)}m" if accuracy else ""
            location = {
                "label": f"Your location {acc_note}".strip(),
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
                "method": f"device GPS{' ' + acc_note if acc_note else ''}",
                "match_count": 1,
                "warnings": [],
            }

        else:
            return render_empty_state(), [], "Unknown search mode."

        # Guard: location must resolve.
        if not location.get("latitude") or not location.get("longitude"):
            return (
                html.Div(
                    className="empty-state",
                    children=[
                        html.H2("Location not resolved"),
                        html.P(
                            "Use a city, district, state, or 6-digit pin code present in the loaded data."
                        ),
                    ],
                ),
                [],
                "Location could not be resolved from the current datasets.",
            )

        candidates = rank_facilities(
            datasets.facilities,
            location=location,
            parsed_need=parsed,
            radius_km=radius_value,
            limit=limit_value,
        )
        if candidates:
            candidates, public_note = enrich_candidate_public_signals(
                candidates,
                care_need=parsed.care_need,
                location_label=location.get("label") or parsed.location,
            )
            if public_note and re.search(r"\b(skipped|unavailable|rate-limited|returned no)\b", public_note, re.IGNORECASE):
                data_notes = data_notes + [f"Public review signal: {public_note}"]

        parsed_dict = parsed.to_dict()
        return (
            render_results(parsed_dict, location, candidates, data_notes),
            candidates,
            (
                f"Found {len(candidates)} candidates within {radius_value:g} km "
                f"(showing up to {limit_value}) using {parsed.source} parsing. {correction_note}"
            ).strip(),
        )

    except Exception as exc:
        return (
            html.Div(
                className="empty-state error-state",
                children=[html.H2("Search failed"), html.P(str(exc))],
            ),
            [],
            "Search failed.",
        )


@app.callback(
    Output("map-selection-panel", "children"),
    Input("candidate-map", "clickData"),
    Input({"type": "leaflet-facility-marker", "index": ALL}, "n_clicks"),
    State("candidate-store", "data"),
    prevent_initial_call=True,
)
def update_map_selection(click_data, marker_clicks, candidates):
    candidates = candidates or []

    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "leaflet-facility-marker":
        candidate_id = triggered.get("index")
        candidate = next((item for item in candidates if item.get("candidate_id") == candidate_id), None)
        return render_map_selection(candidate or (candidates[0] if candidates else None))

    if not click_data or not click_data.get("points"):
        return render_map_selection(candidates[0] if candidates else None)

    candidate_id = click_data["points"][0].get("customdata")
    candidate = next((item for item in candidates if item.get("candidate_id") == candidate_id), None)
    return render_map_selection(candidate or (candidates[0] if candidates else None))


@app.callback(
    Output("shortlist-store", "data"),
    Output("shortlist-panel-body", "children"),
    Input({"type": "save-candidate", "index": ALL}, "n_clicks"),
    Input("clear-shortlist", "n_clicks"),
    State("candidate-store", "data"),
    State("shortlist-store", "data"),
    prevent_initial_call=True,
)
def update_shortlist(save_clicks, clear_clicks, candidates, shortlist):
    triggered = ctx.triggered_id

    if triggered == "clear-shortlist":
        return [], render_shortlist([])

    if not isinstance(triggered, dict) or triggered.get("type") != "save-candidate":
        return no_update, no_update

    shortlist = shortlist or []
    candidates = candidates or []
    candidate_id = triggered.get("index")
    candidate = next((item for item in candidates if item.get("candidate_id") == candidate_id), None)
    if not candidate:
        return no_update, no_update

    if not any(item.get("candidate_id") == candidate_id for item in shortlist):
        shortlist = shortlist + [candidate]

    return shortlist, render_shortlist(shortlist)


@app.callback(
    Output("chat-input", "value", allow_duplicate=True),
    Input({"type": "chat-preset", "index": ALL}, "n_clicks"),
    State("chat-input", "value"),
    prevent_initial_call=True,
)
def apply_chat_preset(preset_clicks, current_value):
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "chat-preset":
        return no_update

    prompt_by_id = {item["id"]: item["prompt"] for item in CHAT_PROMPTS}
    return prompt_by_id.get(triggered.get("index"), current_value or "")


@app.callback(
    Output("chat-history-store", "data"),
    Output("chat-history", "children"),
    Output("chat-input", "value"),
    Output("chat-status", "children"),
    Input("chat-submit", "n_clicks"),
    State("chat-input", "value"),
    State("shortlist-store", "data"),
    State("chat-history-store", "data"),
    prevent_initial_call=True,
)
def ask_copilot(n_clicks, question, shortlist, history):
    question = (question or "").strip()
    history = history or []
    shortlist = shortlist or []

    if not question:
        return no_update, no_update, no_update, "Ask a question about the saved shortlist."

    user_message = {"role": "user", "content": question}
    result = ask_shortlist_copilot(question, shortlist)
    answer = result.get("answer") or "I could not produce an answer."
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "used_search": bool(result.get("used_search")),
    }
    new_history = (history + [user_message, assistant_message])[-10:]
    status = "Answered with web search." if result.get("used_search") else "Answered from shortlist evidence."
    if result.get("error") and result.get("error") not in {"missing_api_key", "missing_openai_sdk"}:
        status = "Answered without web search fallback."

    return new_history, render_chat_history(new_history), "", status


@app.callback(
    Output("shortlist-download", "data"),
    Input("download-shortlist", "n_clicks"),
    State("shortlist-store", "data"),
    prevent_initial_call=True,
)
def download_shortlist(n_clicks, shortlist):
    shortlist = shortlist or []
    rows = []
    for item in shortlist:
        signal = item.get("public_signal") or {}
        rows.append(
            {
                "name": item.get("name"),
                "distance_km": item.get("distance_km"),
                "score": item.get("score"),
                "base_score": item.get("base_score"),
                "public_score_delta": item.get("public_score_delta"),
                "google_rating": signal.get("google_rating"),
                "google_review_count": signal.get("google_review_count"),
                "rating_source": signal.get("rating_source"),
                "rating_url": signal.get("rating_url"),
                "official_website_url": signal.get("official_website_url"),
                "public_signal": json.dumps(item.get("public_signal", {}), ensure_ascii=False),
                "facility_type": item.get("facility_type"),
                "operator_type": item.get("operator_type"),
                "city_state": item.get("city_state"),
                "phone": item.get("phone"),
                "email": item.get("email"),
                "website": item.get("website"),
                "source_urls": json.dumps(item.get("source_urls", []), ensure_ascii=False),
                "evidence": json.dumps(item.get("evidence", []), ensure_ascii=False),
                "missing_or_suspicious": json.dumps(item.get("missing_or_suspicious", []), ensure_ascii=False),
            }
        )
    return dcc.send_data_frame(
        __import__("pandas").DataFrame(rows).to_csv, "referral_shortlist.csv", index=False
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT") or os.getenv("DATABRICKS_APP_PORT") or "8050")
    app.run_server(host="0.0.0.0", port=port, debug=os.getenv("DASH_DEBUG", "false").lower() == "true")
