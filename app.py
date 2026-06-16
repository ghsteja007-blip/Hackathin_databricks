import json
import os
import re
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
import plotly.graph_objects as go

from data_access import load_datasets
from openai_helpers import parse_referral_query, spell_check_location
from referral_engine import rank_facilities, resolve_location


APP_TITLE = "Referral Copilot"

app = Dash(__name__, title=APP_TITLE, suppress_callback_exceptions=True)
server = app.server


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


def render_evidence(items: list[dict[str, Any]]) -> list[html.Div]:
    if not items:
        return [html.Div("No direct matching evidence found in the facility record.", className="muted")]

    rendered = []
    for item in items[:6]:
        label = item.get("field", "evidence").replace("_", " ").title()
        terms = ", ".join(item.get("terms", [])[:5])
        snippet = item.get("snippet") or ""
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


# go.Scattermap  (Plotly >= 5.24) uses bundled Leaflet, no WebGL or CDN.
# go.Scattermapbox needs Mapbox GL JS/CDN/WebGL and can render blank in apps.
# go.Scattergeo  is always present and has zero external tile dependencies.
#
# We deliberately skip Scattermapbox: if Scattermap isn't available we fall
# straight to Scattergeo so the map always renders something.
_SCATTER_MAP_CLS = getattr(go, "Scattermap", None)   # None when Plotly < 5.24
_USE_TILE_MAP = (
    _SCATTER_MAP_CLS is not None
    and os.getenv("ENABLE_TILE_MAP", "false").lower() in {"1", "true", "yes", "on"}
)


def render_candidate_map(location: dict[str, Any], candidates: list[dict[str, Any]]) -> dcc.Graph:
    lats = [c.get("latitude") for c in candidates]
    lons = [c.get("longitude") for c in candidates]
    names = [c.get("name") for c in candidates]
    scores = [c.get("score", 0) for c in candidates]
    distances = [c.get("distance_km", 0) for c in candidates]
    candidate_ids = [c.get("candidate_id") for c in candidates]

    # Hover tooltips
    hover_text = []
    for c in candidates:
        evidence_terms: list[str] = []
        for item in c.get("evidence", [])[:3]:
            evidence_terms.extend(item.get("terms", [])[:3])
        evidence_label = ", ".join(dict.fromkeys(evidence_terms)) or "No direct evidence"
        hover_text.append(
            "<br>".join(
                [
                    f"<b>{c.get('name') or 'Unnamed facility'}</b>",
                    f"Distance: {c.get('distance_km', 0):.1f} km",
                    f"Score: {c.get('score', 0):.0f}",
                    f"Type: {c.get('facility_type') or 'facility'}",
                    f"Evidence: {evidence_label}",
                ]
            )
        )

    # Auto-zoom and center shared by both renderers.
    all_lats = [float(v) for v in lats + [location.get("latitude")] if v is not None]
    all_lons = [float(v) for v in lons + [location.get("longitude")] if v is not None]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    span = max(max(all_lats) - min(all_lats), max(all_lons) - min(all_lons), 0.05)
    zoom = (
        12 if span < 0.25 else
        11 if span < 0.5  else
        10 if span < 1.0  else
        9  if span < 2.5  else
        8  if span < 5    else
        7  if span < 10   else
        6  if span < 20   else
        5
    )

    origin_label = location.get("label") or "Search origin"
    origin_hover = f"<b>{origin_label}</b><br>{location.get('method') or ''}"

    marker_colorscale = [
        [0.0,  "#64748b"],
        [0.35, "#f59e0b"],
        [0.7,  "#0d9488"],
        [1.0,  "#0f766e"],
    ]
    marker_colorbar = {
        "title": {"text": "Score"},
        "thickness": 14,
        "len": 0.55,
        "y": 0.72,
        "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": "#dde3ea",
        "borderwidth": 1,
        "tickfont": {"size": 11},
    }

    fig = go.Figure()

    if _USE_TILE_MAP:
        # go.Scattermap (Plotly >= 5.24, Leaflet, no CDN/WebGL).
        fig.add_trace(
            _SCATTER_MAP_CLS(
                lat=lats,
                lon=lons,
                mode="markers+text",
                text=[str(i + 1) for i in range(len(candidates))],
                textposition="middle center",
                textfont={"size": 11, "color": "#ffffff"},
                hovertext=hover_text,
                hoverinfo="text",
                marker={
                    "size": [max(14, min(34, 14 + s / 8)) for s in scores],
                    "color": scores,
                    "colorscale": marker_colorscale,
                    "opacity": 0.92,
                    "colorbar": marker_colorbar,
                },
                name="Facilities",
                customdata=candidate_ids,
            )
        )
        fig.add_trace(
            _SCATTER_MAP_CLS(
                lat=[location.get("latitude")],
                lon=[location.get("longitude")],
                mode="markers+text",
                text=[f"  {origin_label}"],
                textposition="middle right",
                textfont={"size": 12, "color": "#7f1d1d"},
                hovertext=[origin_hover],
                hoverinfo="text",
                marker={"size": 24, "color": "#b91c1c", "opacity": 1.0},
                name="Search origin",
            )
        )
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hoverlabel={
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "font": {"color": "#f8fafc", "size": 13},
            },
            map={
                "style": "open-street-map",
                "center": {"lat": center_lat, "lon": center_lon},
                "zoom": zoom,
            },
        )

    else:
        # go.Scattergeo fallback (always available, vector outline).
        projection_scale = max(2.5, min(18, 15 / max(span, 1.0)))
        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="markers+text",
                text=[str(i + 1) for i in range(len(candidates))],
                textposition="middle center",
                textfont={"size": 10, "color": "#ffffff"},
                hovertext=hover_text,
                hoverinfo="text",
                marker={
                    "size": [max(10, min(26, 10 + s / 10)) for s in scores],
                    "color": scores,
                    "colorscale": marker_colorscale,
                    "line": {"width": 1, "color": "#ffffff"},
                    "colorbar": marker_colorbar,
                },
                name="Facilities",
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
                hovertext=[origin_hover],
                hoverinfo="text",
                marker={"size": 16, "color": "#b91c1c", "symbol": "star", "line": {"width": 1, "color": "#ffffff"}},
                name="Search origin",
            )
        )
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            geo={
                "scope": "asia",
                "center": {"lat": center_lat, "lon": center_lon},
                "projection": {"type": "mercator", "scale": projection_scale},
                "showland": True, "landcolor": "#f8fafc",
                "showocean": True, "oceancolor": "#e0f2fe",
                "showcountries": True, "countrycolor": "#cbd5e1",
                "showsubunits": True, "subunitcolor": "#e2e8f0",
                "fitbounds": "locations",
            },
        )

    return dcc.Graph(
        id="candidate-map",
        figure=fig,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
        className="candidate-map",
    )


def render_map_selection(candidate: dict[str, Any] | None = None) -> html.Div:
    if not candidate:
        return html.Div("Click a numbered marker to inspect a facility.", className="muted")

    evidence_terms: list[str] = []
    for item in candidate.get("evidence", [])[:3]:
        evidence_terms.extend(item.get("terms", [])[:4])
    evidence_terms = list(dict.fromkeys(evidence_terms))

    return html.Div(
        className="map-selection-card",
        children=[
            html.Div("Selected facility", className="section-label"),
            html.H3(candidate.get("name") or "Unnamed facility"),
            html.Div(
                className="candidate-meta",
                children=[
                    chip(f"{candidate.get('distance_km', 0):.1f} km"),
                    chip(f"score {candidate.get('score', 0):.0f}"),
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
    source_urls = candidate.get("source_urls") or []
    contact_bits = []

    if candidate.get("phone"):
        contact_bits.append(chip(candidate["phone"], "chip chip-soft"))
    if candidate.get("email"):
        contact_bits.append(chip(candidate["email"], "chip chip-soft"))
    if candidate.get("website"):
        contact_bits.append(
            html.A("Website", href=candidate["website"], target="_blank", rel="noreferrer", className="chip chip-link")
        )
    if source_urls:
        contact_bits.append(
            html.A("Source", href=source_urls[0], target="_blank", rel="noreferrer", className="chip chip-link")
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
                        children=[html.Strong(f"{candidate.get('score', 0):.0f}"), html.Span("score")],
                    ),
                ],
            ),
            html.Div(
                className="candidate-meta",
                children=[
                    chip(f"{candidate.get('distance_km', 0):.1f} km"),
                    chip(candidate.get("facility_type") or "facility"),
                    chip(candidate.get("operator_type") or "operator unknown"),
                    chip(candidate.get("city_state") or "location unknown"),
                ],
            ),
            html.Div(className="section-label", children="Matching evidence"),
            html.Div(className="evidence-list", children=render_evidence(evidence)),
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
                        children=[html.H2("Referral Map"), html.Span("Drag, zoom, and hover markers for facility evidence.")],
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
                                    chip(f"top score: {highest.get('name')}", "chip chip-soft"),
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

    return html.Div(
        className="shortlist-items",
        children=[
            html.Div(
                className="shortlist-item",
                children=[
                    html.Strong(item.get("name") or "Unnamed facility"),
                    html.Span(f"{item.get('distance_km', 0):.1f} km - score {item.get('score', 0):.0f}"),
                ],
            )
            for item in shortlist
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
        dcc.Download(id="shortlist-download"),
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
                                    html.Label("Radius km", htmlFor="radius-input"),
                                    dcc.Input(
                                        id="radius-input",
                                        type="number",
                                        min=10,
                                        max=1000,
                                        step=10,
                                        value=250,
                                        className="number-input",
                                    ),
                                ]),
                                html.Div([
                                    html.Label("Results", htmlFor="limit-input"),
                                    dcc.Input(
                                        id="limit-input",
                                        type="number",
                                        min=3,
                                        max=25,
                                        step=1,
                                        value=8,
                                        className="number-input",
                                    ),
                                ]),
                            ],
                        ),
                        html.Button("Search", id="search-button", className="primary-button"),
                        html.Div(id="search-status", className="search-status"),
                    ],
                ),
                # Center: results panel
                html.Section(id="results-panel", className="results-panel", children=render_empty_state()),
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
                        html.Div(id="shortlist-panel-body", children=render_shortlist([])),
                        html.Button("Download CSV", id="download-shortlist", className="secondary-button full-width"),
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

        # Resolve parsed need and location per mode.
        if mode == "freetext":
            raw_query = (query or "").strip()
            if not raw_query:
                return render_empty_state(), [], "Enter a care need and location."
            parsed = parse_referral_query(raw_query)
            location = resolve_location(parsed.location, datasets.facilities, datasets.pincodes)
            if (not location.get("latitude") or not location.get("longitude")) and parsed.location:
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
            radius_km=float(radius_km or 250),
            limit=int(limit or 8),
        )

        parsed_dict = parsed.to_dict()
        return (
            render_results(parsed_dict, location, candidates, data_notes),
            candidates,
            f"Found {len(candidates)} candidates using {parsed.source} parsing. {correction_note}".strip(),
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
    State("candidate-store", "data"),
    prevent_initial_call=True,
)
def update_map_selection(click_data, candidates):
    candidates = candidates or []
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
    Output("shortlist-download", "data"),
    Input("download-shortlist", "n_clicks"),
    State("shortlist-store", "data"),
    prevent_initial_call=True,
)
def download_shortlist(n_clicks, shortlist):
    shortlist = shortlist or []
    rows = []
    for item in shortlist:
        rows.append(
            {
                "name": item.get("name"),
                "distance_km": item.get("distance_km"),
                "score": item.get("score"),
                "facility_type": item.get("facility_type"),
                "operator_type": item.get("operator_type"),
                "city_state": item.get("city_state"),
                "phone": item.get("phone"),
                "email": item.get("email"),
                "website": item.get("website"),
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
