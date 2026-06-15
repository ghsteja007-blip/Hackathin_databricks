import json
import os
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
import plotly.graph_objects as go

from data_access import load_datasets
from openai_helpers import parse_referral_query
from referral_engine import rank_facilities, resolve_location


APP_TITLE = "Referral Copilot"


app = Dash(__name__, title=APP_TITLE, suppress_callback_exceptions=True)
server = app.server


def chip(text: str, class_name: str = "chip") -> html.Span:
    return html.Span(text, className=class_name)


def render_empty_state() -> html.Div:
    return html.Div(
        className="empty-state",
        children=[
            html.H2("Start a referral search"),
            html.P("Try: dialysis near Jaipur, emergency surgery near Patna, or pediatric ICU near 380007."),
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


def render_candidate_map(location: dict[str, Any], candidates: list[dict[str, Any]]) -> dcc.Graph:
    lats = [candidate.get("latitude") for candidate in candidates]
    lons = [candidate.get("longitude") for candidate in candidates]
    names = [candidate.get("name") for candidate in candidates]
    scores = [candidate.get("score", 0) for candidate in candidates]
    distances = [candidate.get("distance_km", 0) for candidate in candidates]

    hover_text = []
    for candidate in candidates:
        evidence_terms = []
        for item in candidate.get("evidence", [])[:3]:
            evidence_terms.extend(item.get("terms", [])[:3])
        evidence_label = ", ".join(dict.fromkeys(evidence_terms)) or "No direct evidence"
        hover_text.append(
            "<br>".join(
                [
                    f"<b>{candidate.get('name') or 'Unnamed facility'}</b>",
                    f"Distance: {candidate.get('distance_km', 0):.1f} km",
                    f"Score: {candidate.get('score', 0):.0f}",
                    f"Type: {candidate.get('facility_type') or 'facility'}",
                    f"Evidence: {evidence_label}",
                ]
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
                "size": [max(10, min(26, 10 + score / 10)) for score in scores],
                "color": scores,
                "colorscale": [[0, "#94a3b8"], [0.45, "#f59e0b"], [1, "#0f766e"]],
                "line": {"width": 1, "color": "#ffffff"},
                "colorbar": {"title": "Score", "thickness": 12},
            },
            name="Candidate facilities",
            customdata=distances,
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
            html.A(
                "Website",
                href=candidate["website"],
                target="_blank",
                rel="noreferrer",
                className="chip chip-link",
            )
        )
    if source_urls:
        contact_bits.append(
            html.A(
                "Source",
                href=source_urls[0],
                target="_blank",
                rel="noreferrer",
                className="chip chip-link",
            )
        )

    return html.Article(
        className="candidate-card",
        children=[
            html.Div(
                className="candidate-topline",
                children=[
                    html.Div(
                        [
                            html.Div(f"#{rank}", className="rank"),
                            html.H3(candidate.get("name") or "Unnamed facility"),
                        ]
                    ),
                    html.Div(
                        className="score-stack",
                        children=[
                            html.Strong(f"{candidate.get('score', 0):.0f}"),
                            html.Span("score"),
                        ],
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
    location_note = f"{location.get('label')} via {location.get('method')} ({location.get('match_count', 0)} matches)"

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
                            html.Span("Drag, zoom, and hover markers for facility evidence."),
                        ],
                    ),
                    render_candidate_map(location, candidates),
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
                    html.Span(f"{item.get('distance_km', 0):.1f} km · score {item.get('score', 0):.0f}"),
                ],
            )
            for item in shortlist
        ],
    )


app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Store(id="candidate-store", data=[]),
        dcc.Store(id="shortlist-store", data=[]),
        dcc.Store(id="geolocation-store", data=None),
        dcc.Download(id="shortlist-download"),
        html.Header(
            className="app-header",
            children=[
                html.Div(
                    [
                        html.H1(APP_TITLE),
                        html.P("Evidence-attached care facility shortlists."),
                    ]
                ),
                html.Div(className="status-pill", children=os.getenv("DATABRICKS_SCHEMA", "local data")),
            ],
        ),
        html.Main(
            className="workspace",
            children=[
                html.Section(
                    className="search-panel",
                    children=[
                        html.Label("Care need and location", htmlFor="query-input"),
                        dcc.Input(
                            id="query-input",
                            type="text",
                            value="dialysis near Jaipur",
                            placeholder="dialysis near Jaipur",
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
                        html.Div(
                            className="controls-row",
                            children=[
                                html.Div(
                                    [
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
                                    ]
                                ),
                                html.Div(
                                    [
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
                                    ]
                                ),
                            ],
                        ),
                        html.Button("Search", id="search-button", className="primary-button"),
                        html.Div(id="search-status", className="search-status"),
                    ],
                ),
                html.Section(id="results-panel", className="results-panel", children=render_empty_state()),
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


@app.callback(
    Output("results-panel", "children"),
    Output("candidate-store", "data"),
    Output("search-status", "children"),
    Input("search-button", "n_clicks"),
    State("query-input", "value"),
    State("radius-input", "value"),
    State("limit-input", "value"),
    State("geolocation-store", "data"),
    prevent_initial_call=True,
)
def run_search(n_clicks: int, raw_query: str, radius_km: int, limit: int, geolocation_data: dict | None):
    if not raw_query or not raw_query.strip():
        return render_empty_state(), [], "Enter a care need and location."

    try:
        datasets, data_notes = load_datasets()
        parsed = parse_referral_query(raw_query)

        gps_lat = geolocation_data.get("latitude") if geolocation_data else None
        gps_lon = geolocation_data.get("longitude") if geolocation_data else None

        if gps_lat is not None and gps_lon is not None and not geolocation_data.get("error"):
            accuracy = geolocation_data.get("accuracy")
            accuracy_note = f"±{int(accuracy)}m" if accuracy else ""
            location = {
                "label": f"Your location {accuracy_note}".strip(),
                "latitude": gps_lat,
                "longitude": gps_lon,
                "method": f"device GPS{' ' + accuracy_note if accuracy_note else ''}",
                "match_count": 1,
                "warnings": [],
            }
        else:
            location = resolve_location(parsed.location, datasets.facilities, datasets.pincodes)

        if not location.get("latitude") or not location.get("longitude"):
            return (
                html.Div(
                    className="empty-state",
                    children=[
                        html.H2("Location not resolved"),
                        html.P("Use a city, district, state, or six-digit pincode present in the loaded data."),
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
            f"Found {len(candidates)} candidates using {parsed.source} parsing.",
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
    return dcc.send_data_frame(__import__("pandas").DataFrame(rows).to_csv, "referral_shortlist.csv", index=False)


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
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy: pos.coords.accuracy
                    });
                },
                function(err) {
                    resolve({error: err.message});
                },
                {timeout: 10000, maximumAge: 60000, enableHighAccuracy: true}
            );
        });
    }
    """,
    Output("geolocation-store", "data"),
    Input("locate-button", "n_clicks"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(data) {
        if (!data) return '';
        if (data.error) return '⚠️ ' + data.error;
        var acc = data.accuracy ? ' (±' + Math.round(data.accuracy) + 'm)' : '';
        return '✅ Location acquired' + acc;
    }
    """,
    Output("gps-status", "children"),
    Input("geolocation-store", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    port = int(os.getenv("PORT") or os.getenv("DATABRICKS_APP_PORT") or "8050")
    app.run_server(host="0.0.0.0", port=port, debug=os.getenv("DASH_DEBUG", "false").lower() == "true")
