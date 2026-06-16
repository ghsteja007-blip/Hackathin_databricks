# Referral Copilot

Dash app for evidence-attached referral shortlists from the `databricks_virtue_foundation_dataset_dais_2026` schema.

## What it does

- Accepts searches like `dialysis near Jaipur` or `emergency surgery near Patna`.
- Parses the location and care need with OpenAI when `OPENAI_API_KEY` is configured.
- Resolves the location from pincode/facility geography.
- Falls back to OpenAI web search for India location resolution when local sample geography is incomplete and `ENABLE_WEB_RESOLUTION=true`.
- Ranks nearby facilities using distance plus evidence from facility fields such as `specialties`, `procedure`, `equipment`, `capability`, and `description`.
- Displays an interactive referral map with numbered facility markers, hover evidence, and click-to-inspect details.
- Includes a light/dark theme toggle.
- Shows matching evidence, missing or suspicious evidence, and a saveable shortlist.
- Adds a shortlist chat copilot that can compare saved facilities and use OpenAI web search for fresh external details when needed.

This is referral decision support, not medical advice. Coordinators should verify availability, emergency readiness, eligibility, and clinical fit before sending a patient.

## Databricks setup

Deploy the package with `app.yaml`, `app.py`, `requirements.txt`, and the Python modules at the source root. Do not upload a zip that wraps the app inside an extra top-level folder, because Databricks Apps will not see the expected source root.

Create a Databricks App and add these resources:

- SQL warehouse resource with key `sql-warehouse`.
- Secret resource with key `secret`, containing your OpenAI API key.
- Unity Catalog table permissions for the app service principal.

The app expects these tables by default:

- `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities`
- `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory`
- `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.nfhs_5_district_health_indicators`

`ENABLE_WEB_RESOLUTION=true` lets the app use OpenAI's Responses API web search tool to resolve city/district coordinates when the sample pincode or facility geography cannot resolve a place. Local data still wins when available.

`ENABLE_TILE_MAP=false` keeps the app on a Plotly India map with a built-in outline, route lines, city labels, and numbered markers. Set it to `true` only if your app environment can render tile maps reliably.

## Secret handling

Do not hardcode API keys in this repo or `app.yaml`. The app reads `OPENAI_API_KEY` from a Databricks secret resource named `secret`.

Because an API key was pasted into chat, rotate that key in the OpenAI dashboard and store the replacement in Databricks secrets before deployment.

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:LOCAL_DATA_DIR="C:\Users\hgannama\Downloads"
$env:OPENAI_API_KEY="<your-rotated-key>"
python app.py
```

Open `http://localhost:8050`.

Without `OPENAI_API_KEY`, the app still works with fallback parsing.

## Notes

- The current ranking is deterministic and evidence-grounded.
- OpenAI parses the query, autocorrects obvious location typos, resolves geography gaps when enabled, and powers shortlist chat.
- Web resolution and chat web search should be treated as external context, not facility evidence replacement.
- The default map uses Plotly `Scattergeo` traces with a built-in simplified India outline, so it does not require a Mapbox token or external tile access.
- The pincode file you inspected appears AP/Telangana-heavy, so city matching may fall back to facility addresses for other states.
- Some facility rows have malformed coordinates or shifted columns; the ranking skips invalid coordinates and flags evidence gaps.
