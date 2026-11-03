# QA Checklist — SepsisSentinel dashboard

How to run: `docker compose up --build` → http://localhost:8080 (or terminal 1 `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000`; terminal 2 `cd frontend; npm run dev` → http://localhost:5173). `MODEL_KIND=lgbm` serves LightGBM alone; the default is the ensemble.

**Re-run on 2026-11-03 by Claude (Phase 9)** against the Docker deployment (images 1.1, `MODEL_KIND=ensemble`, http://127.0.0.1:8080, nginx → FastAPI), driving the page in a Chromium browser at 1280×720; screenshots from headless Chrome at 1440×900 and 1280×720 (`reports/screenshots/capture.py`). The first run (Phase 7, 2026-10-19, LightGBM) also passed 10/10. A human may re-tick after their own walkthrough.

## Manual checklist (Build Plan §11.6)

| # | Check | Evidence (Docker, ensemble, 2026-11-03) | Result |
|---|---|---|---|
| 1 | Ward loads with 24 beds sorted by risk at t = 0 | 24 rows, risk descending; summary 1 Alert / 6 Watch / 17 OK / 0 Discharged; badge "Ensemble"; disclaimer in header and footer | [x] |
| 2 | Play advances hours; risks change | Play at 2× → hour 12 after ~6 s; risks and order changed (top bed Bed 15 → Bed 17); → / ← keys step one hour | [x] |
| 3 | A septic demo patient shows an alert before the onset line (show-truth on) | Bed 03 via `/?t=16&truth=1`: first alert hour 16, onset line hour 22, strip reads "first alert 6 h before onset" | [x] |
| 4 | Slider lower → more alerts; higher → fewer | Hour 40, active beds: 1.5% → 5 Alert, 3.0% → 2, 10.0% → 0; slider announces "1.5%" (`aria-valuetext`); "reset to policy 3.0%" restores 0.03 | [x] |
| 5 | Patient page charts align on the same hours | Risk chart and all 9 vitals/lab charts end at hour 24 on a shared 0–24 axis; onset line drawn in each | [x] |
| 6 | Reasons panel updates when moving hour | Bed 03 hour 16: "FiO2 last measured 0h ago", "Temperature 38.4" → hour 19: "FiO2 last measured 3h ago", "Temperature 38.7"; note "Reasons come from the LightGBM part of the ensemble" | [x] |
| 7 | Discharged patients greyed | Hour 40: 10 discharged beds sorted last, grey row (#4c6168 on #f1f3f2, AA), labelled "Discharged after hour N" | [x] |
| 8 | Backend stopped → clear error message | `docker compose stop backend` → "Cannot reach the API at /api — is the backend running?" + Retry (`role="alert"`); after `docker compose start backend`, Retry restores 24 beds. nginx took ~22 s to return 502 (see caveats) | [x] |
| 9 | Model page numbers equal `final_test_metrics.csv` | Lockbox band: 0.4270 / 0.8475 / 0.1369 / 0.0164 / 0.0027 / 87.54% / 29.0 h / 0.529 / 4.62% / 20.2%; baselines equal `final_test_baselines.csv`; development band equals `ensemble_dev_comparison.csv`, `_weights.csv`, `_bootstrap.csv` | [x] |
| 10 | Works in Chrome at 1280×720 | Ward, patient and model pages usable at 1280×720 and 1440×900; no horizontal overflow (scrollWidth 1265 of 1280) | [x] |

## Phase 9 acceptance

| Criterion | Evidence | Status |
|---|---|---|
| Served model and version visible | Top-bar badge; `/api/health` and `/api/config` report `model_kind` and `ensemble-w0.7-lgbm-final-303r-f26b34fc+gru-f5e0224d` | Pass |
| Lockbox and development results never mixed | `/api/metrics` → `lockbox_test` and `dev_cv`; Model page white "Lockbox test (LightGBM)" band vs dashed tinted "Development comparison (nested 5-fold CV)" band tagged "Development data · not lockbox" | Pass |
| `POST /api/predict` returns ensemble risks | Docker: max \|Δ\| vs stored ensemble predictions 0.0, alerts equal (25-hour demo stay) | Pass |
| Both model kinds work | `backend/tests/test_api.py` runs every test under `ensemble` and `lgbm` (30 tests); Docker restarted with `MODEL_KIND=lgbm` → health `lgbm-final-303r-f26b34fc` | Pass |
| Status never by colour alone | Chips and summary use shape (triangle / dot / bar) + word; discharged rows labelled | Pass |
| Disclaimer on every page | Sticky header banner and footer: "Decision support on retrospective data. Not a medical device." (also on loading, error and 404 states) | Pass |
| Keyboard | Bed names are links; Space / ← / → replay shortcuts; visible focus ring; slider label fixed (`label.control` is the slider) | Pass |
| impeccable detector | `impeccable detect --json frontend/src` → 0 findings (3 before) | Pass |
| `npm run build` | TypeScript clean (`tsc --noEmit`), Vite build in the frontend image | Pass |

## Issues found and fixed during this QA

- The risk chart's text alternative read "in alert during hours 16" for a single hour; now "hour 16".
- `septic_other` was labelled "flagged more than 12 h early", but it also covers alerts less than 4 h before onset (Bed 13 at 2 h); now "flagged outside the 4–12 h early window".
- The P7 Docker stack (images 1.0) was still running and auto-started with Docker Desktop; it was stopped before the 1.1 images were brought up.

## Known caveats

- With the backend container stopped, nginx took ~22 s to answer 502 in this run (P7 recorded ~3 s); the UI shows "Loading…" until then. Restarting the backend and pressing Retry recovers.
- Recharts 2.12.7 logs React `defaultProps` deprecation warnings in the dev console (library issue at the pinned version; no functional effect).
- `threshold_curve.json` is the LightGBM development curve and starts at τ = 0.05; the policy τ = 0.03 is marked separately on the trade-off chart.
- Ward vitals show the last recorded value (labs such as WBC are often "—" early in a stay because they have not been measured yet).
- The threshold slider spans 1–50% (not the plan's 5–95%) because the policy τ = 0.03 lies below 5%.
