# Phase 7 QA Checklist — SepsisSentinel dashboard

How to run: `docker compose up --build` → http://localhost:8080 (or terminal 1 `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000`; terminal 2 `cd frontend; npm run dev` → http://localhost:5173).

**Run on 2026-10-19 by Claude** against the Docker deployment (`docker compose up --build`, http://localhost:8080, nginx → FastAPI) in Chrome at 1280×720. The same checks were run earlier against the Vite dev server. A human may re-tick after their own walkthrough.

## Manual checklist (Build Plan §11.6)

| # | Check | Evidence (Docker, 2026-10-19) | Result |
|---|---|---|---|
| 1 | Ward loads with 24 beds sorted by risk at t = 0 | 24 rows, risk descending; 2 Alert / 5 Watch / 17 OK; API check `sorted: True` | [x] |
| 2 | Play advances hours; risks change | Play at 2× → hour 15; order and risks changed (Bed 03 top at 2.7%) | [x] |
| 3 | A septic demo patient shows an alert before the onset line (show-truth on) | Bed 03: alert at hour 16, onset line at hour 22 ("first alert 6 h before onset") | [x] |
| 4 | Slider lower → more alerts; higher → fewer | t = 19: 1.5% → 3 Alert, 3.0% → 2, 10.0% → 0; API at t = 40: 0.015 → 6, 0.03 → 3, 0.10 → 0 | [x] |
| 5 | Patient page charts align on the same hours | Risk, 5 vitals and 4 lab charts share x-domain 0–24 h; onset line at the same hour in each | [x] |
| 6 | Reasons panel updates when moving hour | Hour 16: "FiO2 last measured 0h ago", "Temperature 38.4" → hour 19: "FiO2 last measured 3h ago", "Temperature 38.7" | [x] |
| 7 | Discharged patients greyed | Hour 30: 4 discharged beds greyed, sorted last, labelled "Discharged" | [x] |
| 8 | Backend stopped → clear error message | `docker compose stop backend` → nginx 502 in ~3 s → "Cannot reach the API at /api — is the backend running?" + Retry; after restart, Retry restores the ward | [x] |
| 9 | Model page numbers equal `final_test_metrics.csv` | 0.4270 / 0.8475 / 0.1369 / 0.0164 / 0.0027 / 87.54% / 29.0 h / 0.529 / 4.62% / 20.2%; baselines match `final_test_baselines.csv`; figures load via `/api/figures` | [x] |
| 10 | Works in Chrome at 1280×720 | Ward, patient and model pages usable at 1280×720 (also checked at 1440×900) | [x] |

## Phase 7 acceptance criteria

| Criterion | Evidence | Status |
|---|---|---|
| P7.T1 cohort = 8 early / 2 missed-late / 10 quiet / 4 false-alert, stay 24–96 h, lockbox test only, seed 42 | `python -m sepsis.demo_cohort`; `test_demo_cohort_composition` | Pass |
| Demo outputs written | `data/demo/demo_rows.parquet` (1,022 rows), `demo_predictions.parquet`, `demo_patients.csv` (Bed 01–24, shuffled) | Pass |
| Online path equals batch path | demo build asserts max \|Δrisk\| ≤ 1e-6 (observed 0.0); `test_predict_consistency` | Pass |
| All 7 endpoints + figures endpoint respond; 404 unknown patient; 422 invalid params | `backend/tests/test_api.py` (14 tests) | Pass |
| CORS for :5173 and :8080 | `backend/app/main.py` | Pass |
| `npm run build` with no TypeScript errors | Vite 5.4.8 build OK (only a bundle-size warning) | Pass |
| Status never by colour alone | Level chips carry text + glyph (▲ Alert, ● Watch, – OK) | Pass |
| Disclaimer visible | Banner on every page + footer | Pass |
| `docker compose up --build` → ward at :8080; replay works end to end | Both images built; backend and frontend containers **healthy**; nginx proxies `/api/` and serves SPA routes (`/patient/:id` → 200) | Pass |

## Issues found and fixed during QA

- Frontend container reported *unhealthy*: inside Alpine, `localhost` resolves to `::1` while nginx listens on IPv4 only. Healthcheck changed to `http://127.0.0.1/` in `frontend/Dockerfile`; container now healthy.
- Discharged rows previously still showed their last level chip and "for N h"; now they show "Discharged" with no alert duration.

## Known caveats

- Recharts 2.12.7 logs React `defaultProps` deprecation warnings in the dev console (library issue at the pinned version; no functional effect).
- `threshold_curve.json` starts at τ = 0.05, so the policy τ = 0.03 is marked separately on the trade-off chart.
- With the backend down, the error appears after ~3 s in Docker (nginx connect failure) vs immediately under the Vite dev proxy.
- Ward vitals show the last recorded value (labs such as WBC are often "—" early in a stay because they have not been measured yet).
