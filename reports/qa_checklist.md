# QA Checklist — SepsisSentinel dashboard

How to run: `docker compose up --build` → http://localhost:8080 (or terminal 1 `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000`; terminal 2 `cd frontend; npm run dev` → http://localhost:5173). `MODEL_KIND=lgbm` serves LightGBM alone; the default is the ensemble.

**Re-run on 2026-11-03 by Claude for the Gallery Label redesign** (branch `feature/ui-redesign`) against the Docker deployment (images 1.1, `MODEL_KIND=ensemble`, http://127.0.0.1:8080, nginx → FastAPI), driving the page in a Chromium browser at 1280×720; screenshots from headless Chrome at 1440×900 (`reports/screenshots/capture.py`). Earlier runs (Phase 7, LightGBM; Phase 9, refined UI) also passed 10/10. A human may re-tick after their own walkthrough.

## Manual checklist (Build Plan §11.6)

| # | Check | Evidence (Docker, ensemble, Gallery Label) | Result |
|---|---|---|---|
| 1 | Ward loads with 24 beds sorted by risk at t = 0 | Ward wall: 24 rows, risk descending; badge "Ensemble"; disclaimer in header and footer | [x] |
| 2 | Play advances hours; risks change | Play at 2× → hour 12 after ~6 s, risks changed; → steps to hour 13 | [x] |
| 3 | A septic demo patient shows an alert before the onset line (show-truth on) | Tour stop 2 → `/bed/p003071` at hour 16: first alert hour 16; the tombstone reads "WHAT HAPPENED Sepsis onset at ICU hour 22: the first alert came 6 h before it."; onset line on the chart | [x] |
| 4 | Slider lower → more alerts; higher → fewer | Tour stop 5 (hour 40), active beds: 1.5% → 5 Alert, 3.0% → 2, 10.0% → 0; "reset to policy" restores 3.0% | [x] |
| 5 | Patient page charts align on the same hours | Risk chart axis ends at hour 24 (= stay − 1); all 34 measurement charts share that axis; flowsheet has one column per hour 0–16 | [x] |
| 6 | Reasons panel updates when moving hour | Bed 03 hour 16: "FiO2 last measured 0h ago", "Temperature 38.4" → hour 19: "FiO2 last measured 3h ago", "Temperature 38.7"; *Hour by hour* lists each hour's reasons | [x] |
| 7 | Discharged patients greyed | Hour 40: 10 discharged rows sorted last, muted, labelled "Discharged after hour N" | [x] |
| 8 | Backend stopped → clear error message | `docker compose stop backend` → "Cannot reach the API at /api — is the backend running?" + Retry (`role="alert"`) after 3.3 s; `docker compose start backend` + Retry restores 24 beds | [x] |
| 9 | Model page numbers equal `final_test_metrics.csv` | Evidence → Lockbox room: 0.4270 / 0.8475 / 0.1369 / 0.0164 / 0.0027 / 87.54% / 29.0 h / 0.529 / 4.62% / 20.2%; baselines 0.0000 / 0.0507 / 0.0122 / 0.2438; development annex equals `ensemble_dev_comparison.csv` and `ensemble_dev_weights.csv` | [x] |
| 10 | Works in Chrome at 1280×720 | Ward, bed (all three tabs), tour and evidence usable at 1280×720 and 1440×900; no horizontal overflow; the bed's WHAT HAPPENED line and threshold crossing are on the first screen | [x] |

## Gallery Label acceptance

| Criterion | Evidence | Status |
|---|---|---|
| Every vital and every lab, every hour | Bed → *Every measurement*: 34 charts (8 vitals, 26 labs) + flowsheet of every measurement at every hour; API test `test_patient_view_has_every_measurement_and_hourly_insights` | Pass |
| Insights for every hour | Bed → *Hour by hour*: risk and change, LightGBM and GRU parts, status, all measurements that hour, that hour's top-5 reasons (`reasons_by_hour`) | Pass |
| Lockbox and development results never mixed | Evidence: lockbox room on the gallery wall, development annex on its own slate-blue wall with a dashed edge and "Development data · not lockbox" tag | Pass |
| Status never by colour alone | Shape + word everywhere (triangle Alert, dot Watch, bar OK, "Discharged after hour N"); flowsheet status uses words | Pass |
| Contrast | Measured in the browser: every text pair ≥ 5.98:1 on wall and label grounds (dimmed strip ≈ 5:1) | Pass |
| `POST /api/predict` in Docker | max \|Δ\| vs stored ensemble predictions 0.0, alerts equal | Pass |
| impeccable detector / finish review | `impeccable detect` 0 findings; finish review: all 8 material fixes resolved (the eighth, self-hosted fonts, after the owner approved the download) | Pass |

## Issues found and fixed during this QA

- With the backend down, nginx kept the stopped container's address and waited out its 60 s connect timeout before answering 502; `proxy_connect_timeout 3s` in `frontend/nginx.conf` brings the error message to ~3 s.

## Known caveats

- Fonts are self-hosted (`frontend/public/fonts/`, SIL OFL 1.1); with every host but 127.0.0.1 blocked, both families load from `/fonts/` and the page makes no external request.
- Recharts 2.12.7 logs React `defaultProps` deprecation warnings in the dev console (library issue at the pinned version; no functional effect).
- `threshold_curve.json` is the LightGBM development curve and starts at τ = 0.05; the policy τ = 0.03 is marked separately.
- The threshold slider spans 1–50% (not the plan's 5–95%) because the policy τ = 0.03 lies below 5%.
- Lab units follow the PhysioNet 2019 documentation (e.g. lactate in mg/dL as documented).
