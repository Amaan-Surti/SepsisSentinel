# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Exhibition visitors** at a college project exhibition: classmates, faculty, examiners and members of the public who walk up to the presenter's laptop, stand about an arm's length to a metre away, and stay one to five minutes. Many arrive mid-demo and need to understand what they are looking at within seconds. Some will lean in and click or scrub themselves.
- **The presenter** (the project author, Amaan Surti) drives the demo on his own laptop, tells the story bed by bed, and answers questions, including hard methodology questions from examiners.

## Product Purpose

SepsisSentinel replays 24 real, de-identified ICU stays from a held-out test set hour by hour and shows a sepsis early-warning model at work: calibrated risk per bed, alerts at a chosen threshold, the reasons behind each risk, and what actually happened (onset) when ground truth is revealed. Success at the exhibition: a visitor leaves knowing what the system predicts, that it can warn hours before onset, that it sometimes misses or false-alarms, and why its numbers can be trusted.

## Positioning

- Every number is real and traceable to a file the pipeline wrote; nothing is mocked.
- Risks are calibrated (a displayed 3% means about 3 in 100 such patient-hours turn septic).
- The lockbox test set (8,068 patients) was evaluated exactly once, on LightGBM (normalized utility 0.4270). The served model is a LightGBM + GRU ensemble chosen on development data; its lockbox score has not been measured. These two kinds of result are never shown as if comparable.
- Failures are shown on purpose: missed septic patients, false alarms, and a 32.7% utility drop when the model moves to another hospital.

## Operating Context

- Runs locally on the presenter's laptop (Docker Compose: API on :8000, UI on :8080, or the Vite dev server); exhibition Wi-Fi cannot be relied on.
- A laptop screen at 1280×720 to 1920×1080, viewed up close in a bright, noisy exhibition hall, often with several people looking at once.
- The presenter moves through a replay clock (play/pause, step, speed), opens beds, toggles "show ground truth", and moves the alert threshold; keyboard shortcuts (Space, ← / →) and deep links (`?t=<hour>&truth=1`) exist.
- Supporting material: slides, demo script, viva guide (`reports/`).

## Capabilities and Constraints

- Data: 24 demo beds (8 early catches, 2 missed/late, 10 quiet, 4 false-alarm under LightGBM; recomputed per served model), hourly vitals, labs, calibrated risk, alerts, top-5 SHAP reasons per hour (from the LightGBM component), ground-truth onset hour.
- API (FastAPI): `/api/config`, `/api/ward`, `/api/patients/{id}`, `/api/predict`, `/api/metrics` (separate `lockbox_test` and `dev_cv` sections), `/api/model-card`, `/api/figures/{name}`; `MODEL_KIND` = `ensemble` (default) or `lgbm`.
- Frontend: React 18 + TypeScript + Vite + Recharts; no heavy new dependencies.
- Status (Alert / Watch / OK / Discharged) is never shown by colour alone.
- The disclaimer "Decision support on retrospective data. Not a medical device." appears on every page.
- Keyboard accessible, WCAG AA contrast, usable at 1280×720 and 1440×900.
- Terminology: Alert = calibrated risk ≥ threshold τ; Watch = risk ≥ ½τ; policy τ = 0.03, k = 1, sticky = 0.

## Brand Commitments

- Name: SepsisSentinel.
- Must not look like a startup SaaS product (the author's explicit constraint).
- Voice: calm, factual, honest about limits.

## Evidence on Hand

- `data/demo/` (24 beds: rows, predictions for both models, patient table), `reports/tables/` (all result tables), `reports/figures/` (calibration, ROC/PR, cross-hospital and other figures), `reports/model_card.md`.
- No clinician testimonials, deployments, users or regulatory claims exist; none may be implied.

## Product Principles

1. Show, don't claim: the replay of real stays is the argument.
2. Honesty is the feature: misses, false alarms and limits are as visible as successes.
3. Keep the evidence apart: lockbox (LightGBM) and development (ensemble) numbers never share a table.
4. Understandable in seconds by someone who walked up mid-story; deep enough for an examiner who stays.

## Accessibility & Inclusion

WCAG AA contrast, full keyboard operation, status by shape and word as well as colour, readable at arm's length on a laptop in bright ambient light.
