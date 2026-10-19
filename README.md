# SepsisSentinel

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-0E7C7B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/tests-67%20passing-2ea44f)
![Lint](https://img.shields.io/badge/ruff-clean-D7FF64)
![Status](https://img.shields.io/badge/use-research%20only-D9622B)

**Real-time early warning of ICU sepsis on the PhysioNet/Computing in Cardiology Challenge 2019 data, with calibrated risk, SHAP explanations, a cross-hospital generalization study and a replay dashboard.**

> ⚠️ Research demonstration on retrospective, de-identified ICU data. **Not a medical device** and not approved for clinical diagnostic use.

![Ward view](reports/screenshots/ward.jpg)

---

## Results at a glance

Lockbox test set: **8,068 patients**, evaluated **once**, 95% CIs from 1,000 patient-level bootstrap resamples.

| Metric | SepsisSentinel | 95% CI |
|---|---|---|
| **Normalized utility** (official Challenge score) | **0.4270** | 0.3929 – 0.4604 |
| AUROC | 0.8475 | 0.7965 – 0.8990 |
| AUPRC (chance = 1.79%) | 0.1369 | 0.0854 – 0.2311 |
| Patient sensitivity (alert in [t−12 h, t+3 h]) | 87.54% | 84.81% – 90.13% |
| Median first alert before onset | 29.0 h | 23.0 – 33.0 h |
| False alerts per patient-day | 0.529 | 0.512 – 0.547 |
| Expected calibration error | 0.0027 | 0.0013 – 0.0084 |

| Same test set | Utility | AUROC | Sensitivity | False alerts/day |
|---|---|---|---|---|
| B0 never alert | 0.0000 | 0.500 | 0.00% | 0.00 |
| B1 SIRS ≥ 2 | 0.0507 | 0.615 | 82.76% | 1.53 |
| B2 qSOFA ≥ 2 (partial) | 0.0122 | 0.516 | 25.94% | 0.45 |
| B3 logistic regression | 0.2438 | 0.737 | 67.06% | 0.25 |
| **SepsisSentinel (LightGBM + isotonic)** | **0.4270** | **0.848** | **87.54%** | **0.53** |

**Cross-hospital finding:** a model trained on hospital A loses **32.7%** of its utility on hospital B (0.4405 → 0.2963). Trained on B and tested on A it scores 0.3188, and false alerts nearly triple (0.33 → 0.93/day). Ranking survives (AUROC −0.03 to −0.08); the alert threshold does not.

Full write-up: [`reports/report.md`](reports/report.md) · all tables and figures: [`reports/results.md`](reports/results.md) · model card: [`reports/model_card.md`](reports/model_card.md)

---

## How it works

```mermaid
flowchart LR
    A[PhysioNet 2019<br/>40,336 PSV files] --> B[io + cleaning<br/>1,552,210 hourly rows]
    B --> C[Patient-level split<br/>32,268 dev / 8,068 lockbox]
    C --> D[298 causal features<br/>LOCF, missingness, rolling, deltas, scores]
    D --> E[LightGBM<br/>Optuna, 5-fold CV]
    E --> F[Isotonic calibration<br/>on dev OOF]
    F --> G[Alert policy<br/>τ = 0.03, k = 1, sticky = 0]
    G --> H[(models/final)]
    H --> I[FastAPI backend<br/>+ SHAP TreeExplainer]
    I --> J[React dashboard<br/>Ward · Patient · Model]
    C -. once .-> K[Lockbox test<br/>final_test_metrics.csv]
```

- **No leakage:** splits by patient; every feature at hour *t* uses hours ≤ *t* (enforced by a prefix-equality test); all fitting on dev data; the lockbox opened once.
- **Primary metric:** the official Challenge normalized utility, reproduced and unit-tested.
- **Calibrated:** isotonic regression, so a displayed 3% risk means about 3%.
- **Explainable:** top-5 SHAP reasons per hour in plain language ("Lactate last measured 2h ago", "Temperature 38.4").
- **Same code online and offline:** `src/sepsis/inference.py` serves both the demo cohort and `POST /api/predict` (parity test < 1e-6).

| Patient view | Model view |
|---|---|
| ![Patient](reports/screenshots/patient.jpg) | ![Model](reports/screenshots/model.jpg) |

---

## Repository layout

```
├── configs/config.yaml          # all paths, seeds, feature and model settings
├── src/sepsis/                  # importable pipeline (python -m sepsis.<module>)
│   ├── io.py cleaning.py eda.py splits.py utility.py metrics.py alerting.py
│   ├── features.py baselines.py train_lgbm.py tune_lgbm.py ablations.py final_model.py
│   ├── cross_hospital.py calibrate.py explain.py evaluate_final.py
│   ├── inference.py demo_cohort.py report.py
├── tests/                       # pytest for src/sepsis (53 tests)
├── backend/app/                 # FastAPI: main.py, service.py, schemas.py, settings.py
├── backend/tests/test_api.py    # API tests (14)
├── frontend/                    # Vite + React + TypeScript + Recharts dashboard
├── models/final/                # frozen model, calibrator, policy (tracked, 4 MB)
├── data/demo/                   # 24-bed demo cohort (tracked)
├── reports/                     # tables/, figures/, results.md, report.md, model_card.md,
│                                # slides/, demo_script.md, viva_qa.md, qa_checklist.md
├── docker-compose.yml
└── STATUS.md                    # task board, verified numbers, decisions
```

---

## Quick start: run the dashboard

The frozen model and demo cohort are in the repo, so the raw data is **not** needed to run the demo.

### Option A: Docker (recommended)

Requires Docker Desktop (WSL2 backend on Windows).

```bash
docker compose up --build
```

Open **http://localhost:8080**. The API is at http://localhost:8000/api/health (interactive docs at http://localhost:8000/docs).

### Option B: two terminals

```powershell
# one-time setup
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
cd frontend; npm ci; cd ..

# terminal 1 — API on :8000
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000

# terminal 2 — UI on :5173 (proxies /api to :8000)
cd frontend; npm run dev
```

Then press **Play**, open a bed, toggle **Show ground truth**, and move the threshold slider. A guided walkthrough is in [`reports/demo_script.md`](reports/demo_script.md).

### API

| Endpoint | Description |
|---|---|
| `GET /api/health` | status and model version |
| `GET /api/config` | policy (τ, k, sticky), replay length, disclaimer |
| `GET /api/ward?t=&threshold=` | 24 beds sorted by risk, with level ALERT / WATCH / OK |
| `GET /api/patients/{id}?t=&threshold=&show_truth=` | vitals, labs, risk, alerts, top-5 SHAP reasons |
| `POST /api/predict` | raw hourly rows → cleaning → features → model → calibrator → policy |
| `GET /api/metrics` | frozen test, baseline, cross-hospital and ablation tables |
| `GET /api/model-card` | model card (Markdown + sections) |
| `GET /api/figures/{name}` | selected result figures |

---

## Reproduce the research pipeline

### 1. Get the data

Download the public training sets from PhysioNet (open access): https://physionet.org/content/challenge-2019/1.0.0/

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/
```

Place the files so that `data/raw/training_setA/` holds 20,336 `.psv` files and `data/raw/training_setB/` holds 20,000.

### 2. Run the phases (from the repo root, venv active)

| Phase | Command | Output |
|---|---|---|
| P1 load | `python -m sepsis.io` | `data/interim/all_rows.parquet` (1,552,210 rows) |
| P1 clean | `python -m sepsis.cleaning` | `clean_rows.parquet`, `cleaning_report.csv` |
| P1 EDA | `python -m sepsis.eda` | EDA tables and figures, adversarial validation |
| P2 splits | `python -m sepsis.splits` | `data/splits/lockbox.csv`, `folds.csv` |
| P3 all | `python -m sepsis.run_phase3` | features, baselines, default/tuned LightGBM, ablations, final model |
| P4 | `python -m sepsis.cross_hospital` | `cross_hospital.csv`, figures |
| P5 calibrate + policy | `python -m sepsis.calibrate` | `calibrator.joblib`, `policy.json`, `threshold_curve.json` |
| P5 SHAP | `python -m sepsis.explain` | SHAP tables and figures |
| P5 **lockbox** | `python -m sepsis.evaluate_final` | `final_test_*.csv`, test figures. ⚠️ **Already run once. Re-running re-opens the test set; don't do it to chase numbers.** |
| P7 demo cohort | `python -m sepsis.demo_cohort` | `data/demo/*` |
| P8 results | `python -m sepsis.report` | `reports/results.md` |

Phase 3 can be run step by step with `python -m sepsis.features`, `sepsis.baselines`, `sepsis.train_lgbm`, `sepsis.tune_lgbm`, `sepsis.ablations`, `sepsis.final_model`, or use the `--skip-*` flags of `run_phase3`. Random seed 42 everywhere. Feature building takes about 5–15 minutes and Optuna about 2–3 hours on a laptop.

### 3. Check

```powershell
.venv\Scripts\python.exe -m pytest                       # 67 tests
.venv\Scripts\python.exe -m ruff check src tests backend
cd frontend; npm run build                               # TypeScript + Vite build
```

---

## Limitations

- Retrospective, de-identified data from two hospitals; the Challenge's hidden hospital C is not public, so external validity is only simulated with A ↔ B transfer.
- Labels come from Sepsis-3 criteria applied to records and are not perfect ground truth.
- The utility-optimal threshold (3%) puts 20.2% of hours in alert (event PPV 4.6%); any threshold trades missed cases against alarm fatigue.
- SHAP explains the model, not causes; measurement-timing features reflect clinician behaviour.
- The optional GRU comparison was not run.
- **Not a medical device.**

## Documentation

| Document | Contents |
|---|---|
| [`reports/report.md`](reports/report.md) | full technical report (~7,000 words) |
| [`reports/results.md`](reports/results.md) | auto-generated tables and figures |
| [`reports/model_card.md`](reports/model_card.md) | model card (Mitchell et al., 2019) |
| [`reports/slides/slides.md`](reports/slides/slides.md) | 14-slide Marp deck with speaker notes |
| [`reports/demo_script.md`](reports/demo_script.md) | 3½-minute demo video script |
| [`reports/viva_qa.md`](reports/viva_qa.md) | viva defence Q&A |
| [`reports/qa_checklist.md`](reports/qa_checklist.md) | Phase 7 QA checklist |
| [`STATUS.md`](STATUS.md) | task board, verified numbers, decisions |

## Citation

Data: Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Crit Care Med. 2020;48(2):210–217. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation. 2000;101(23):e215–e220.

Author: Amaan Surti — DSML Final Project, NUV, 2026.
