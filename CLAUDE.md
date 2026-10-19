# SepsisSentinel — Claude Handover & Working Instructions

## 1. Project Context & Environment
- **Repository:** `D:\NUV SEM 7\DATA SCIENCE AND ML\Final Project`
- **Environment:** Windows, Shell is PowerShell.
- **Python Environment:** Virtual environment is located in `.venv/`. Always invoke Python using:
  ```powershell
  .venv\Scripts\python.exe
  ```
  Or activate the virtualenv before executing commands.
- **Dependencies:** All pinned dependencies (lightgbm 4.5.0, scikit-learn 1.5.1, shap 0.46.0, fastapi 0.115.0, uvicorn, pydantic 2.9.2, pytest, ruff) are installed inside `.venv`.
- **Testing:** `67/67` tests currently pass (53 in `tests/`, 14 in `backend/tests/`):
  ```powershell
  .venv\Scripts\python.exe -m pytest
  ```
- **Linter & Hygiene:** Zero errors:
  ```powershell
  .venv\Scripts\python.exe -m ruff check src tests
  ```

---

## 2. STRICT INVARIANTS (Do Not Break)
1. **LOCKBOX TEST SET IS FROZEN:** The 8,068 lockbox test patients were evaluated **strictly once** in Phase 5.T6 per the build plan. Do **not** retrain models, alter `models/final/`, or re-evaluate the test set to seek higher numbers.
2. **ZERO FEATURE LEAKAGE:** Features are causal (no future data leakage, no lookahead). The feature list (`models/final/feature_list.json`) excludes `patient_id`, `hospital`, `hour`, `SepsisLabel`, `Unit1`, `Unit2`.
3. **REAL VERIFIED NUMBERS ONLY:** All reporting, paper text, slides, and API responses must use the exact empirical numbers in `reports/tables/final_test_metrics.csv` and `STATUS.md`. Never fabricate or approximate metrics.

---

## 3. Current Project State (Phases 0–5 Completed)

| Phase | Description | Status | Key Artifacts |
|---|---|---|---|
| **Phase 0** | Repo skeleton, git repo, dependencies | **DONE** | pyproject.toml, requirements.txt, STATUS.md |
| **Phase 1** | Download, raw I/O, cleaning, EDA | **DONE** | 40,336 patients (1.55M rows); adversarial AUROC = 1.0000; all EDA tables & figures |
| **Phase 2** | Stratified splits, official utility scorer, metrics suite, alert policy search, baselines code | **DONE** | 80/20 split (32,268 dev / 8,068 lockbox); 5 folds; utility function; alerting module |
| **Phase 3** | Feature engineering (298 features), Baselines B0–B3, Default LGBM, Optuna tuning (40 trials), Tuned LGBM, Ablations, Final dev model | **DONE** | `data/processed/features.parquet`, `models/lgbm_tuned/`, `models/final/lgbm.txt` (303 rounds on all dev rows) |
| **Phase 4** | Cross-hospital generalization (Hospital A vs Hospital B, variants R0–R3) | **DONE** | `reports/tables/cross_hospital.csv`, `reports/figures/fig_cross_hospital.png`, `reports/figures/fig_score_distributions_A_vs_B.png` |
| **Phase 5** | Isotonic calibration, alert policy search, SHAP TreeExplainer, subgroup fairness, MLflow logging, Single Lockbox Test | **DONE** | `models/final/` (`calibrator.joblib`, `policy.json`, `threshold_curve.json`), `reports/tables/final_test_metrics.csv`, `reports/figures/` (all test & dev figures) |

### Key Verified Results:
- **Lockbox Test (8,068 patients, 1,000 bootstrap resamples):**
  - **Normalized Utility:** `0.4270` [95% CI: `0.3929` – `0.4604`] (matches dev OOF `0.4267`)
  - **AUROC:** `0.8475` [95% CI: `0.7965` – `0.8990`]
  - **AUPRC:** `0.1369` [95% CI: `0.0854` – `0.2311`] (7.6× over 1.8% baseline prevalence)
  - **Patient Sensitivity:** `87.54%` [95% CI: `84.81%` – `90.13%`]
  - **Median Early Lead Time:** `29.0 hours` [95% CI: `23.0h` – `33.0h`] before onset
  - **False Alert Rate:** `0.5290` alerts/patient-day [95% CI: `0.5123` – `0.5469`] (~1 alarm / 45 hours)
  - **ECE (10 bins):** `0.0027` (perfectly calibrated via isotonic regression)
- **Baseline Comparison on Test Set:**
  - B0 (Never alert): utility = `0.0000`, sensitivity = `0.00%`
  - B1 (SIRS $\ge$ 2): utility = `0.0507`, sensitivity = `82.76%`, false alerts = `1.53/day`
  - B2 (qSOFA $\ge$ 2): utility = `0.0122`, sensitivity = `25.94%`, false alerts = `0.45/day`
  - B3 (Logistic Regression): utility = `0.2438`, AUROC = `0.7367`, sensitivity = `67.06%`
  - **SepsisSentinel:** utility = `0.4270` (+75% relative gain over B3, +742% over SIRS)
- **Cross-Hospital Transfer Drop (Phase 4):**
  - Hospital A Internal: `0.4405` $\rightarrow$ A $\rightarrow$ B External: `0.2963` ($-32.7\%$ drop)
  - Hospital B Internal: `0.3759` $\rightarrow$ B $\rightarrow$ A External: `0.3188` (false alarms spike from 0.33 to 0.93/day)
- **Optimal Alert Policy:**
  - `threshold = 0.0300`, `k = 1`, `sticky = 0` (recorded in `models/final/policy.json`)

---

## 4. What Remains: Phase 7 (Product) and Phase 8 (Report & Viva)

Read [`SepsisSentinel_Build_Plan.md`](file:///d:/NUV%20SEM%207/DATA%20SCIENCE%20AND%20ML/Final%20Project/SepsisSentinel_Build_Plan.md) §11 and §13 for granular task requirements.

### Phase 7: Product Engineering (Dashboard, API & Docker)
1. **P7.T1 Demo Cohort (`src/sepsis/demo_cohort.py`):**
   - Sample exactly 24 ICU beds from **lockbox test** patients with stay 24–96h (seed 42):
     - 8 septic patients detected early (lead time 4–12h)
     - 2 septic patients missed or late
     - 10 non-septic patients with no alerts
     - 4 non-septic patients with $\ge 1$ false alert
   - Output: `data/demo/demo_rows.parquet` (cleaned raw rows), `data/demo/demo_predictions.parquet` (`patient_id, hour, risk, alert, top_reasons_json`), `data/demo/demo_patients.csv` (shuffled beds Bed 01..Bed 24).
2. **P7.T2 FastAPI Backend (`backend/app/`):**
   - Structure: `main.py`, `service.py`, `schemas.py`, `settings.py`.
   - Lifespan loads `models/final/*` and `data/demo/*`.
   - Endpoints:
     - `GET /api/health`
     - `GET /api/config`
     - `GET /api/ward` (`t`, `threshold` $\rightarrow$ 24 beds sorted by risk, WATCH if risk $\ge 0.5\tau$)
     - `GET /api/patients/{id}` (`t`, `threshold`, `show_truth` $\rightarrow$ trajectory vitals, labs, risk, alert, top 5 SHAP reasons)
     - `POST /api/predict` (raw rows $\rightarrow$ clean $\rightarrow$ build features $\rightarrow$ model $\rightarrow$ calibrator $\rightarrow$ policy)
     - `GET /api/metrics` (test metrics, baselines, cross-hospital, threshold curve)
     - `GET /api/model-card`
   - Tests: `backend/tests/test_api.py` with `fastapi.testclient.TestClient`.
3. **P7.T3 Vite React TypeScript Frontend (`frontend/`):**
   - `npm create vite@5.4.8 frontend -- --template react-ts`
   - Structure: `src/api.ts`, `src/state/ReplayContext.tsx`, `src/components/`, `src/pages/` (`WardPage`, `PatientPage`, `ModelPage`), `src/styles.css`.
   - Clinical design palette: background `#F4F6F3`, text `#16303A`, headings `#10242B`, accent teal `#0E7C7B`, alert orange `#D9622B`, watch amber `#B7791F`.
   - Replay clock with Play/Pause, Step $\pm 1$h, speed select, threshold slider (0.05–0.95), "Show ground truth" toggle.
   - Recharts for risk trajectory with onset line and shaded alert zones; vitals line charts; labs point plots; SHAP reasons list.
   - Disclaimer: *"Decision support on retrospective data. Not a medical device."*
4. **P7.T4 Docker Deployment:**
   - `backend/Dockerfile` (python:3.11-slim, libgomp1)
   - `frontend/Dockerfile` (multi-stage: node:20-alpine build $\rightarrow$ nginx:1.27-alpine serve)
   - `docker-compose.yml` (backend on 8000, frontend on 8080)
5. **P7.T5 Documentation Artifacts:**
   - `reports/model_card.md`
   - `reports/qa_checklist.md`

### Phase 8: Reports, Slides, Demo Video & Viva Prep
1. **P8.T1 Results Assembler (`src/sepsis/report.py` $\rightarrow$ `reports/results.md`):**
   - Script that parses all CSVs in `reports/tables/` and figures in `reports/figures/` into a complete Markdown document.
2. **P8.T2 Academic Technical Report (`reports/report.md` or LaTeX/PDF):**
   - Comprehensive thesis/paper covering Abstract, Clinical Background (Sepsis-3), Related Work (PhysioNet 2019), Methodology (Causal features, LightGBM, Isotonic calibration, Alert policy), Empirical Results, Cross-Hospital Transfer Analysis, Ablations, Product Architecture, Limitations, and Ethics.
3. **P8.T3 Presentation Slide Deck (`reports/slides/`):**
   - Slide deck outline/content with verified numbers, charts, and demo flow.
4. **P8.T4 Demo Video Script (`reports/demo_script.md`):**
   - 3-minute clinician-facing walkthrough script.
5. **P8.T5 Viva Defense Q&A Guide (`reports/viva_qa.md`):**
   - Comprehensive technical defense cheat-sheet for oral examination.
6. **P8.T6 Final README.md:**
   - Project overview, installation, reproduction instructions, architecture diagrams.

---

## 5. Startup Validation Checklist for Claude
When starting your session, run:
1. `git status` (confirm working tree is clean)
2. `.venv\Scripts\python.exe -m pytest` (confirm 67/67 tests pass)
3. `.venv\Scripts\python.exe -m ruff check src tests` (confirm 0 errors)
4. View `STATUS.md` for current progress and key numbers.
5. Phases 7 and 8 are complete (2026-10-19); see STATUS.md for remaining human steps.
