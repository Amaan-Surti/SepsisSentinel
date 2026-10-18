# SepsisSentinel — Complete Build Plan (Phases 0–8)

**Project:** Real-time ICU sepsis early-warning system on the PhysioNet/Computing in Cardiology Challenge 2019 data.
**Owner:** Amaan (solo, DSML course project). **Builder:** an AI coding agent following this file.
**Rule zero:** this file is the single source of truth. Every number, name, path and parameter below is decided. Do not improvise. If something here is impossible, stop and ask the human (see §0.4).

---

## 0. READ FIRST — instructions for the coding agent

### 0.1 Kickoff prompt (the human pastes this at the start of every new chat)

```
You are the coding agent for the SepsisSentinel project. The full plan is in PLAN.md and
progress is tracked in STATUS.md (both in the repo root; I will paste them if you cannot
read files). Rules:
1. Read STATUS.md first. Continue from the first task that is not marked DONE.
2. Do exactly one task at a time, in order. Follow PLAN.md literally: same file names,
   function names, column names, parameters, seeds.
3. A task is DONE only when every acceptance check listed under it passes. Show me the
   command output that proves it.
4. After each task: update STATUS.md (task status, key numbers, decisions, problems),
   then tell me the exact git commit command to run.
5. Never change a LOCKED decision (PLAN.md §0.3). If you think one is wrong, say why and
   wait for my answer.
6. If something fails 3 times, stop, write the blocker into STATUS.md, and ask me.
7. Give me complete files, never "..." placeholders or "rest unchanged".
Start now: tell me which task you are on and what you will do.
```

### 0.2 Which model to use for which phase

| Phase | Risk if done wrong | Use |
|---|---|---|
| 0 Setup, 1 Data + EDA | Low–medium | Faster model is fine |
| 2 Evaluation harness, 3 Features + LightGBM, 4 Cross-hospital | **High** (silent leakage or wrong scoring ruins every result) | Strongest model you have |
| 5 Calibration, alerts, SHAP | Medium–high | Strongest model |
| 6 GRU (optional) | Medium | Either |
| 7 API + dashboard + Docker | Low (bugs are visible) | Faster model is fine |
| 8 Report + slides | Low | Either |

Switching accounts or models mid-project is safe **because** of STATUS.md: a new chat only needs the kickoff prompt, PLAN.md and STATUS.md.

### 0.3 LOCKED decisions (never change without the human's written OK)

1. Data = public training sets A (20,336 patients) and B (20,000 patients) only. The hidden test set and hospital C do not exist for us.
2. Every split is by **patient**, never by row. One patient's rows are always in exactly one split.
3. Features at hour *t* use only rows with hour ≤ *t* (causal). No backward fill, ever.
4. `hospital` is **never** a model input feature. It is only used for splitting and reporting.
5. Primary metric = official Challenge **normalized utility**, computed with the official scoring function (§2.2).
6. Anything fitted (imputer, scaler, calibrator, threshold, alert policy, early stopping) is fitted on training/dev data only, never on the lockbox test set.
7. Random seed = **42** everywhere (numpy, python `random`, LightGBM, Optuna, PyTorch, sklearn).
8. The lockbox test set (§2.1) is evaluated **once per final model**, in Phase 3.7 and 5.6. No tuning after looking at it.
9. Core model = LightGBM. GRU is a comparison only and is never served by the API.
10. The product is decision support on retrospective data. Never claim clinical validity.

### 0.4 When to stop and ask the human

- Raw data counts do not match §1.2 acceptance checks.
- The official scoring file cannot be obtained and the reference implementation tests (§2.2) fail.
- RAM errors that downcasting (§3.4) does not fix.
- Any result that looks too good: internal AUROC > 0.95 or normalized utility > 0.6. That almost always means leakage. Stop and report.

### 0.5 Coding conventions

- Python 3.11. All code in `src/sepsis/` as importable modules. Notebooks only for EDA plots, and they import from `src/`.
- Every module runnable as `python -m sepsis.<module> [args]` with `argparse`; defaults come from `configs/config.yaml`.
- Use `logging` (INFO), not print, in library code. Log shapes, counts and timings.
- Type hints and a one-line docstring on every public function.
- Store tables as Parquet (`pyarrow`), floats as `float32`.
- Never print whole DataFrames; at most `.head()` or summary stats.
- Format with `ruff format`, lint with `ruff check`.
- Commit after every task: `git commit -m "P<phase>.T<task>: <short summary>"`.

---

## 1. Fixed facts about the data (do not re-derive, but do verify in Phase 1)

- Source: https://physionet.org/content/challenge-2019/1.0.0/ (open access; cite Reyna et al., *Crit Care Med* 2020;48(2):210–217).
- Layout: `training_setA/` (20,336 `.psv` files), `training_setB/` (20,000 `.psv` files). One file per patient, one row per ICU hour, pipe-separated `|`, header row, missing values are the literal string `NaN`.
- 41 columns in this exact order:

```
HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2,
BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride,
Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium,
Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets,
Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS,
SepsisLabel
```

- Groups used throughout (put these lists in `src/sepsis/constants.py` exactly):

```python
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
MEASURED = VITALS + LABS                     # 34 time-varying measurements
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
VITALS_ROLL = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]   # no EtCO2 (almost always missing)
LABS_ROLL = ["Lactate", "WBC", "Creatinine", "Platelets", "BUN",
             "Bilirubin_total", "pH", "HCO3"]
BASELINE_VARS = VITALS_ROLL + ["Lactate", "WBC", "Creatinine", "Platelets"]
SEED = 42
```

- Label semantics: for septic patients, `SepsisLabel` = 1 from **6 hours before** clinical onset *t_sepsis* onward. So **t_sepsis = (index of first row with label 1) + 6**. Non-septic patients are 0 everywhere.
- Expected (verify, do not assume): ~1.55M rows total, ~2,932 septic patients (~7.3%), ~2% of rows positive, stays 8–336 h.

---

## 2. Repository layout (create exactly this in Phase 0)

```
sepsis-sentinel/
├── PLAN.md                     # this file
├── STATUS.md                   # progress log (template in §12)
├── README.md
├── requirements.txt
├── pyproject.toml              # only for ruff + pytest config and src layout
├── .gitignore
├── configs/
│   └── config.yaml
├── data/                       # git-ignored except .gitkeep
│   ├── raw/training_setA/  raw/training_setB/
│   ├── interim/                # all_rows.parquet, clean_rows.parquet
│   ├── processed/              # features.parquet, sequences/
│   ├── splits/                 # lockbox.csv, folds.csv
│   └── demo/                   # demo cohort for the dashboard
├── third_party/
│   └── evaluate_sepsis_score.py   # official scorer (unmodified copy)
├── src/sepsis/
│   ├── __init__.py
│   ├── constants.py   config.py   io.py   cleaning.py   eda.py
│   ├── splits.py      utility.py  metrics.py  alerting.py
│   ├── features.py    baselines.py  train_lgbm.py  tune_lgbm.py
│   ├── cross_hospital.py  calibrate.py  explain.py  final_model.py
│   ├── sequences.py   train_gru.py  ensemble.py
│   ├── demo_cohort.py  report.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 06_gru_colab.ipynb
├── models/                     # git-ignored; trained artifacts
├── reports/
│   ├── figures/   tables/   results.md
├── mlruns/                     # MLflow local store (git-ignored)
├── backend/
│   ├── app/ (main.py, schemas.py, service.py, settings.py)
│   ├── tests/
│   └── Dockerfile
├── frontend/                   # Vite + React + TypeScript
│   └── Dockerfile, nginx.conf
├── tests/                      # pytest for src/sepsis
└── docker-compose.yml
```

---

## 3. Environment

### 3.1 Python (Windows PowerShell; use the equivalent on Mac/Linux)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` (pinned; do not upgrade):

```
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
scipy==1.13.1
scikit-learn==1.5.1
lightgbm==4.5.0
optuna==3.6.1
shap==0.46.0
matplotlib==3.9.2
seaborn==0.13.2
mlflow==2.16.2
pyyaml==6.0.2
joblib==1.4.2
tqdm==4.66.5
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
httpx==0.27.2
pytest==8.3.3
ruff==0.6.9
jupyter==1.1.1
```

PyTorch is **not** installed locally; the GRU runs on Google Colab or Kaggle (Phase 6), which already ship PyTorch.

`pyproject.toml` must declare the `src` layout (`[tool.setuptools.packages.find] where = ["src"]`), project name `sepsis`, `[tool.pytest.ini_options] testpaths = ["tests", "backend/tests"]`, and ruff line length 100.

### 3.2 Node (for Phase 7)

Node 20 LTS. Frontend packages pinned: `react@18.3.1 react-dom@18.3.1 react-router-dom@6.26.2 recharts@2.12.7`, dev: `vite@5.4.8 typescript@5.5.4 @vitejs/plugin-react@4.3.2 @types/react@18.3.11 @types/react-dom@18.3.0`.

### 3.3 Hardware

- Laptop with ≥16 GB RAM recommended. With 8 GB, use the downcasting and negative-subsampling options in §3.4/§6.4.
- If the laptop cannot cope, run Phases 3–5 in a Kaggle notebook (30 GB RAM, free) by uploading `data/processed/features.parquet`.

### 3.4 Memory rules

- All features `float32`; integer flags `int8`; ids as `category`.
- Load only needed columns with `pd.read_parquet(columns=[...])`.

### 3.5 `configs/config.yaml` (create exactly)

```yaml
seed: 42
paths:
  raw_a: data/raw/training_setA
  raw_b: data/raw/training_setB
  all_rows: data/interim/all_rows.parquet
  clean_rows: data/interim/clean_rows.parquet
  features: data/processed/features.parquet
  lockbox: data/splits/lockbox.csv
  folds: data/splits/folds.csv
  models: models
  figures: reports/figures
  tables: reports/tables
split:
  lockbox_frac: 0.20
  n_folds: 5
utility:
  dt_early: -12
  dt_optimal: -6
  dt_late: 3
  max_u_tp: 1.0
  min_u_fn: -2.0
  u_fp: -0.05
  u_tn: 0.0
features:
  vital_windows: [6, 12, 24]
  lab_windows: [24]
  tsl_cap_hours: 72
  baseline_hours: 6
lgbm:
  objective: binary
  learning_rate: 0.03
  num_leaves: 63
  max_depth: -1
  min_child_samples: 200
  feature_fraction: 0.7
  bagging_fraction: 0.8
  bagging_freq: 1
  lambda_l1: 0.0
  lambda_l2: 1.0
  scale_pos_weight: 5.0
  max_bin: 255
  num_boost_round: 5000
  early_stopping_rounds: 200
  metric: average_precision
  verbose: -1
  n_jobs: -1
tuning:
  n_trials: 40
  timeout_hours: 3
  folds_used: [0, 1, 2]
alert_policy:
  threshold_grid_start: 0.01
  threshold_grid_stop: 0.99
  threshold_grid_step: 0.01
  k_consecutive: [1, 2, 3]
  sticky_hours: [0, 6]
gru:
  hidden: 128
  layers: 2
  dropout: 0.2
  input_proj: 128
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  max_epochs: 30
  patience: 5
  grad_clip: 1.0
  focal_gamma: 2.0
  focal_alpha: 0.75
  max_len: 336
```

---

## 4. Phase 0 — Project setup (≈0.5 day)

**P0.T1 Repo skeleton.** Create the tree in §2, `.gitignore` (ignore `.venv/ data/* !data/**/.gitkeep models/ mlruns/ __pycache__/ .ipynb_checkpoints/ frontend/node_modules/ frontend/dist/`), `requirements.txt`, `pyproject.toml`, `configs/config.yaml`, `src/sepsis/constants.py` (§1), `src/sepsis/config.py` (function `load_config(path="configs/config.yaml") -> dict` plus `set_seeds(seed)` that seeds `random`, `numpy`, and sets `PYTHONHASHSEED`).
*Acceptance:* `pip install -e .` succeeds; `python -c "import sepsis, lightgbm, shap; print('ok')"` prints `ok`; `pytest` runs (0 tests is fine); `ruff check src` clean.

**P0.T2 STATUS.md.** Create from the template in §12.
*Acceptance:* file exists with all phases listed as TODO.

**P0.T3 Git.** `git init`, first commit, push to a private GitHub repo (human creates the repo).

---

## 5. Phase 1 — Data loading, cleaning and EDA (≈2 days)

### P1.T1 Download (HUMAN step, agent gives instructions)

Preferred: on https://physionet.org/content/challenge-2019/1.0.0/ use "Download the ZIP file", unzip, and place the folders so the paths are `data/raw/training_setA/*.psv` and `data/raw/training_setB/*.psv`.
Alternative (Git Bash or WSL): `wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/` then move the two folders into `data/raw/`.
*Acceptance:* `python -c "import glob;print(len(glob.glob('data/raw/training_setA/*.psv')),len(glob.glob('data/raw/training_setB/*.psv')))"` prints `20336 20000`.

### P1.T2 `io.py` — build one table

Functions:
- `read_patient(path: Path) -> pd.DataFrame` — `pd.read_csv(path, sep="|")`; checks the 41 columns are exactly §1 in order (raise `ValueError` otherwise).
- `build_all_rows(cfg) -> pd.DataFrame` — loops both folders with `tqdm` (use `concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()-1)` for speed), adds:
  - `patient_id` = file stem (e.g. `p000001`) as string,
  - `hospital` = `"A"` or `"B"` from the folder,
  - `hour` = 0-based row index within the patient (int16).
  Casts measurements to float32, `Gender/Unit1/Unit2/SepsisLabel` to float32 (Unit can be NaN). Sorts by `patient_id, hour`. Writes `data/interim/all_rows.parquet`.
- CLI: `python -m sepsis.io`.

*Acceptance (print and record in STATUS.md):* 40,336 unique patients; patients per hospital 20,336 / 20,000; total rows (expect ≈1.55M); number of septic patients (patients with any label 1, expect ≈2,932); share of positive rows (expect ≈2%); check that `ICULOS` increases by exactly 1 per row within each patient — print the number of patients where it does not (if > 0, log it; `hour` is used everywhere so the pipeline still works).

### P1.T3 `cleaning.py` — physiological sanity ranges

Values outside the range become `NaN` (do **not** clip). Ranges (inclusive):

| Var | Min | Max | | Var | Min | Max |
|---|---|---|---|---|---|---|
| HR | 20 | 300 | | pH | 6.5 | 8.0 |
| O2Sat | 50 | 100 | | Lactate | 0.1 | 30 |
| Temp | 25 | 45 | | WBC | 0.1 | 200 |
| SBP | 40 | 300 | | Platelets | 1 | 2000 |
| MAP | 20 | 250 | | Creatinine | 0.1 | 25 |
| DBP | 10 | 200 | | BUN | 1 | 250 |
| Resp | 1 | 80 | | Bilirubin_total | 0.1 | 80 |
| EtCO2 | 1 | 100 | | Glucose | 10 | 2000 |

Special case **FiO2**: values > 1 and ≤ 100 are percentages → divide by 100; then keep only 0.21–1.0, else NaN.
`Gender`: keep 0/1. `Unit1/Unit2` → new column `unit` (int8): 1 if Unit1==1, 2 if Unit2==1, 0 otherwise; drop Unit1/Unit2 afterwards from features (keep in table).
Function `clean(df) -> (df, report_df)`; `report_df` lists per variable how many values were set to NaN. Save `reports/tables/cleaning_report.csv` and `data/interim/clean_rows.parquet`.
*Acceptance:* report saved; no variable loses more than 1% of its observed values (if one does, stop and ask).

### P1.T4 EDA (`eda.py` functions + `notebooks/01_eda.ipynb` that calls them)

Produce these files exactly (PNG at 150 dpi, CSVs in `reports/tables/`):

1. `dataset_summary.csv` — per hospital and total: patients, rows, septic patients, % septic patients, % positive rows, median/IQR stay length (h), median age, % male.
2. `missingness_by_var.csv` — per variable: % missing overall, in A, in B.
3. `fig_missingness_heatmap.png` — variables (y) × hospital (x), % missing.
4. `fig_measurement_frequency.png` — median measurements per patient-day for each lab, A vs B bars.
5. `fig_stay_length.png` — histogram of stay length by septic status.
6. `fig_onset_timing.png` — histogram of t_sepsis (hours since ICU admission) for septic patients.
7. `fig_trajectories_aligned.png` — for HR, MAP, Resp, Temp, Lactate, WBC: median (and IQR band) of LOCF values for septic patients aligned at t_sepsis (−48 h to +6 h) vs non-septic patients aligned at a random hour (seed 42) of their stay. 6 small panels.
8. `ks_tests_A_vs_B.csv` — two-sample KS statistic and p-value for every MEASURED var (observed values only) A vs B.
9. `adversarial_validation.csv` + `fig_adversarial_top20.png` — train a LightGBM classifier (default params, 200 rounds, patient-level 5-fold) to predict hospital from per-patient summary features (mean of each MEASURED var + % missing of each var). Report AUROC and top-20 importance. High AUROC = strong hospital shift; this motivates Phase 4.
10. `fig_class_imbalance.png` — bar: % positive rows vs % septic patients.

*Acceptance:* all 10 outputs exist; STATUS.md gets a 5-line "EDA findings" summary with the real numbers (these replace the "≈" values on slides).

---

## 6. Phase 2 — Evaluation harness and baselines (≈2 days) — HIGH RISK, use strongest model

### P2.T1 `splits.py`

1. Patient table: one row per patient with `patient_id, hospital, septic (0/1)`. Stratum = `hospital + "_" + str(septic)`.
2. **Lockbox test set:** `train_test_split(patients, test_size=0.20, stratify=stratum, random_state=42)`. Save `data/splits/lockbox.csv` with columns `patient_id, hospital, septic, split` where split ∈ {`dev`, `test`}.
3. **Dev folds:** on dev patients only, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the patient table using `stratum` as y. Save `data/splits/folds.csv` with `patient_id, hospital, septic, fold` (0–4).
4. **Cross-hospital folds** (for Phase 4): for each hospital H separately, 5-fold `StratifiedKFold` on H's patients stratified by `septic`, seed 42. Save into `folds.csv` as extra column `fold_within_hospital`.
5. Helper `get_rows(df, patient_ids)` returns rows of those patients.

*Acceptance tests (`tests/test_splits.py`):* no patient appears in two of {dev, test}; no patient in two folds; septic rate per fold within ±0.5 percentage points of overall; hospital share per fold within ±1 point; re-running produces identical files (hash compare).

### P2.T2 `utility.py` — official scoring

1. Download the official scorer `evaluate_sepsis_score.py` from the Challenge's evaluation code (GitHub repository `physionetchallenges/evaluation-2019`, also linked from the Challenge web page) into `third_party/` unmodified. Import `compute_prediction_utility` from it.
2. If it cannot be downloaded, use this reference implementation (it reproduces the official function):

```python
def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                               max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    labels = np.asarray(labels); predictions = np.asarray(predictions)
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal          # first label index + 6
    else:
        is_septic = False
        t_sepsis = float("inf")
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early); b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal); b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal); b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return np.sum(u)
```

3. Write `normalized_utility(labels_by_patient: dict[str, np.ndarray], preds_by_patient: dict[str, np.ndarray], cfg) -> float`:
   - observed = Σ over patients of `compute_prediction_utility(labels, preds)`;
   - best = Σ of utility with best predictions: zeros, and for septic patients ones on indices `max(0, t_sepsis + dt_early)` up to `min(t_sepsis + dt_late + 1, n)` (exclusive), where `t_sepsis = argmax(labels) + 6`;
   - inaction = Σ of utility with all-zero predictions;
   - return `(observed − inaction) / (best − inaction)`.
4. Also `patient_utilities(...) -> pd.DataFrame` (per-patient observed utility) for debugging.
5. Vectorize only if it is too slow (it takes about a minute on all patients; acceptable).

*Acceptance tests (`tests/test_utility.py`) — all must pass (values verified by hand):*
- Non-septic, n=10, all-zero predictions → utility 0.0.
- Non-septic, n=10, all-one predictions → −0.5.
- Septic, n=40, labels 1 from index 20, all-zero predictions → **−10.0**.
- Same patient, all-one predictions → **6.8**.
- Same patient, prediction 1 only at index 20 (the optimal hour) → contribution at t=20 equals 1.0; total = 1.0 + (FN penalties at t=21..29 = −10.0) → **−9.0**.
- `normalized_utility` for that single septic patient with all-one predictions → **0.96** (= (6.8+10)/(7.5+10)).
- Normalized utility of all-zero predictions on any set = 0.0; of best predictions = 1.0.
- If the official file is present: reference and official functions agree on 1,000 random label/prediction pairs.

### P2.T3 `metrics.py`

Function `evaluate_predictions(df, prob_col, alert_col, cfg) -> dict` where `df` has `patient_id, hour, SepsisLabel, <prob_col>, <alert_col>`. Returns:
- `utility_norm` (§P2.T2),
- `auroc`, `auprc` (row level, `sklearn.metrics.roc_auc_score / average_precision_score` on `prob_col` vs `SepsisLabel`),
- `brier` (row level),
- `ece` (10 equal-width bins),
- `patient_sensitivity`: share of septic patients with ≥1 alert hour in [t_sepsis−12, t_sepsis+3],
- `median_lead_time_h`: over detected septic patients, t_sepsis − first alert hour, considering only alert hours in [t_sepsis−48, t_sepsis+3],
- `false_alerts_per_patient_day`: number of alert **events** (rising edges 0→1, the first hour counts if it is 1) in non-septic patients ÷ (non-septic patient-hours / 24),
- `event_ppv`: share of all alert events that start inside [t_sepsis−12, t_sepsis+3] of a septic patient,
- `alert_hour_rate`: share of hours with alert = 1.

Also `subgroup_metrics(df, ...)` producing the same metrics by hospital, gender, age band (<45, 45–64, 65–79, ≥80) and unit (0/1/2), skipping AUROC when a group has only one class.
*Acceptance tests:* on a synthetic 3-patient example the lead time, event counts and false alerts/day equal hand-computed values (write the example in the test with comments).

### P2.T4 `alerting.py` — turn probabilities into alerts

`apply_policy(df, prob_col, threshold, k, sticky) -> np.ndarray` per patient, in hour order:
1. `raw = prob >= threshold`.
2. **k-consecutive:** alert at t only if `raw` is true for hours t−k+1..t (all within the stay).
3. **sticky:** once the k-rule fires at t, keep alert = 1 for the next `sticky` hours as well (if sticky = 0, nothing extra).
`search_policy(df, prob_col, cfg) -> (best_params, grid_df)`: grid over thresholds 0.01..0.99 step 0.01 × k ∈ {1,2,3} × sticky ∈ {0,6}; maximizes `utility_norm`. Save the full grid as CSV. Ties → choose the higher threshold.
*Acceptance tests:* k and sticky behaviour on a hand-made probability series; alerts never depend on future hours (test: compute on full series and on prefix; prefix results identical).

### P2.T5 `baselines.py`

Requires the scores from features (Phase 3) for B2/B3, so implement now and run after P3.T2. Baselines:
- **B0 Never alert:** utility 0 by definition (sanity check of harness).
- **B1 SIRS rule:** alert when `score__sirs >= 2` (definition in §7.2 F9).
- **B2 Partial-qSOFA rule:** alert when `score__qsofa_partial >= 2`.
- **B3 Logistic regression:** inputs = the 34 LOCF current values + 34 `__meas` flags + `score__sirs`, `score__sofa_partial`, `idx__shock_index`, `static__age`, `static__gender`. Pipeline: `SimpleImputer(strategy="median")` → `StandardScaler()` → `LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=42)`. Evaluated with 5-fold dev OOF; alert policy chosen by `search_policy` on OOF.
Output `reports/tables/baselines_dev_oof.csv` (all §P2.T3 metrics per baseline).
*Acceptance:* B0 utility = 0.0 exactly; B1/B2 utility is a finite number (it may be negative; that is expected and fine); B3 OOF AUROC between 0.70 and 0.85.

---

## 7. Phase 3 — Features and LightGBM (≈3 days) — HIGH RISK, use strongest model

### 7.1 Naming convention (mandatory; the API and SHAP text depend on it)

Double underscore separates the variable from the derived part:
- `HR` → LOCF current value of HR
- `HR__meas`, `HR__tsl`, `Lactate__cnt24`
- `HR__mean_6h`, `HR__min_6h`, `HR__max_6h`, `HR__std_6h`, `HR__slope_6h` (same for 12h, 24h)
- `HR__diff6`, `HR__dbase`
- `idx__shock_index`, `idx__mod_shock_index`, `idx__pulse_pressure`, `idx__bun_cr`, `idx__sf_ratio`, `idx__shock_index__slope_6h`
- `score__sirs`, `score__qsofa_partial`, `score__sofa_partial`, `score__sofa_partial_d24`
- `static__age`, `static__gender`, `static__unit`, `static__hosp_adm_time`, `static__iculos`

### 7.2 `features.py` — exact definitions

All operations are **per patient, in hour order**, using only current and past rows. `x_raw` = cleaned observed value (NaN if not measured that hour). `x_locf` = forward-fill of `x_raw` within the patient (NaN before first observation).

- **F1 LOCF current values** (34): `x_locf` for every MEASURED var, named by the bare var name.
- **F2 Measured flag** (34): `__meas` = 1 if `x_raw` not NaN else 0 (int8).
- **F3 Hours since last measurement** (34): `__tsl` = hour − hour of last observation (0 if measured now); if never measured so far, use the cap; clip to `tsl_cap_hours` = 72. float32.
- **F4 Lab count in last 24 h** (26, LABS only): `__cnt24` = number of observed values in rows t−23..t.
- **F5 Rolling stats on observed values** (not LOCF):
  - VITALS_ROLL × windows {6,12,24} × {mean, min, max, std, slope} = 105 features; `std` needs ≥2 observed values, `slope` needs ≥3, else NaN.
  - LABS_ROLL × window {24} × {mean, min, max, slope} = 32 features; `slope` needs ≥2 observed values.
  - Window w covers rows t−w+1..t (`rolling(w, min_periods=1)` on the per-patient series).
  - **Slope** = least-squares slope of value vs hour over observed points in the window, computed vectorised with rolling sums. With `m` = observed mask, `x` = hour, `y0` = y with NaN→0: `n=Σm, Sx=Σ(x·m), Sy=Σy0, Sxy=Σ(x·m·y0), Sxx=Σ(x²·m)`; `slope = (n·Sxy − Sx·Sy) / (n·Sxx − Sx²)`; NaN when the denominator is 0 or n below the minimum. Units: change per hour.
- **F6 6-hour difference** (7, VITALS_ROLL): `__diff6` = `x_locf(t) − x_locf(t−6)`; NaN for t < 6.
- **F7 Baseline delta** (11, BASELINE_VARS): baseline at time t = mean of observed values in hours 0..min(5, t); if none observed in that span, baseline = first observed value at or before t; otherwise NaN. `__dbase` = `x_locf(t) − baseline(t)`. (This definition is causal: before hour 6 the baseline is built only from data seen so far.)
- **F8 Clinical indices** (on LOCF values; division by 0 or NaN → NaN):
  - `idx__shock_index` = HR / SBP
  - `idx__mod_shock_index` = HR / MAP
  - `idx__pulse_pressure` = SBP − DBP
  - `idx__bun_cr` = BUN / Creatinine
  - `idx__sf_ratio` = O2Sat / FiO2 (FiO2 already as fraction)
  - `idx__shock_index__slope_6h` = slope of shock index over 6 h (computed on hours where both HR and SBP were observed, min 3 points)
- **F9 Clinical scores** (on LOCF values; a missing input counts as 0 points):
  - `score__sirs` (0–4): Temp > 38 or < 36; HR > 90; Resp > 20 or PaCO2 < 32; WBC > 12 or < 4.
  - `score__qsofa_partial` (0–2): Resp ≥ 22; SBP ≤ 100 (mental status is not available).
  - `score__sofa_partial` (0–13): coagulation (Platelets <150→1, <100→2, <50→3, <20→4) + liver (Bilirubin_total 1.2–1.9→1, 2.0–5.9→2, 6.0–11.9→3, ≥12→4) + renal (Creatinine 1.2–1.9→1, 2.0–3.4→2, 3.5–4.9→3, ≥5→4) + cardiovascular (MAP < 70 → 1).
  - `score__sofa_partial_d24` = `score__sofa_partial(t)` − min of `score__sofa_partial` over rows t−23..t.
- **F10 Static** (5): `static__age` = Age, `static__gender` = Gender, `static__unit` = unit (0/1/2; LightGBM categorical), `static__hosp_adm_time` = HospAdmTime, `static__iculos` = hour + 1.
- **Carried columns (not features):** `patient_id, hospital, hour, SepsisLabel`.

Total ≈ 298 features. `build_features(clean_df) -> features_df` must also write `models/feature_list.json` (ordered list of feature names; `hospital`, `patient_id`, `hour`, `SepsisLabel` excluded). CLI: `python -m sepsis.features` → `data/processed/features.parquet`.

Implementation notes: use `groupby("patient_id", sort=False)` with `transform`/`rolling`; avoid Python loops over patients except where unavoidable (baseline: use cumulative sums with a mask for hours ≤5 — vectorisable). Expect 5–20 minutes runtime.

*Acceptance tests (`tests/test_features.py`) — mandatory:*
1. **Causality:** for 20 random patients (seed 42), features computed on the full stay equal features computed on the prefix `rows[:k]` for rows < k (k random per patient), using `np.allclose(..., equal_nan=True)`.
2. Hand-made patient (8 hours, HR = 80, 82, NaN, 90, NaN, NaN, 100, 101): check `HR` LOCF, `HR__meas`, `HR__tsl` (0,0,1,0,1,2,0,0), `HR__mean_6h` at t=5 = mean(80,82,90)=84.0, `HR__slope_6h` at t=7 (window = rows 2..7, observed points (3,90), (6,100), (7,101)) equals `np.polyfit([3,6,7],[90,100,101],1)[0]`.
3. Score checks on hand-made rows: SIRS = 4 for Temp 39, HR 120, Resp 25, WBC 15; SOFA partial = 3+1+0+1 for Platelets 40, Bilirubin 1.5, Creatinine 1.0, MAP 60 → 5.
4. `feature_list.json` has no forbidden names (`hospital, patient_id, SepsisLabel, Unit1, Unit2`).
5. Output dtype float32 for all non-categorical features, row count equals clean_rows.

### 7.3 P3.T2 Run baselines (§P2.T5) now that scores exist.

### 7.4 P3.T3 `train_lgbm.py` — 5-fold dev training with default params

- For fold f in 0..4: train on dev patients with fold ≠ f, validate on fold f. `lgb.Dataset(X, y, categorical_feature=["static__unit"], free_raw_data=False)`. Params from `config.yaml → lgbm` (add `seed: 42`, `deterministic: true`, `force_row_wise: true`). `lgb.train(..., num_boost_round=5000, valid_sets=[val], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])`.
- Collect out-of-fold (OOF) probabilities for all dev rows → `models/lgbm_default/oof.parquet` (`patient_id, hour, SepsisLabel, prob`).
- Save each fold model (`fold{f}.txt`) and best iterations.
- Log to MLflow (`mlflow.set_tracking_uri("mlruns")`, experiment `sepsis`): params, per-fold AUROC/AUPRC, best_iter.
- Run `search_policy` on OOF → report all §P2.T3 metrics (OOF).
*Acceptance:* OOF AUROC in 0.78–0.90; OOF normalized utility in 0.25–0.50; if outside, stop and report (above → suspect leakage).

### 7.5 P3.T4 `tune_lgbm.py` — Optuna

- `optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))`, `n_trials=40`, `timeout=3h`.
- Objective = mean validation AUPRC over folds [0,1,2] (training on the other dev folds), with early stopping as above. (AUPRC is smooth; the threshold is chosen later on utility.)
- Search space:
  - `learning_rate` log-uniform [0.01, 0.1]
  - `num_leaves` int log [15, 255]
  - `min_child_samples` int log [20, 1000]
  - `feature_fraction` [0.4, 1.0]
  - `bagging_fraction` [0.5, 1.0]
  - `lambda_l2` log [1e-3, 10]
  - `scale_pos_weight` log [1, 20]
- Use `optuna.integration` pruning only if it works out of the box; otherwise none.
- Save `models/tuning/best_params.json` and `reports/tables/optuna_trials.csv`.
- Low-RAM option: for tuning only, keep all rows of septic patients and a random 50% of non-septic patients in the training folds (seed 42); never subsample the validation fold.
*Acceptance:* best params saved; best value ≥ default-param value from P3.T3 (if lower, keep defaults and note it).

### 7.6 P3.T5 Retrain 5-fold with tuned params → `models/lgbm_tuned/` (same outputs as P3.T3). Choose the policy on OOF. Compare default vs tuned in `reports/tables/lgbm_dev_oof.csv`.

### 7.7 P3.T6 Feature-group ablations (on dev OOF, tuned params, 5 folds)

Run with these groups removed one at a time: (a) F2+F3+F4 missingness signals, (b) F5 rolling stats, (c) F6+F7 deltas, (d) F8+F9 indices/scores. Report utility/AUROC/AUPRC vs full model in `reports/tables/ablations.csv` and `fig_ablations.png`. This is the evidence for the "missingness is a signal" claim.

### 7.8 P3.T7 Final dev model + one lockbox evaluation (`final_model.py`)

- Train on **all dev patients** with tuned params and `num_boost_round = round(1.1 × mean(best_iteration over 5 folds))`, no early stopping.
- Save `models/final/lgbm.txt`, `feature_list.json`, `params.json`.
- Calibration and policy come from Phase 5 (OOF-based). Evaluate on the lockbox **only after** Phase 5.T1–T2 are done (see §9.6). Until then, do not touch the test set.

---

## 8. Phase 4 — Cross-hospital generalization (≈2 days) — the headline finding

`cross_hospital.py`, tuned params from P3.T4, everything below on **all patients of the named hospitals** (the lockbox split is ignored in this phase, because training and test are different hospitals).

### P4.T1 Internal reference
For H in {A, B}: 5-fold CV within H using `fold_within_hospital`; choose the policy on H's OOF; record OOF metrics. ("How well does it work inside its own hospital?")

### P4.T2 External test
For (train H, test H′) in {(A,B), (B,A)}:
1. Train on all of H with `num_boost_round = round(1.1 × mean best_iter from P4.T1 on H)`.
2. Fit isotonic calibrator and choose the policy on H's OOF from P4.T1 (never on H′).
3. Predict on H′, apply calibrator + policy, compute all metrics.

### P4.T3 Robustness variants (repeat P4.T1 + P4.T2 for each)
- **R0** full features (= P4.T1/T2).
- **R1** without missingness features (no `__meas`, `__tsl`, `__cnt24`): tests whether hospital ordering habits are what fails to transfer.
- **R2** per-hospital standardization: before feature building, z-score each MEASURED raw variable within each hospital using that hospital's own mean/std of observed values (unsupervised, no labels used; allowed for the target hospital). Then rebuild features and repeat.
- **R3** without `static__hosp_adm_time` and `static__iculos` (administrative timing features that may differ by hospital).

### P4.T4 Outputs
- `reports/tables/cross_hospital.csv`: rows = variant × direction (A internal, B internal, A→B, B→A); columns = utility_norm, auroc, auprc, patient_sensitivity, false_alerts_per_patient_day, median_lead_time_h.
- `fig_cross_hospital.png`: grouped bars of utility for internal vs external per variant.
- `fig_score_distributions_A_vs_B.png`: histogram of predicted risk on non-septic patients in each hospital (shows threshold mismatch).
- STATUS.md: 5-line interpretation: size of the drop, which variant helps (if any).
*Acceptance:* table complete with 4 variants × 4 rows; the drop is reported honestly whatever its size.

---

## 9. Phase 5 — Calibration, alert policy, explainability, final evaluation (≈2 days)

### P5.T1 `calibrate.py`
`IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` fitted on the tuned LightGBM dev OOF probabilities vs labels. Save `models/final/calibrator.joblib`. Plot `fig_calibration_dev.png` (reliability diagram before/after, 10 bins, with counts). Report Brier and ECE before/after.

### P5.T2 Final alert policy
Run `search_policy` on **calibrated** OOF probabilities. Save `models/final/policy.json` = `{"threshold": τ, "k": k, "sticky": s, "oof_utility": u}`. Save the full grid `reports/tables/policy_grid.csv` and plot `fig_policy_tradeoff.png`: x = threshold, lines = utility_norm, patient_sensitivity, false_alerts_per_patient_day (second y-axis), for the chosen k/sticky; mark τ.
Also save `models/final/threshold_curve.json` — list of `{threshold, utility_norm, patient_sensitivity, false_alerts_per_patient_day, median_lead_time_h}` for thresholds 0.05..0.95 step 0.01 with the chosen k/sticky (the dashboard reads this).

### P5.T3 `explain.py` — SHAP
- `shap.TreeExplainer(booster)`; sample for global plots: all rows of 300 random septic dev patients + 20,000 random non-septic dev rows (seed 42).
- `fig_shap_beeswarm_top20.png`, `fig_shap_bar_top20.png`, `reports/tables/shap_global.csv` (mean |SHAP| per feature).
- `explain_rows(booster, X_rows, top_n=5) -> list[list[dict]]`, each dict = `{feature, value, shap, text}`, only positive contributions, sorted descending.
- **Human text** (`feature_text(name, value)`): parse the §7.1 naming convention. Var labels:
  `HR "Heart rate", O2Sat "Oxygen saturation", Temp "Temperature", SBP "Systolic BP", MAP "Mean arterial pressure", DBP "Diastolic BP", Resp "Respiratory rate", EtCO2 "End-tidal CO2", Lactate "Lactate", WBC "White cell count", Creatinine "Creatinine", Platelets "Platelets", BUN "Urea (BUN)", Bilirubin_total "Bilirubin", pH "Blood pH", HCO3 "Bicarbonate"` (others: the raw name).
  Templates: current value → "{label} {value:.1f}"; `slope_Xh` → "{label} rising over {X}h" if value > 0 else "{label} falling over {X}h"; `mean/min/max_Xh` → "{label} {stat} over {X}h: {value:.1f}"; `__meas` → "{label} measured this hour"; `__tsl` → "{label} last measured {value:.0f}h ago"; `__cnt24` → "{label} tested {value:.0f}× in 24h"; `__diff6` → "{label} changed {value:+.1f} in 6h"; `__dbase` → "{label} {value:+.1f} vs patient baseline"; indices/scores → fixed phrases (e.g., "Shock index {value:.2f}", "SIRS criteria met: {value:.0f}/4", "Partial SOFA {value:.0f}").
- **Stability check:** top-10 features by mean |SHAP| of each of the 5 fold models; report pairwise Jaccard (mean) in `shap_stability.csv`.
*Acceptance:* unit test that `feature_text` returns a non-empty string for every name in `feature_list.json`.

### P5.T4 Subgroups
`subgroup_metrics` on calibrated dev OOF with the final policy → `reports/tables/subgroups_dev.csv`.

### P5.T5 MLflow
Log the final run with all artifacts in `models/final/`.

### P5.T6 **Lockbox evaluation (once)**
Apply `models/final/lgbm.txt` → calibrator → policy to the test patients. Save `reports/tables/final_test_metrics.csv` (all metrics + 95% bootstrap CI: 1,000 resamples of **patients**, seed 42), `final_test_subgroups.csv`, `fig_roc_pr_test.png`, `fig_calibration_test.png`, and `fig_lead_time_hist_test.png`. Also evaluate B0, B1, B2, B3 on the test set with their dev-chosen policies → `final_test_baselines.csv`.
*Acceptance:* the numbers are recorded in STATUS.md and never "improved" afterwards.

---

## 10. Phase 6 — GRU comparison (optional, ≈2 days, Colab/Kaggle GPU)

### P6.T1 `sequences.py` (runs locally)
Per hour, input vector (in this order, float32):
1. 34 LOCF values, z-scored with mean/std of observed values from **dev training folds only** (fold ≠ 0), then NaN → 0;
2. 34 `__meas` flags;
3. 34 `__tsl` ÷ 72;
4. `score__sirs`/4, `score__sofa_partial`/13;
5. static: `Age/100`, `Gender`, unit one-hot (3), `clip(HospAdmTime, −1000, 0)/1000`, `(hour+1)/336`.
Total = 108 dims. Save ragged format: `X.npy` (all rows, float32), `y.npy`, `offsets.npy` (start index per patient), `patients.csv` (patient_id, split: train = dev folds 1–4, val = dev fold 0, test = lockbox), `norm_stats.json`. Upload the folder to Google Drive.

### P6.T2 `train_gru.py` (run inside `notebooks/06_gru_colab.ipynb`)
- Model: `Linear(108,128) → ReLU → GRU(128, hidden=128, num_layers=2, dropout=0.2, batch_first=True, unidirectional) → Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,1)`. Output logit per hour (causal by construction).
- Batches: 64 patients, sequences truncated to the first 336 hours, padded; a mask excludes padding from the loss.
- **Focal loss** per hour: `p_t = σ(z)` if y=1 else `1−σ(z)`; `α_t = 0.75` if y=1 else `0.25`; `FL = −α_t (1−p_t)^2 log(p_t)`; mean over unmasked hours. (Use `binary_cross_entropy_with_logits(reduction="none")` for numerical stability: `FL = α_t·(1−p_t)^γ·BCE`.)
- Optimizer `AdamW(lr=1e-3, weight_decay=1e-4)`; `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)` on val AUPRC; gradient clipping 1.0; max 30 epochs; early stopping patience 5 on val AUPRC; mixed precision (`torch.autocast` + `GradScaler`) on GPU; seed 42 + `torch.backends.cudnn.deterministic=True`.
- Small α check: also train α ∈ {0.5, 0.9} (same everything else), keep the best val AUPRC. Record all three.
- After training: isotonic calibration and `search_policy` on the **val** fold; then evaluate on the lockbox test set once.
- Save `models/gru/model.pt`, `val_preds.parquet`, `test_preds.parquet`, `metrics.json`.

### P6.T3 `ensemble.py`
Weighted average of calibrated LightGBM and GRU probabilities, weight w ∈ {0.3, 0.4, 0.5, 0.6, 0.7} for LightGBM chosen on the val fold (dev fold 0; use the LightGBM fold-0 OOF predictions). Re-fit the policy on val, evaluate on test once. Output `reports/tables/model_comparison_test.csv` with rows B0–B3, LightGBM, GRU, Ensemble.
**Rule:** the API serves LightGBM regardless; the ensemble is reported only.

---

## 11. Phase 7 — Product: API, dashboard, Docker (≈5 days)

### 11.1 P7.T1 Demo cohort (`demo_cohort.py`)
From the **lockbox test** patients with stay length 24–96 h, after applying the final model + policy, choose (seed 42): 8 septic patients detected early (lead time 4–12 h), 2 septic patients missed or late, 10 non-septic patients without alerts, 4 non-septic patients with at least one false alert. Total 24. Save `data/demo/demo_rows.parquet` (cleaned raw rows), `data/demo/demo_predictions.parquet` (`patient_id, hour, risk, alert, top_reasons_json`) and `data/demo/demo_patients.csv` (`patient_id, display_name "Bed 01".."Bed 24", age, gender, unit, stay_hours, septic, t_sepsis`). Shuffle bed numbers with seed 42 so septic patients are not grouped. The honest mix (misses and false alarms included) is deliberate.

### 11.2 P7.T2 Backend (`backend/app/`)
- `settings.py`: paths to `models/final/*` and `data/demo/*` (env var overrides `MODEL_DIR`, `DEMO_DIR`).
- `service.py`: loads LightGBM booster, calibrator, policy, feature list, threshold curve, demo tables at startup (FastAPI `lifespan`). Builds a `shap.TreeExplainer` once. Functions:
  - `ward_view(t, threshold)`
  - `patient_view(pid, t, threshold, show_truth)`
  - `predict_rows(rows)` — the real online path: cleaning → `build_features` → model → calibrator → policy (uses the same `sepsis` package code).
  - Alerts for a requested threshold are recomputed with `apply_policy` (same k/sticky as `policy.json`) from the stored calibrated risk.
- `main.py` endpoints (all JSON, prefix `/api`):

| Method & path | Query/body | Returns |
|---|---|---|
| `GET /api/health` | – | `{status:"ok", model_version}` |
| `GET /api/config` | – | `{threshold, k, sticky, max_hour, n_patients, model_version, disclaimer}` |
| `GET /api/ward` | `t:int≥0`, `threshold:float` (default policy) | list of `{patient_id, display_name, age, gender, unit, hour, status:"active"|"discharged", risk, risk_trend:[last 12 risks], alert:bool, alert_since:int|null, level:"ALERT"|"WATCH"|"OK"}` sorted by risk desc; `hour = min(t, stay_hours−1)`; WATCH if risk ≥ 0.5×threshold and no alert |
| `GET /api/patients/{id}` | `t`, `threshold`, `show_truth:bool=false` | `{patient fields, hours:[0..h], vitals:{HR,MAP,SBP,Resp,Temp,O2Sat: [values or null]}, labs:{Lactate,WBC,Creatinine,Platelets: [values or null]}, risk:[...], alert:[...], reasons:[{text, feature, value, shap}] (top 5 at hour h), onset_hour (only if show_truth)}` |
| `POST /api/predict` | `{rows:[{hour, <41 raw columns optional>}]}` | `{risk:[...], alert:[...], reasons_last_hour:[...]}` |
| `GET /api/metrics` | – | test metrics table, baselines, cross-hospital table, threshold curve |
| `GET /api/model-card` | – | model card JSON (§11.5) |

- Pydantic v2 models in `schemas.py`; 404 for unknown patient; 422 for invalid params.
- CORS: allow `http://localhost:5173` and `http://localhost:8080`.
- Run: `uvicorn backend.app.main:app --reload --port 8000`.
- Tests (`backend/tests/test_api.py`, `fastapi.testclient`): health 200; ward returns 24 patients sorted by risk; `t` beyond stay → discharged; patient 404; **consistency test**: `POST /api/predict` with a demo patient's raw rows returns the same risks as `demo_predictions.parquet` (abs diff < 1e-6); threshold param changes alerts.

### 11.3 P7.T3 Frontend (`frontend/`, Vite React TS)
- `npm create vite@5.4.8 frontend -- --template react-ts`, then install the pinned packages (§3.2). Vite dev proxy: `/api` → `http://localhost:8000`.
- Structure: `src/api.ts` (typed fetch helpers), `src/state/ReplayContext.tsx`, `src/components/` (`TopBar`, `RiskBar`, `Sparkline`, `LevelChip`, `RiskChart`, `VitalsGrid`, `LabsChart`, `ReasonsPanel`, `Disclaimer`), `src/pages/` (`WardPage`, `PatientPage`, `ModelPage`), `src/styles.css`.
- **Replay clock lives in the frontend** (`ReplayContext`): state `{t, playing, speedMs (1000|500|250), threshold, showTruth}`; when playing, `t += 1` every `speedMs` until `max_hour` from `/api/config`; controls: Play/Pause, Step +1h, Step −1h, Reset, speed select, threshold slider (0.05–0.95, step 0.01, initial = policy threshold), "Show ground truth (demo)" toggle.
- **Ward page (`/`)**: table columns Bed, Age/Sex, Unit (MICU/SICU/–), ICU hour, Risk (bar + %), Trend (12-h sparkline), Level chip (ALERT orange `#D9622B` with text "Alert", WATCH amber `#B7791F` text "Watch", OK grey text "OK"), Alert since ("for 3 h"). Row click → patient page. Discharged rows greyed and sorted last.
- **Patient page (`/patient/:id`)**: header (bed, age, sex, unit, ICU hour, level chip); `RiskChart` (Recharts `ComposedChart`: risk line, dashed threshold line, orange shaded alert hours, vertical onset line labelled "Sepsis onset" only when show-truth is on); `VitalsGrid` = 5 small line charts (HR, MAP, Resp, Temp, O2Sat) sharing the x-axis; `LabsChart` = dot plots for Lactate, WBC, Creatinine, Platelets (points only where measured); `ReasonsPanel` = "Why this risk" list of up to 5 reasons with ▲/▼ glyph plus text; footer disclaimer "Decision support on retrospective data. Not a medical device."
- **Model page (`/model`)**: test metrics table with CIs; baselines comparison; cross-hospital table; trade-off chart from `threshold_curve` (utility, sensitivity, false alerts/day vs threshold) with a marker at the current slider threshold; model card text.
- Fetching: on every `t`/threshold change, fetch ward (and the open patient); debounce slider by 200 ms; show loading/error states ("Cannot reach the API at /api — is the backend running?").
- Style: background `#F4F6F3`, text `#16303A`, headings `#10242B`, accent teal `#0E7C7B`, alert orange `#D9622B`; font "IBM Plex Sans" (Google Fonts link in `index.html`) with Arial fallback; status never shown by colour alone (always text); keyboard focus visible; layout works at 1280×720 and 1920×1080.
- *Acceptance:* `npm run build` succeeds with no TypeScript errors; manual QA checklist (§11.6) passes.

### 11.4 P7.T4 Docker
- `backend/Dockerfile`: `python:3.11-slim`; `apt-get install -y --no-install-recommends libgomp1`; copy `requirements.txt`, `src/`, `backend/`, `models/final/`, `data/demo/`; `pip install -r requirements.txt && pip install -e .`; `CMD uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`.
- `frontend/Dockerfile`: stage 1 `node:20-alpine` → `npm ci && npm run build`; stage 2 `nginx:1.27-alpine` serving `dist/`, `nginx.conf` proxies `/api/` to `http://backend:8000/api/` and falls back to `index.html` for client routes.
- `docker-compose.yml`: services `backend` (port 8000 internal), `frontend` (port `8080:80`, depends_on backend).
- Windows: Docker Desktop with WSL2 backend (human installs).
- *Acceptance:* `docker compose up --build` → http://localhost:8080 shows the ward; replay works end to end.

### 11.5 Model card (`reports/model_card.md`, also served by the API)
Sections: model details (LightGBM version, params, features count, training data = dev split A+B), intended use (research/education demo; decision support only), not intended for (clinical use, unseen hospitals without local validation), metrics (test + CIs + cross-hospital), subgroup results, calibration, known limitations (§13.4), ethics (de-identified public data, licence/citation).

### 11.6 Manual QA checklist (agent writes it to `reports/qa_checklist.md`; human ticks it)
1. Ward loads with 24 beds sorted by risk at t=0. 2. Play advances hours; risks change. 3. A septic demo patient shows an alert before the onset line (show-truth on). 4. Slider lower → more alerts; higher → fewer. 5. Patient page charts align on the same hours. 6. Reasons panel updates when moving hour. 7. Discharged patients greyed. 8. Backend stopped → clear error message. 9. Model page numbers equal `final_test_metrics.csv`. 10. Works in Chrome at 1280×720.

---

## 12. STATUS.md template (create in P0.T2; update after every task)

```markdown
# STATUS — SepsisSentinel
Last updated: <date> by <model name>

## Current task
P?.T? — <name> — IN PROGRESS / BLOCKED

## Task board
| Task | Status | Notes |
|---|---|---|
| P0.T1 Repo skeleton | TODO | |
| P0.T2 STATUS.md | TODO | |
| P0.T3 Git | TODO | |
| P1.T1 Download | TODO | |
| P1.T2 io.py | TODO | |
| P1.T3 cleaning.py | TODO | |
| P1.T4 EDA | TODO | |
| P2.T1 splits | TODO | |
| P2.T2 utility | TODO | |
| P2.T3 metrics | TODO | |
| P2.T4 alerting | TODO | |
| P2.T5 baselines (code) | TODO | |
| P3.T1 features | TODO | |
| P3.T2 baselines (run) | TODO | |
| P3.T3 LGBM default 5-fold | TODO | |
| P3.T4 Optuna | TODO | |
| P3.T5 LGBM tuned 5-fold | TODO | |
| P3.T6 ablations | TODO | |
| P3.T7 final dev model | TODO | |
| P4.T1–T4 cross-hospital | TODO | |
| P5.T1 calibration | TODO | |
| P5.T2 policy | TODO | |
| P5.T3 SHAP | TODO | |
| P5.T4 subgroups | TODO | |
| P5.T5 MLflow | TODO | |
| P5.T6 LOCKBOX TEST (once) | TODO | |
| P6.T1–T3 GRU + ensemble (optional) | TODO | |
| P7.T1 demo cohort | TODO | |
| P7.T2 backend | TODO | |
| P7.T3 frontend | TODO | |
| P7.T4 docker | TODO | |
| P8.T1–T5 report, slides, demo, viva | TODO | |

## Key numbers (fill as they appear)
- Patients / rows / septic / % positive rows:
- Dev OOF (default LGBM): AUROC / AUPRC / utility:
- Dev OOF (tuned LGBM):
- Policy: threshold / k / sticky:
- Cross-hospital A→B, B→A utility:
- LOCKBOX TEST: utility [CI] / AUROC / AUPRC / sensitivity / lead time / false alerts per day:

## Decisions and deviations (with reason)
-

## Blockers / questions for the human
-

## Experiment log
| Date | Run | Change | Key metric | Kept? |
|---|---|---|---|---|
```

---

## 13. Phase 8 — Report, slides, demo, viva (≈3 days)

### P8.T1 `report.py` → `reports/results.md`
Auto-generates one Markdown file assembling every table in `reports/tables/` and figure in `reports/figures/` with captions, in the order of the report structure below. Re-run whenever results change.

### P8.T2 Final report (human writes with agent help; 15–25 pages)
1. Abstract (150 words, real numbers). 2. Introduction: sepsis, Sepsis-3, why early warning, alarm fatigue. 3. Related work: the 2019 Challenge (Reyna et al.), gradient boosting winners, the hidden-hospital generalization failure, SHAP (Lundberg & Lee 2017), LightGBM (Ke et al. 2017), focal loss (Lin et al. 2017). 4. Data: source, cohort table, missingness, label construction, A vs B shift (EDA figures). 5. Methods: splits (lockbox + folds), utility score, features (§7.2 table), models, tuning, calibration, alert policy, SHAP, cross-hospital design. 6. Results: baselines vs LightGBM (dev OOF and test), ablations, cross-hospital table, calibration, lead time, subgroups, GRU comparison. 7. Product: architecture diagram, screenshots of ward/patient/model pages. 8. Discussion: what the drop across hospitals means, limitations (§13.4), what real deployment would need. 9. Conclusion. 10. References. Appendix: full feature list, hyperparameters, policy grid.

### P8.T3 Slides update
Replace estimates in the existing deck: slide "By the numbers" (exact rows, septic patients, % positive rows), "Missing values" (swap the illustrative grid for the real missingness figure), "One patient's stay" (use a real demo patient's charts), "Evaluation" (replace targets with test results + CIs), add slides: "Results vs baselines", "Cross-hospital finding", "Ablation: missingness is a signal", "Live demo".

### P8.T4 Demo script (`reports/demo_script.md`, 3 minutes)
Start compose → open ward at t=0 → play at 500 ms → pause when the first septic bed turns ALERT → open patient, read the reasons → toggle show-truth to reveal onset after the alert → move threshold slider to show the false-alarm trade-off on the Model page → close with the decision-support disclaimer. Include a fallback: a 60-second screen recording (human records it) in case the laptop fails.

### P8.T5 Viva preparation (`reports/viva_qa.md`)
Answers with the real numbers to: why not accuracy; how leakage was prevented (show the causality test); why missingness is informative (show ablation); why LightGBM over deep learning (show GRU comparison); what happens on a new hospital (show Phase 4); why AUPRC looks low (chance level = positive rate); why 6-hour-early labels; how calibration works; what SHAP shows and its limits; what deployment would require.

### 13.4 Limitations to state everywhere
Retrospective, de-identified data; labels from Sepsis-3 criteria applied to records, not perfect ground truth; no hospital C, so external validity is only simulated with A↔B; thresholds trade missed cases against alarm fatigue; SHAP explains the model, not causes; not a medical device.

---

## 14. Effort estimate (focused working days)

| Phase | Days |
|---|---|
| 0 Setup | 0.5 |
| 1 Data + EDA | 2 |
| 2 Harness + baselines | 2 |
| 3 Features + LightGBM | 3 |
| 4 Cross-hospital | 2 |
| 5 Calibration, policy, SHAP, test | 2 |
| 6 GRU (optional) | 2 |
| 7 Product | 5 |
| 8 Report, slides, demo | 3 |
| **Total** | **≈21.5 (≈19.5 without GRU)** |

If time runs short, cut in this order: Phase 6 → P4.T3 variants R2/R3 → Model page extras → Docker (run with two terminals instead). Never cut: P2 harness, P3 features + tuned LightGBM, P4.T1–T2, P5, a working ward + patient page.

---

## 15. Final checklist before submission

- [ ] `pytest` all green; `ruff check` clean.
- [ ] Causality test passes; no `hospital` in `feature_list.json`.
- [ ] Lockbox evaluated exactly once; numbers in STATUS.md match `final_test_metrics.csv`.
- [ ] `reports/results.md` regenerated after the last change.
- [ ] `docker compose up --build` works on a clean clone.
- [ ] README has: what it is, setup, how to download data, how to reproduce each phase (commands), how to run the demo, citation of the dataset.
- [ ] Slides and report use the same numbers.
