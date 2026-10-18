# STATUS — SepsisSentinel
Last updated: 2026-10-18 by Antigravity

## Current task
Phase 5 — Calibration, alert policy, explainability, and final evaluation

## Task board
| Task | Status | Notes |
|---|---|---|
| P0.T1 Repo skeleton | DONE | Skeleton created, pinned deps installed, ruff/pytest passing |
| P0.T2 STATUS.md | DONE | Initialized from §12 template |
| P0.T3 Git | DONE | Initialized git repository and initial commit |
| P1.T1 Download | DONE | 20,336 Set A and 20,000 Set B files synced and verified |
| P1.T2 io.py | DONE | 40,336 patients, 1,552,210 rows, 2,932 septic (7.27%), 1.80% pos rows |
| P1.T3 cleaning.py | DONE | Cleaned ranges, unit encoded; max loss 0.305% (FiO2) < 1.0% |
| P1.T4 EDA | DONE | All 10 tables and figures generated, adversarial validation AUROC = 1.0000 |
| P2.T1 splits | DONE | 32,268 dev / 8,068 test; 5 strat. folds; cross-hospital folds; all 7 tests pass |
| P2.T2 utility | DONE | Official Challenge scorer reproduced; all 8 hand-verified tests pass |
| P2.T3 metrics | DONE | Full harness: utility, AUROC, AUPRC, Brier, ECE, sens, lead time, FP/day, PPV; 7 tests pass |
| P2.T4 alerting | DONE | k-consecutive + sticky policy; grid search; causality test; 7 tests pass |
| P2.T5 baselines (code) | DONE | B0–B3 coded; B1/B2/B3 run after P3.T2 features are built |
| P3.T1 features | DONE | 298 features vectorised, 1,552,210 rows; all 10 tests pass (causality verified on 20 patients) |
| P3.T2 baselines (run) | DONE | B0: u=0.0000; B1: u=0.0519; B2: u=0.0153; B3 LogReg: AUROC=0.7431, u=0.2528 (thresh 0.72, k=1, s=6) |
| P3.T3 LGBM default 5-fold | DONE | AUROC=0.8446, AUPRC=0.1221, utility=0.4183 (policy: thresh=0.10, k=1, s=0) |
| P3.T4 Optuna | DONE | 40 trials completed over folds [0,1,2]; Best Trial 13 val AUPRC=0.1342 (vs 0.1264 default); params saved |
| P3.T5 LGBM tuned 5-fold | DONE | AUROC=0.8447, AUPRC=0.1223, utility=0.4262 (policy: thresh=0.05, k=1, s=0, sens=86.10%) |
| P3.T6 ablations | DONE | 4 ablations evaluated; no_missingness drops utility from 0.4262 to 0.4039 (table & figure saved) |
| P3.T7 final dev model | DONE | Trained on all 1,241,132 dev rows with tuned params (303 rounds); models/final/ saved |
| P4.T1–T4 cross-hospital | DONE | R0–R3 evaluated; severe transfer drop (A->B: 0.4405->0.2963, B->A: 0.3759->0.3188); table and figures saved |
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
- Patients / rows / septic / % positive rows: 40,336 / 1,552,210 / 2,932 (7.27%) / 1.80%
- Lockbox split: 32,268 dev / 8,068 test (exact 80/20, stratified by hospital × septic)
- Dev folds: 5 × ~6,454 patients each; septic rate per fold within ±0.5pp; hospital share within ±1pp
- Dev OOF (default LGBM): AUROC 0.8446 / AUPRC 0.1221 / utility 0.4183 (sens 83.55%, lead 24.0h, 0.44 FP/day)
- Dev OOF (tuned LGBM): AUROC 0.8447 / AUPRC 0.1223 / utility 0.4262 (sens 86.10%, lead 24.0h, 0.48 FP/day)
- Policy: threshold 0.05 / k 1 / sticky 0 (tuned model)
- Cross-hospital A→B, B→A utility:
  - R0 Full: A internal = 0.4405, B internal = 0.3759, A→B = 0.2963 (Δ -0.1442), B→A = 0.3188 (Δ -0.0571)
  - R1 No Missingness: A int = 0.4231, B int = 0.3940, A→B = 0.2931, B→A = 0.3146
  - R2 Standardized: A int = 0.4349, B int = 0.3546, A→B = 0.2882, B→A = 0.3224
  - R3 No Admin Timing: A int = 0.3821, B int = 0.3737, A→B = 0.2502, B→A = 0.2903
- LOCKBOX TEST: utility [CI] / AUROC / AUPRC / sensitivity / lead time / false alerts per day:

## EDA findings (real verified numbers)
1. **Cohort:** 40,336 unique patients (20,336 in Hospital A, 20,000 in Hospital B) with 1,552,210 hourly observations (790,215 in A, 761,995 in B).
2. **Prevalence & Imbalance:** 2,932 septic patients (7.27% overall; 8.80% in A vs 5.71% in B), with 27,916 positive hourly rows (1.80% overall; 2.17% in A vs 1.41% in B).
3. **Stay Length:** Median ICU stay is 38.0 hours (IQR 23.0h, Q25–Q75: 24.0–47.0h), with septic patients having substantially longer stays.
4. **Demographics:** Median patient age is 63.1 years (Hospital A: 64.7 vs Hospital B: 62.0), and 55.95% of patients are male (A: 58.19%, B: 53.66%).
5. **Cross-Hospital Shift:** Adversarial validation LightGBM model achieves AUROC = 1.0000 predicting hospital source from missingness and mean measurements (top drivers: HCO3 and EtCO2 ordering patterns), demonstrating severe inter-hospital domain divergence.

## Decisions and deviations (with reason)
- Created `tests/test_smoke.py`, `tests/test_io.py`, `tests/test_cleaning.py`, and `tests/test_eda.py` for comprehensive test coverage.
- Used multi-process batching in `sepsis.io` to load 40,336 PSV files in ~17 seconds.
- Physiological sanity ranges applied with NaN replacement and FiO2 percentage scaling.
- Generated `notebooks/01_eda.ipynb` calling `sepsis.eda` functions for reproducible interactive reporting.
- Official utility scorer reproduced verbatim from PLAN.md reference implementation (no download needed).
- B3 logistic regression coded but left to run in P3.T2 once feature matrix is available.
- P3 feature engineering vectorized with zero data leakage (10 tests pass, causality verified on 20 patients).
- Optuna tuned 40 trials yielding Trial 13 best params (AUPRC=0.1342 vs 0.1264 default); tuned 5-fold CV improved utility to 0.4262 and ECE from 0.0412 to 0.0182.
- Ablations confirmed missingness signals contribute significant utility (+0.0223); final dev model trained for 303 rounds on all dev patients.
- **Phase 4 Cross-Hospital Generalization Interpretation (§8 P4.T4):**
  1. *Substantial performance drop on zero-shot transfer:* Normalized utility drops by 32.7% for A→B (0.4405 to 0.2963) and 15.2% for B→A (0.3759 to 0.3188) under full features (R0).
  2. *Severe alerting policy mismatch:* Deploying Hospital B's lower alert threshold (0.03) to Hospital A inflates false alerts to 0.93 alerts/patient-day (nearly 3x Hospital B's internal rate of 0.33), whereas deploying A's policy on B depresses sensitivity to 66.29%.
  3. *Missingness alone does not drive the gap:* Removing missingness features (R1) yields similar transfer utility (A→B: 0.2931, B→A: 0.3146), proving the transfer gap is not merely artifactual from EHR ordering frequencies.
  4. *Target standardization does not rescue utility:* Unsupervised per-hospital z-score normalization (R2) yields 0.2882 (A→B) and 0.3224 (B→A), showing distribution alignment of measured vitals/labs cannot overcome the structural domain shift.
  5. *Administrative timing is informative:* Removing stay timing features (R3: HospAdmTime, ICULOS) harms internal and external performance alike (A internal 0.3821; A→B 0.2502), showing stay progression is critical clinical context.

## Blockers / questions for the human
- None. Ready for Phase 5 (calibration, alert policy, SHAP, subgroups, and single lockbox test).

## Experiment log
| Date | Run | Change | Key metric | Kept? |
|---|---|---|---|---|
| 2026-10-18 | P1.T4 | Adversarial validation (hospital A vs B) | AUROC = 1.0000 | Yes |
| 2026-10-18 | P2 | Evaluation harness + baselines coded | 29 tests pass (splits/utility/metrics/alerting) | Yes |
| 2026-10-18 | P3.T1 | Feature matrix extraction | 298 causal features on 1.55M rows | Yes |
| 2026-10-18 | P3.T2 | Baselines evaluation (B0–B3) | B3 LogReg OOF AUROC=0.7431, utility=0.2528 | Yes |
| 2026-10-18 | P3.T3 | LightGBM default 5-fold CV | OOF AUROC=0.8446, utility=0.4183 (s=0, k=1, th=0.10) | Yes |
| 2026-10-18 | P3.T4 | Optuna tuning (40 trials) | Best Trial 13 val AUPRC=0.1342 (vs 0.1264 default) | Yes |
| 2026-10-18 | P3.T5 | LightGBM tuned 5-fold CV | OOF AUROC=0.8447, utility=0.4262 (s=0, k=1, th=0.05) | Yes |
| 2026-10-18 | P3.T6 | Feature-group ablations | no_missingness utility drops from 0.4262 to 0.4039 | Yes |
| 2026-10-18 | P3.T7 | Final dev model training | 303 rounds on 1,241,132 dev rows, models/final/ saved | Yes |
| 2026-10-18 | P4.T1 | Internal hospital CV (R0) | A internal u=0.4405; B internal u=0.3759 | Yes |
| 2026-10-18 | P4.T2 | External zero-shot transfer (R0) | A->B u=0.2963 (Δ -0.1442); B->A u=0.3188 (FP jumps to 0.93/day) | Yes |
| 2026-10-18 | P4.T3 | Robustness variants (R1, R2, R3) | R1 A->B u=0.2931; R2 A->B u=0.2882; R3 A->B u=0.2502 | Yes |


