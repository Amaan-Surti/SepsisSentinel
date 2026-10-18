# STATUS — SepsisSentinel
Last updated: 2026-10-18 by Antigravity

## Current task
Phase 0 & Phase 1 — COMPLETED (Awaiting kickoff for Phase 2)

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
- Patients / rows / septic / % positive rows: 40,336 / 1,552,210 / 2,932 (7.27%) / 1.80%
- Dev OOF (default LGBM): AUROC / AUPRC / utility:
- Dev OOF (tuned LGBM):
- Policy: threshold / k / sticky:
- Cross-hospital A→B, B→A utility:
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

## Blockers / questions for the human
- None. Phase 0 and Phase 1 completed according to plan.

## Experiment log
| Date | Run | Change | Key metric | Kept? |
|---|---|---|---|---|
| 2026-10-18 | P1.T4 | Adversarial validation (hospital A vs B) | AUROC = 1.0000 | Yes |
