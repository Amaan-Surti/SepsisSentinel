# STATUS — SepsisSentinel
Last updated: 2026-10-18 by Antigravity

## Current task
P1.T3 — cleaning.py — IN PROGRESS

## Task board
| Task | Status | Notes |
|---|---|---|
| P0.T1 Repo skeleton | DONE | Skeleton created, pinned deps installed, ruff/pytest passing |
| P0.T2 STATUS.md | DONE | Initialized from §12 template |
| P0.T3 Git | DONE | Initialized git repository and initial commit |
| P1.T1 Download | DONE | 20,336 Set A and 20,000 Set B files synced and verified |
| P1.T2 io.py | DONE | 40,336 patients, 1,552,210 rows, 2,932 septic (7.27%), 1.80% pos rows |
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
- Patients / rows / septic / % positive rows: 40,336 / 1,552,210 / 2,932 (7.27%) / 1.80%
- Dev OOF (default LGBM): AUROC / AUPRC / utility:
- Dev OOF (tuned LGBM):
- Policy: threshold / k / sticky:
- Cross-hospital A→B, B→A utility:
- LOCKBOX TEST: utility [CI] / AUROC / AUPRC / sensitivity / lead time / false alerts per day:

## Decisions and deviations (with reason)
- Created `tests/test_smoke.py` and `tests/test_io.py` for comprehensive test coverage.
- Used multi-process batching in `sepsis.io` to load 40,336 PSV files in ~17 seconds.

## Blockers / questions for the human
-

## Experiment log
| Date | Run | Change | Key metric | Kept? |
|---|---|---|---|---|
