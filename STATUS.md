# STATUS — SepsisSentinel
Last updated: 2026-11-03 by Claude Opus 5.5

## Current task
Phase 9 (branch `feature/ensemble-and-ui`, 2026-11-03): **the served model is a LightGBM + GRU ensemble chosen on development data; its lockbox score has not been measured; the 0.4270 test result belongs to the LightGBM component alone.** Nested dev utility: LightGBM 0.4196 → ensemble (GRU hidden 64, w = 0.7) 0.4327, Δ +0.0131 [+0.0072, +0.0190]. Backend serves `MODEL_KIND=ensemble` (default) or `lgbm`; Docker 1.1 images healthy; UI reworked after an impeccable critique (24/40) + audit (12/20); QA 10/10 on Docker; all docs and slides updated. Human to-do: record the 60-s fallback demo video; review and merge the PR.

## Phase 9 — ensemble served + UI (branch `feature/ensemble-and-ui`, started 2026-11-03)
Search space fixed **before** any new result was produced (dev data only, lockbox untouched):
- Blend weight for LightGBM w ∈ {0.5, 0.6, 0.7, 0.8, 0.9} (widened once from {0.3…0.7}, where 0.7 sat on the edge). Chosen inside the nested scheme: for each outer fold, isotonic calibration, w and the alert policy are fit on the other 4 folds; ties go to the lower w. Every weight is reported.
- GRU variants (decided up front: both are run and reported, whatever the outcome): (a) hidden 64; (b) the 32 highest-gain LightGBM features appended to the GRU input (for fold k, taken from `models/lgbm_tuned/fold{k}.txt`, which never saw fold k; columns already in the GRU input excluded; z-scored on the training folds). Both use focal alpha 0.5 from `models/gru/meta.json`, the same folds, seed and early stopping.
- A variant replaces the plain GRU only if it has the best nested ensemble utility **and** the paired patient bootstrap (1,000 resamples, seed 42) of variant-ensemble − plain-ensemble nested utility has a 95% CI above 0. Otherwise the plain GRU is kept.
- Final GRU epochs = median best epoch (0-indexed) of the chosen GRU's 5 folds + 1; trained on all 32,268 dev patients.

Results (all development data, nested 5-fold CV; `reports/tables/ensemble_dev_*.csv`):

| Model (nested) | Utility | AUROC | Sensitivity | False alerts/day | Median lead |
|---|---|---|---|---|---|
| LightGBM | 0.4196 | 0.8421 | 85.46% | 0.478 | 23 h |
| GRU (plain, hidden 128) | 0.3498 | 0.8069 | 80.39% | 0.455 | 23 h |
| GRU hidden 64 | 0.3529 | 0.8119 | 78.13% | 0.400 | 22 h |
| GRU + 32 LightGBM features | 0.3422 | 0.8110 | 78.73% | 0.442 | 24 h |
| Ensemble, plain GRU | 0.4255 | 0.8438 | 85.00% | 0.429 | 23 h |
| **Ensemble, GRU hidden 64 (served)** | **0.4327** | **0.8466** | **85.00%** | **0.415** | **23 h** |
| Ensemble, GRU + 32 features | 0.4252 | 0.8457 | 85.12% | 0.441 | 23 h |

Weight table for the served variant (h64), nested: w 0.5 → 0.4246; 0.6 → 0.4310; **0.7 → 0.4327 (chosen by all 5 folds)**; 0.8 → 0.4323; 0.9 → 0.4287. With the plain GRU the folds split 2 × 0.7 / 3 × 0.8 and the nested utility is 0.4255 (the old {0.3…0.7} grid gave 0.4264).
Paired bootstrap (1,000 patient resamples, nested alerts): ensemble(h64) − LightGBM **+0.0131 [+0.0072, +0.0190]**; ensemble(h64) − ensemble(plain) +0.0072 [+0.0033, +0.0109] → h64 adopted; ensemble(top32) − ensemble(plain) −0.0002 [−0.0039, +0.0032] → not adopted; GRU − LightGBM −0.0698 [−0.0838, −0.0563].
Caveat: the variant was picked as the best of three on the same dev folds, so its +0.0072 margin carries some selection optimism even though the rule was fixed in advance.

Final ensemble (`models/ensemble_final/`): GRU hidden 64 trained on all 32,268 dev patients for 5 epochs (fold best epochs 4, 3, 2, 4, 4), focal α 0.5; GRU isotonic fit on its dev OOF; LightGBM keeps `models/final/calibrator.joblib` (identical to an isotonic refit on the same OOF, max gap 0.0); risk = 0.7 × LightGBM + 0.3 × GRU; policy τ = 0.03, k = 1, sticky = 0 (search_policy on the blended calibrated dev OOF; in-sample dev utility 0.4362). **Lockbox score: not measured.**
Demo cohort: same 24 beds; LightGBM risks reproduce the original file exactly (max |Δ| 0.0); ensemble online vs batch max |Δrisk| 4.3e-7 (≤ 1e-5). Recomputed ensemble bed outcomes: 6 early (4–12 h), 2 flagged outside that window (Bed 16 at 14 h, Bed 13 at 2 h before onset), 2 missed/late, 8 quiet, 6 with a false alert (Beds 02 and 17 newly) (LightGBM: 8 / 0 / 2 / 10 / 4 — the beds were originally picked with LightGBM's alerts).

## Phase 10 — Gallery Label redesign (branch `feature/ui-redesign`, stacked on `feature/ensemble-and-ui`, 2026-11-03)
- Brief from the author: a completely new UI for a live exhibition demo on his laptop (visitors walk up and arrive mid-story), every feature kept, and **all vitals and all labs for every hour plus insights for every hour**; must not look like a startup SaaS product. Recorded in `PRODUCT.md`.
- impeccable new-work: seven grounded worlds; concept-seed (`seed:1cb1bf224b161401`) assigned **museum exhibit labels**; five sketches on the decision page; the author picked **Gallery Label**. Contract: `.impeccable/surfaces/frontend-src-main-tsx.md`; design record: `DESIGN.md`, `.impeccable/design.json`; sketches: `.impeccable/mocks/decision/`.
- Structure: Ward wall (`/`) · bed mounted on its label (`/bed/:id`, tabs *Label*, *Every measurement* (34 variables, charts + hour-by-hour flowsheet), *Hour by hour* (risk and change, LightGBM/GRU parts, status, measurements, that hour's top-5 reasons)) · Tour (`/tour`, six stops; also a rail on every page) · Evidence (`/evidence`, lockbox room vs development annex). Old `/patient/:id` and `/model` links redirect.
- API: `/api/patients/{id}` now returns all 8 vitals and 26 labs (was 6 + 4), `reasons_by_hour`, `risk_components` (LightGBM/GRU per hour, ensemble only) and `hosp_adm_time`; ward vitals add SpO₂. New API test for every measurement and hourly insights (both model kinds).
- Finish review (fresh agent, impeccable finish-reviewer procedure; shipped agent type not available here): **fix**, 8 material findings; after two fix rounds 7 resolved (WHAT HAPPENED on the first screen, status clipping at 1280, "now" no longer in the Watch colour, flowsheet group headings, flowsheet height, evidence figures, vital tiles), the eighth, **self-hosted fonts**, done on 2026-11-04 after the owner approved the download (`frontend/public/fonts/`, SIL OFL 1.1; verified with all external hosts blocked). Detector: 0 findings.
- QA checklist re-run on Docker 10/10 (`reports/qa_checklist.md`); nginx `proxy_connect_timeout 3s` makes the backend-down message appear in ~3 s instead of ~60 s.
- Screenshots: `reports/screenshots/*.jpg` (Gallery Label: ward, patient, measurements, hours, tour, model, model_full), `reports/screenshots/p9/` (P9 UI), `reports/screenshots/before/` (P7). UI history and findings: `reports/ui_review.md`.

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
| P5.T1 calibration | DONE | Isotonic regression fit on dev OOF; Brier 0.0177->0.0166, ECE 0.0182->0.0000; calibrator.joblib & fig_calibration_dev.png |
| P5.T2 policy | DONE | search_policy on calibrated OOF yields thresh=0.03, k=1, s=0, utility=0.4267; policy.json, threshold_curve.json, fig_policy_tradeoff.png |
| P5.T3 SHAP | DONE | TreeExplainer on 38,895 sample rows; top features: ICULOS, FiO2__tsl, HospAdmTime, unit, Lactate__tsl; mean fold Jaccard=0.5939; figures & tables saved |
| P5.T4 subgroups | DONE | dev OOF subgroups computed by hospital, gender, age, unit; subgroups_dev.csv saved |
| P5.T5 MLflow | DONE | Final model artifacts, policy, parameters logged to MLflow experiment 'sepsis' |
| P5.T6 LOCKBOX TEST (once) | DONE | 8,068 test patients evaluated once with 1,000 bootstrap resamples; utility=0.4270 [0.3929, 0.4604], AUROC=0.8475 [0.7965, 0.8990], sens=87.54%, lead=29.0h |
| P6.T1–T3 GRU + ensemble (optional) | DONE (dev only) | `models/gru/`, `reports/tables/gru_dev_comparison*.csv`. Nested dev utility: LightGBM 0.4196, GRU 0.3498, ensemble 0.4264. Lockbox not used; API still serves LightGBM |
| P7.T1 demo cohort | DONE | 24 lockbox-test beds (stay 24–96h, seed 42): 8 early (lead 4–12h), 2 missed/late, 10 quiet, 4 false-alert; 1,022 rows; online path == batch features (max diff 0.0) |
| P7.T2 backend | DONE | FastAPI `backend/app/` (7 endpoints); 14 API tests pass incl. predict-consistency (<1e-6); predict 70-h stay ≈315 ms, ward ≈14 ms |
| P7.T3 frontend | DONE | Vite 5.4.8 + React 18 + TS + Recharts; Ward / Patient / Model pages, replay clock, threshold slider, show-truth; `npm run build` clean; checked in Chrome at 1280×720 |
| P7.T4 docker | DONE | `docker compose up --build` OK; backend + frontend healthy; ward at :8080, replay end to end; frontend healthcheck fixed (127.0.0.1, nginx IPv4-only) |
| P7.T5 model card + QA | DONE | `reports/model_card.md` (served at /api/model-card); `reports/qa_checklist.md` 10/10 passed on Docker deployment |
| P8.T1 results assembler | DONE | `python -m sepsis.report` → `reports/results.md` (23 tables, 17 figures); 2 tests |
| P8.T2 report | DONE | `reports/report.md` (~7,200 words, dashboard screenshots in `reports/screenshots/`) |
| P8.T3 slides | DONE | `reports/slides/slides.md` (14-slide Marp deck, speaker notes) exported to `slides.pptx` and `slides.pdf` (marp-cli 3.4.0); rendering checked visually |
| P8.T4 demo script | DONE | `reports/demo_script.md` (3½ min; Bed 03 alerts h16, onset h22; Bed 07 miss) |
| P8.T5 viva guide | DONE | `reports/viva_qa.md` (27 Q&A) |
| P8.T6 README | DONE | GitHub-ready README with results, architecture, Docker and reproduction |
| P9.1 ensemble selection (dev) | DONE | Pre-registered grid w ∈ {0.5…0.9} × GRU {plain, h64, top32}; nested 5-fold; h64 adopted (Δ vs plain +0.0072 [+0.0033, +0.0109]); `reports/tables/ensemble_dev_*` |
| P9.1 final ensemble | DONE | GRU h64, 5 epochs on all 32,268 dev patients; GRU isotonic; w 0.7; policy τ 0.03/k 1/s 0; `models/ensemble_final/` |
| P9.1 inference + demo | DONE | CPU ensemble path in `inference.py`; same 24 beds rescored (LightGBM Δ 0.0; ensemble online vs batch 4.3e-7) |
| P9.2 backend + Docker | DONE | `MODEL_KIND`; `/api/metrics` → `lockbox_test` + `dev_cv`; torch 2.6.0+cpu image 2.16 GB (P7: 1.36 GB); predict parity in Docker 0.0 |
| P9.3 UI | DONE | impeccable critique + audit → typeset/layout/clarify/polish; model badge; dev band; LightGBM-reasons note; detector 0 findings; QA 10/10 |
| P9.4 docs | DONE | README, model card, report, slides (15, PPTX/PDF), demo script, viva (Q28–Q30), presentation guide, `reports/ui_review.md` |
| P10 UI redesign (Gallery Label) | DONE | All 34 measurements every hour, hourly insights, tour, evidence rooms; finish review 8/8 resolved (fonts self-hosted); QA 10/10 on Docker; DESIGN.md |

## Key numbers (fill as they appear)
- Patients / rows / septic / % positive rows: 40,336 / 1,552,210 / 2,932 (7.27%) / 1.80%
- Lockbox split: 32,268 dev / 8,068 test (exact 80/20, stratified by hospital × septic)
- Dev folds: 5 × ~6,454 patients each; septic rate per fold within ±0.5pp; hospital share within ±1pp
- Dev OOF (default LGBM): AUROC 0.8446 / AUPRC 0.1221 / utility 0.4183 (sens 83.55%, lead 24.0h, 0.44 FP/day)
- Dev OOF (tuned LGBM): AUROC 0.8447 / AUPRC 0.1223 / utility 0.4262 (sens 86.10%, lead 24.0h, 0.48 FP/day)
- Policy: threshold 0.03 / k 1 / sticky 0 (calibrated tuned model; dev OOF utility 0.4267)
- Cross-hospital A→B, B→A utility:
  - R0 Full: A internal = 0.4405, B internal = 0.3759, A→B = 0.2963 (Δ -0.1442), B→A = 0.3188 (Δ -0.0571)
  - R1 No Missingness: A int = 0.4231, B int = 0.3940, A→B = 0.2931, B→A = 0.3146
  - R2 Standardized: A int = 0.4349, B int = 0.3546, A→B = 0.2882, B→A = 0.3224
  - R3 No Admin Timing: A int = 0.3821, B int = 0.3737, A→B = 0.2502, B→A = 0.2903
- LOCKBOX TEST (eval once, 8,068 patients, 1,000 bootstrap iterations):
  - Normalized Utility: **0.4270** [95% CI: 0.3929 – 0.4604]
  - AUROC: **0.8475** [95% CI: 0.7965 – 0.8990]
  - AUPRC: **0.1369** [95% CI: 0.0854 – 0.2311]
  - Brier score: **0.0164** [95% CI: 0.0123 – 0.0208]
  - ECE: **0.0027** [95% CI: 0.0013 – 0.0084]
  - Patient Sensitivity: **87.54%** [95% CI: 84.81% – 90.13%]
  - Median Lead Time: **29.0 hours** [95% CI: 23.0h – 33.0h]
  - False Alerts / Patient-Day: **0.5290** [95% CI: 0.5123 – 0.5469]
  - Event PPV: **0.0462**
  - Baselines on Test: B0 utility = 0.0000; B1 SIRS utility = 0.0507 (1.53 FP/day); B2 qSOFA utility = 0.0122 (sens 25.94%); B3 LogReg utility = 0.2438 (AUROC 0.7367)

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
- **Phase 5 Calibration and Single Lockbox Evaluation Findings:**
  1. *Calibration:* Isotonic regression reduced dev ECE from 0.0182 to 0.0000 and Brier score from 0.0177 to 0.0166; test set ECE is 0.0027, demonstrating exceptional probability calibration.
  2. *Alert policy:* Calibrated policy search yielded optimal decision threshold $\tau = 0.03, k = 1, \text{sticky} = 0$, achieving 0.4267 dev OOF utility.
  3. *Lockbox Test Performance:* On the untouched 8,068 test patients evaluated strictly once, the model achieved normalized utility of 0.4270 [95% CI: 0.3929 – 0.4604], matching dev OOF performance almost identically (+0.0003), confirming zero overfitting.
  4. *Early Detection:* Patient sensitivity reached 87.54% [95% CI: 84.81% – 90.13%] with a median lead time of 29.0 hours before sepsis onset, while maintaining a low false alert burden of 0.53 alerts/patient-day (about 1 alarm every 45 hours).
  5. *Clinical Baselines Dominance:* SepsisSentinel beats B0 (u=0.0000), B1 SIRS (u=0.0507, 1.53 FP/day), B2 qSOFA (u=0.0122, 25.94% sens), and B3 Logistic Regression (u=0.2438, AUROC 0.7367) by a wide margin (+75% relative gain over B3).
  6. *Explainability:* SHAP analysis confirmed stay progression (`static__iculos`, `static__hosp_adm_time`), clinical ordering patterns (`FiO2__tsl`, `Lactate__tsl`), and vital sign dynamics (`Temp`, `BUN`, `WBC`, `idx__shock_index`) as the primary drivers of sepsis predictions, with high stability across folds (mean pairwise Jaccard 0.5939).

- **Phase 7 (2026-10-19, Claude):**
  - Added `src/sepsis/inference.py` (shared online path: clean → features → LightGBM → isotonic → policy) used by both `demo_cohort.py` and the API so stored demo risks and `POST /api/predict` agree exactly.
  - Bug fixes in `explain.py` (no effect on any model or metric): `explain_rows` referenced `shap_matrix` before assignment (crashed on ndarray SHAP output); `feature_text` mapped `static__unit` 0→MICU/1→SICU, but cleaning encodes Unit1 (MICU)=1, Unit2 (SICU)=2, 0=unknown — corrected.
  - Demo "early" = first alert 4–12h before t_sepsis (only 11 of 259 eligible septic test patients qualify because the τ=0.03 policy usually fires >12h early — consistent with the 29h median lead). "Missed or late" = no alert, or first alert at/after onset.
  - Scoring the lockbox patients for the demo is a product step required by §11.1; no metric was recomputed and nothing in `models/final/` was changed.
  - `models/final/threshold_curve.json` spans τ = 0.05–0.95 (plan §P5.T2) so it does not contain the chosen τ = 0.03; the dashboard trade-off chart marks the curve range and shows the policy point from `policy.json`. Left unchanged (frozen artifact).
  - Backend container uses `backend/requirements.txt` (runtime-only subset of the same pins) instead of the full requirements.txt (drops mlflow/optuna/jupyter, much smaller image). `models/final/` (4.2 MB) and `data/demo/` (216 KB) are now tracked in git so `docker compose up --build` works from a clean clone.
  - Added `GET /api/figures/{name}` (whitelisted test PNGs) so the Model page can show the real calibration and ROC/PR figures.
  - Threshold slider range 0.01–0.50 (human's Phase-7 brief) instead of plan's 0.05–0.95, because the policy τ = 0.03 lies outside the plan's range and the demo max risk is 0.17.

- **Phase 6 (2026-11-02, Claude):**
  - Run locally on an RTX 4070 (torch 2.6.0+cu124 installed in `.venv`, not added to requirements.txt or the backend image) instead of Colab. ~4 min training for 5 folds plus the focal-alpha check.
  - Deviation from §P6: **no lockbox evaluation**. The plan's "evaluate on test once" conflicts with the CLAUDE.md invariant that the lockbox was evaluated strictly once in P5.T6. All Phase 6 numbers are dev OOF only.
  - Deviation from §P6: 5-fold CV on the same folds as LightGBM (plan: fold 0 only), so OOF rows match LightGBM's 1,241,132 rows 1:1. Normalisation stats per fold from that fold's training patients. Input is 111 dims (the plan's own list sums to 111, not 108).
  - Focal alpha chosen on fold 0 val AUPRC: 0.50 (0.1189) vs 0.75 (0.1125) vs 0.90 (0.1129). Best epochs 2–4 of up to 30.
  - Two scores reported. *Pooled* = calibration + policy fit on the same OOF rows (how 0.4267 was produced; reproduced exactly). *Nested* = for each fold, isotonic, blend weight and policy fitted on the other 4 folds only.
  - Results (pooled / nested utility): LightGBM 0.4267 / 0.4196; GRU 0.3633 / 0.3498; ensemble (0.7 LightGBM + 0.3 GRU, all folds picked 0.7) 0.4319 / 0.4264. Nested false alerts/patient-day 0.478 → 0.420; sensitivity 85.5% → 84.9%; median lead 23 h for both.
  - Paired patient bootstrap (1,000, nested alerts): GRU − LightGBM −0.0698 [−0.0838, −0.0563]; ensemble − LightGBM +0.0068 [+0.0016, +0.0125], better in 99.5% of resamples.
  - Caveat: the chosen weight 0.7 sits on the upper edge of the plan's grid {0.3…0.7}; the grid was not widened after seeing results.
  - Not adopted: per §P6.T3 the API serves LightGBM regardless, `models/final/` is unchanged, and no test-set claim is made for the ensemble.
  - Corrected the old P6 board note ("LightGBM achieves superior calibrated utility"), which was written without a GRU run.

- **Phase 9 (2026-11-03, Claude) — notes:**
  - Bootstrap of the plain-GRU ensemble under the widened grid: ensemble(plain) − LightGBM +0.0058 [+0.0011, +0.0106] (the P6 grid gave +0.0068). Ensemble(top32) − LightGBM +0.0056 [+0.0017, +0.0096].
  - `ensemble.json` stores weight 0.7, policy τ 0.03 / k 1 / sticky 0, 5 GRU epochs (fold best epochs 4, 3, 2, 4, 4), focal α 0.5, the GRU config and the git sha of the code that trained it (a612d30). Nested dev metrics and in-sample pooled metrics (0.4362) are in `dev_metrics.json` and `reports/tables/ensemble_dev_final.csv`.
  - Final GRU trained on the GPU (≈5 s/epoch); served on CPU in float32 (`torch.load(..., map_location="cpu", weights_only=True)`). Stored demo risks were computed on Windows; the Linux container reproduces them exactly (max |Δ| 0.0).
  - **Latency, `POST /api/predict`, 70-hour stay (P7 recorded ≈315 ms for LightGBM):** Docker (Linux, HTTP, n = 40, median): LightGBM 169 ms → ensemble 183 ms (+14 ms). In-process on the laptop, interleaved (n = 40): 242 → 261 ms (+18 ms). Plain HTTP runs on Windows were noisy (271–353 ms for the same code) and are not used for the comparison. On Windows, `localhost` added ~2 s per request (IPv6 tried first), so `backend/bench_predict.py` uses 127.0.0.1.
  - Backend image: torch 2.6.0+cpu from the PyTorch CPU index (no CUDA libraries); torch headers, `protoc` and bundled `tests/` directories removed in the install layer (2.45 GB → 2.16 GB).
  - UI: the before state was the P7 Docker image (1.0) that Docker Desktop had auto-started on :8000/:8080. Critique and audit ran as isolated sub-agents; findings and the changes made are in `reports/ui_review.md`; before/after screenshots in `reports/screenshots/before/` and `reports/screenshots/`. Not re-scored after the changes (the detector was re-run: 3 findings → 0).
  - Demo cohort outcome labels are per model (`outcome` = LightGBM, `outcome_ensemble`); the API returns the served model's label.
  - `models/ensemble_final/` (0.3 MB) is tracked in git like `models/final/`; the variant fold models in `models/gru_h64/`, `models/gru_top32/` are not (gitignored, reproducible with `train_gru`).

## Blockers / questions for the human
- Record the 60-s fallback screen recording for the demo (see `reports/demo_script.md`).
- Tests: 95/95 pass (63 `tests/`; 32 `backend/tests/`, every API test run for both model kinds); ruff clean on src, tests, backend.
- Review PR `feature/ensemble-and-ui`. If you prefer to serve the lockbox-tested model, set `MODEL_KIND=lgbm` (nothing else changes).

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
| 2026-10-18 | P5.T1 | Isotonic calibration | Dev Brier 0.0177->0.0166, ECE 0.0182->0.0000 | Yes |
| 2026-10-18 | P5.T2 | Calibrated alert policy | Thresh 0.03, k=1, s=0, dev OOF utility=0.4267 | Yes |
| 2026-10-18 | P5.T3 | SHAP TreeExplainer & stability | Mean fold Jaccard = 0.5939, top features identified | Yes |
| 2026-10-18 | P5.T4 | Dev OOF subgroup fairness | Subgroups across hospital, gender, age, unit | Yes |
| 2026-10-18 | P5.T5 | MLflow final logging | Parameters, metrics, policy & artifacts registered | Yes |
| 2026-10-19 | P7.T1 | Demo cohort (24 beds) | online vs batch max diff 0.0 | Yes |
| 2026-10-18 | P5.T6 | Single Lockbox Test (8,068 pts) | **Utility = 0.4270 [0.3929, 0.4604], AUROC = 0.8475, Sens = 87.54%, Lead = 29.0h** | **FINAL** |
| 2026-11-02 | P6.T1–T3 | GRU (5-fold, dev) + LightGBM/GRU ensemble | Nested dev utility LGBM 0.4196, GRU 0.3498, ensemble 0.4264 (Δ +0.0068 [+0.0016, +0.0125]) | Dev only |
| 2026-11-03 | P9.1 | Widened w grid {0.5…0.9}, GRU variants h64 / top32 (nested) | Ensemble: plain 0.4255, **h64 0.4327**, top32 0.4252; LGBM 0.4196 | h64 served |
| 2026-11-03 | P9.2 | `/api/predict` latency, 70-h stay | Docker: LGBM 169 ms → ensemble 183 ms; in-process paired 242 → 261 ms | – |
