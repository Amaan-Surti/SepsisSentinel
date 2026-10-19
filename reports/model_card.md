# Model Card — SepsisSentinel LightGBM (final)

Format follows Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). All numbers are copied from `reports/tables/` and `models/final/`; the lockbox test set was evaluated exactly once (Phase 5.T6).

## Model details

- **Developer:** Amaan Surti, DSML course project (solo), 2026.
- **Model type:** gradient-boosted decision trees, LightGBM 4.5.0, binary objective, followed by isotonic calibration (scikit-learn 1.5.1) and a threshold alert policy.
- **Version:** `models/final/lgbm.txt`, 303 boosting rounds (mean best iteration across 5 dev folds = 275.6, scaled to all dev rows).
- **Hyperparameters (Optuna, 40 trials, best = trial 13):** learning_rate 0.0102, num_leaves 126, min_child_samples 847, feature_fraction 0.897, bagging_fraction 0.517 (freq 1), lambda_l2 0.301, scale_pos_weight 2.79, max_bin 255, seed 42, deterministic.
- **Inputs:** 298 causal features per ICU hour computed from the 40 PhysioNet 2019 variables using only the current and earlier hours: LOCF values (34), measured-this-hour flags (34), hours since last measurement (34), 24-h lab counts (26), rolling mean/min/max/std/slope over 6/12/24 h (137), 6-h differences (7), deltas vs patient baseline (11), clinical indices (6), partial SIRS/qSOFA/SOFA scores (4), static (5). `hospital`, `patient_id`, `Unit1`, `Unit2` and the label are never inputs.
- **Output:** calibrated probability that the hour carries `SepsisLabel = 1` (i.e. sepsis onset within the next 6 h or already ongoing). Alert at hour *t* if calibrated risk ≥ τ.
- **Alert policy (`models/final/policy.json`):** τ = 0.03, k = 1 consecutive hour, sticky = 0 h; chosen by grid search on calibrated dev out-of-fold predictions (dev OOF normalized utility 0.4267).
- **Explanations:** SHAP TreeExplainer (shap 0.46.0); the API returns the top-5 positive contributions per hour as plain-language text.
- **Licence / citation:** data from PhysioNet/CinC Challenge 2019 (Reyna et al., *Crit Care Med* 2020;48(2):210–217), open access.

## Intended use

- Research and teaching demonstration of a real-time early-warning pipeline on retrospective, de-identified ICU data.
- Decision support only: a prompt for a clinician to look again at a patient, never an automatic diagnosis or treatment trigger.
- Intended users: students, examiners, and researchers studying sepsis prediction, calibration, alert policies and cross-site generalization.

## Not intended for

- Any clinical use, diagnosis, triage or treatment decision. This is not a medical device and has no regulatory clearance.
- Deployment at a hospital that did not contribute training data without local re-validation and re-tuning of the threshold (see cross-hospital results: utility drops 32.7% from A to B).
- Paediatric patients, non-ICU wards, or data from a different EHR feed without remapping and validation.

## Training and evaluation data

- **Source:** PhysioNet 2019 public training sets A (20,336 patients, 790,215 hourly rows) and B (20,000 patients, 761,995 rows). Total 40,336 patients, 1,552,210 rows, 2,932 septic patients (7.27%), 27,916 positive rows (1.80%).
- **Splits (by patient, stratified by hospital × septic):** dev 32,268 patients (5 folds, used for all fitting, tuning, calibration and policy choice) and lockbox test 8,068 patients (evaluated once).
- **Labels:** Sepsis-3 based labels supplied by the Challenge; `SepsisLabel = 1` from 6 h before clinical onset *t_sepsis* onward.
- **Pre-processing:** physiological range cleaning (max loss 0.305% of observed values, FiO2), FiO2 percentages converted to fractions, no backward fill, no imputation beyond causal LOCF.

## Metrics

Primary metric: official Challenge normalized utility. Test values with 95% bootstrap CIs (1,000 patient resamples, seed 42).

| Metric (lockbox test, 8,068 patients) | Value | 95% CI |
|---|---|---|
| Normalized utility | 0.4270 | 0.3929 – 0.4604 |
| AUROC (hourly) | 0.8475 | 0.7965 – 0.8990 |
| AUPRC (hourly; prevalence 1.79%) | 0.1369 | 0.0854 – 0.2311 |
| Brier score | 0.0164 | 0.0123 – 0.0208 |
| ECE (10 bins) | 0.0027 | 0.0013 – 0.0084 |
| Patient sensitivity (alert in [t−12, t+3]) | 87.54% | 84.81% – 90.13% |
| Median lead time before onset | 29.0 h | 23.0 – 33.0 h |
| False alert events / non-septic patient-day | 0.529 | 0.512 – 0.547 |
| Event PPV | 0.0462 | – |
| Share of hours in alert | 20.2% | – |

**Baselines on the same test set (dev-chosen policies):**

| Model | Utility | AUROC | Sensitivity | False alerts/day |
|---|---|---|---|---|
| B0 never alert | 0.0000 | 0.500 | 0.00% | 0.00 |
| B1 SIRS ≥ 2 | 0.0507 | 0.615 | 82.76% | 1.53 |
| B2 qSOFA (partial) ≥ 2 | 0.0122 | 0.516 | 25.94% | 0.45 |
| B3 logistic regression | 0.2438 | 0.737 | 67.06% | 0.25 |
| **SepsisSentinel LightGBM** | **0.4270** | **0.848** | **87.54%** | **0.53** |

**Cross-hospital generalization (full features, R0):**

| Setting | Utility | AUROC | Sensitivity | False alerts/day |
|---|---|---|---|---|
| A internal (CV) | 0.4405 | 0.830 | 84.69% | 0.54 |
| A → B (train A, test B) | 0.2963 | 0.802 | 66.29% | 0.29 |
| B internal (CV) | 0.3759 | 0.857 | 76.62% | 0.33 |
| B → A (train B, test A) | 0.3188 | 0.777 | 91.56% | 0.93 |

Removing missingness features (R1), per-hospital standardization (R2) or removing admission timing (R3) did not close the gap (A → B utility 0.2931, 0.2882, 0.2502).

## Subgroup results (lockbox test)

| Subgroup | Utility | AUROC | Sensitivity | False alerts/day |
|---|---|---|---|---|
| Hospital A | 0.4376 | 0.827 | 87.71% | 0.66 |
| Hospital B | 0.4099 | 0.866 | 87.28% | 0.40 |
| Female | 0.3798 | 0.837 | 85.52% | 0.53 |
| Male | 0.4554 | 0.853 | 88.77% | 0.53 |
| Age < 45 | 0.3689 | 0.826 | 88.64% | 0.57 |
| Age 45–64 | 0.4246 | 0.851 | 88.04% | 0.51 |
| Age 65–79 | 0.4830 | 0.870 | 89.42% | 0.51 |
| Age ≥ 80 | 0.3514 | 0.799 | 80.25% | 0.57 |

Sensitivity is similar across sexes and hospitals; utility is lowest for patients ≥ 80 and < 45, and alert burden is higher at Hospital A. Subgroup test sets are small (CIs not computed per subgroup), so differences should be read as indicative only.

## Calibration

Isotonic regression fitted on dev OOF probabilities only. Dev Brier 0.0177 → 0.0166 and dev ECE 0.0182 → 0.0000 (in-sample for the calibrator); on the untouched test set ECE = 0.0027 and Brier = 0.0164. A risk of 3% therefore means roughly 3 in 100 such hours are labelled positive.

## Explainability

Global mean |SHAP| ranking (dev sample of 300 septic patients + 20,000 non-septic rows): ICU length of stay (`static__iculos`), hours since last FiO2 (`FiO2__tsl`), hospital-admission time (`static__hosp_adm_time`), ICU unit, hours since last lactate, temperature, 12-h max temperature, BUN, WBC, shock index. Top-10 sets are moderately stable across fold models (mean pairwise Jaccard 0.594). SHAP explains the model's behaviour, not physiology or causes; measurement-timing features reflect clinician ordering behaviour.

## Known limitations

- Retrospective, de-identified data; no prospective or silent-mode evaluation.
- Labels come from Sepsis-3 criteria applied to records, not perfect ground truth; onset times are approximate.
- Only two hospitals are available; external validity is simulated with A ↔ B transfer, and it drops substantially (A → B utility −32.7%). The hidden Challenge hospital C was not available.
- The low threshold (τ = 0.03) trades alarm burden for sensitivity: 20.2% of hours are in alert and event PPV is 4.6%. Any threshold trades missed cases against alarm fatigue.
- Strong reliance on stay-timing and measurement-timing features means the model partly learns care processes of these hospitals.
- Not a medical device.

## Ethical considerations

- Data are public, de-identified and used under the PhysioNet open-access terms; no re-identification was attempted.
- `hospital` is never an input; sex and age are inputs (as in the Challenge data) and their subgroup performance is reported above.
- False alerts cost clinician attention; missed alerts cost patients. Both are reported together rather than one headline number.
- The dashboard always shows the disclaimer: "Research demonstration on retrospective ICU data. Not approved for clinical diagnostic use."
