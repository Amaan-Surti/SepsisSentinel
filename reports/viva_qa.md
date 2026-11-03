# SepsisSentinel — Viva Defence Guide

Model answers with the real numbers. Each answer opens with a one-line headline, then the evidence. "Show:" points to the artifact to open if the examiner wants proof.

**Numbers to know by heart (lockbox test, 8,068 patients, once):**

| Metric | Value |
|---|---|
| Normalized utility | **0.4270** [0.3929–0.4604] (dev OOF 0.4267) |
| AUROC | **0.8475** [0.7965–0.8990] |
| AUPRC | **0.1369** (chance 1.79%, 7.6×) |
| Sensitivity | **87.54%** |
| Median lead time | **29.0 h** |
| False alerts | **0.529 / patient-day** |
| ECE | **0.0027** |
| Brier | **0.0164** |
| Event PPV | **4.62%** |
| Policy | **τ = 0.03, k = 1, sticky = 0** |
| *Served model (dev only)* | **LightGBM + GRU ensemble, 0.7 / 0.3**; lockbox score **not measured** |
| *Dev nested CV utility* | LightGBM 0.4196 · GRU (h64) 0.3529 · ensemble **0.4327** · Δ **+0.0131 [+0.0072, +0.0190]** |
| Baselines (utility) | B1 SIRS 0.0507 · B2 qSOFA 0.0122 · B3 LogReg 0.2438 |
| Cross-hospital utility | A internal 0.4405 → A→B 0.2963 (−32.7%) · B internal 0.3759 → B→A 0.3188 (FA/day 0.33 → 0.93) |

---

## A. Problem framing and metrics

### Q1. Why not report accuracy?
**Headline: accuracy rewards doing nothing.** Only 1.80% of hours are positive. A model that never alerts is 98.2% accurate and clinically useless (its utility is exactly 0, baseline B0). We report the official Challenge utility as the primary metric, plus AUROC/AUPRC, calibration, and patient-level clinical metrics (sensitivity, lead time, false alerts per patient-day, event PPV).

### Q2. Explain the PhysioNet utility function.
**Headline: it scores each patient-hour by clinical value of the alert's timing.**
- **Septic patients:** an alert between *t_sepsis* − 12 h and *t_sepsis* + 3 h earns credit. It rises linearly from 0 at −12 h to +1 at −6 h and stays at +1 until +3 h. An alert earlier than −12 h costs −0.05 like a false alarm. A **missed** hour from −6 h to +3 h is penalised, ramping to **−2**.
- **Non-septic patients:** each alert hour costs −0.05.
- **Normalisation:** U_norm = (U − U_no-alerts) / (U_oracle − U_no-alerts), so never alerting = 0 and an oracle = 1.

Implications: a miss near onset costs twice what a perfect alert earns, so the optimum favours sensitivity. False alarms are cheap individually (−0.05/h) but add up across the 98% of non-septic hours. *Show:* `src/sepsis/utility.py`, `tests/test_utility.py` (8 hand-verified cases).

### Q3. Why are labels shifted 6 hours early?
**Headline: so that predicting the label means anticipating onset.** The Challenge sets `SepsisLabel = 1` from 6 h before *t_sepsis*, so a classifier trained on it learns to fire before clinical onset. We recover *t_sepsis* = first positive index + 6. The utility's peak credit window starts at −6 h for the same reason.

### Q4. Why is AUPRC only 0.137? Isn't that bad?
**Headline: AUPRC's chance level is the prevalence, 1.79%, so 0.137 is 7.6× chance.** AUROC is prevalence-independent; AUPRC is not. Hourly AUPRC is also intrinsically hard here: hours just before and just after the label switches on look almost identical, and every septic patient contributes many positive hours. Logistic regression gets 0.056 and SIRS 0.025 on the same test set.

### Q5. Your utility of 0.427 is higher than the Challenge winner's 0.360. Did you beat them?
**Headline: no, the settings are not comparable.** The official 0.360 was measured on a hidden test set that included hospital C, which contributed no training data. Our test patients come from the same two hospitals as our training data, an easier in-distribution setting. Our own cross-hospital experiment shows how much harder transfer is: A→B utility is 0.2963.

---

## B. Leakage and validation

### Q6. How did you prevent data leakage?
**Headline: patient-level splits, causal features, dev-only fitting, a once-only lockbox, and tests that enforce causality.**
1. **Splits by patient:** no patient's rows appear in two splits.
2. **Lockbox:** 8,068 patients (20%, stratified by hospital × septic) were locked before modelling.
3. **Causal features:** every feature at hour *t* uses rows ≤ *t*; LOCF only, no backward fill, rolling windows look back only, the baseline uses hours 0–5 as available so far.
4. **Dev-only fitting:** hyperparameters, early stopping, calibrator and threshold were all fitted on dev data or dev OOF predictions.
5. **No identifiers:** `hospital`, `patient_id`, `Unit1/Unit2` and the label are not features (tested).
6. **Sanity alarms:** the build plan says to stop if utility > 0.6 or AUROC > 0.95; we are at 0.427 / 0.848.

*Show:* `tests/test_features.py`.

### Q7. Show me the causality test.
**Headline: features on a truncated stay must equal features on the full stay for the overlapping hours.** For 20 random patients (seed 42), we compute features on the full stay and on a random prefix `rows[:k]`. For every row < k the two must match (`np.allclose(..., equal_nan=True)`). Any look-ahead (a centred window, a backward fill, a normalisation over the whole stay) would break the equality. A second hand-built 8-hour patient checks LOCF, time-since-last (0, 0, 1, 0, 1, 2, 0, 0), a 6-h mean of 84.0 and a 6-h slope against `np.polyfit`. The API adds an online-vs-batch parity test: `POST /api/predict` reproduces stored risks to < 10⁻⁶ (observed 0).

### Q8. You evaluated on the test set. How do I know you didn't tune on it?
**Headline: the test set was evaluated once, and dev and test agree to within 0.0003.** Every choice (features, Optuna, calibration, policy) was made on 5-fold dev OOF. The single lockbox run is logged in `STATUS.md` with its date, and the numbers were never changed afterwards. Dev OOF utility 0.4267 vs test 0.4270, and dev AUROC 0.8447 vs test 0.8475, are what you would expect from an untouched holdout. Building the demo cohort later re-scored some test patients with the frozen model, but that computes no metrics and changes nothing.

### Q9. Why bootstrap patients rather than rows?
**Headline: rows within a patient are highly correlated.** Resampling rows would treat 1.5M dependent hours as independent and give falsely narrow intervals. We resample the 8,068 patients 1,000 times (seed 42) and recompute every metric, which respects within-patient correlation and the patient-level utility.

---

## C. Features and model

### Q10. Why is missingness informative? Isn't that a bug?
**Headline: in an ICU, a test being ordered is a clinical decision that reflects concern.** Lactate is drawn when sepsis is suspected; FiO2 is recorded when oxygen therapy changes. Removing the missingness families (measured flags, hours-since-last, 24-h counts) is the largest ablation loss: dev utility 0.4262 → 0.4039 (−0.0223, −5.2%), versus at most −0.0057 for any other group. The top SHAP features include hours since FiO2, lactate, PaCO2 and EtCO2 were recorded. The flip side is that missingness is very site-specific (the adversarial AUROC of 1.0000 is driven by HCO3 missingness) and partly reflects suspicion clinicians already have. *Show:* `reports/tables/ablations.csv`, `fig_ablations.png`.

### Q11. Why LightGBM rather than deep learning?
**Headline: tabular, irregular, mostly-missing hourly data with 1.5M rows is where gradient boosting is strongest, and it won the Challenge.**
- Handles missing values natively.
- Trains in minutes on a laptop.
- Exact, fast SHAP via TreeSHAP.
- Easy to calibrate.
- Top Challenge entries were mostly boosted trees on engineered temporal features.

Our causal rolling/slope/time-since features give the trees the temporal context an RNN would learn. **Evidence (development data, nested 5-fold CV):** a unidirectional GRU alone scores 0.3498 (hidden 128) or 0.3529 (hidden 64) against LightGBM's 0.4196; GRU − LightGBM −0.0698 [−0.0838, −0.0563]. The GRU is weaker alone but useful as a second opinion (Q28).

### Q12. What did Optuna tuning buy you?
**Headline: little in AUROC, more in utility and calibration.** 40 TPE trials on folds 0–2 maximised validation AUPRC: 0.1342 vs 0.1264 for the defaults. On 5-fold OOF, AUROC barely moved (0.8446 → 0.8447), but utility rose from 0.4183 to 0.4262 and ECE fell from 0.0412 to 0.0182. The tuned model is more regularised (min_child_samples 847, learning rate 0.010, scale_pos_weight 2.79 vs 5). The final model was trained on all dev rows for 303 rounds (1.1 × the mean best iteration of 275.6).

### Q13. Why is ICU length of stay the top feature? Is that leakage?
**Headline: it is causal (known at hour *t*) and clinically meaningful, but it's a context feature.** `static__iculos` = hour + 1 is available in real time. Sepsis onsets cluster later in stays, and still being in the ICU is itself a severity marker. Removing timing features (R3) lowers internal utility at hospital A from 0.4405 to 0.3821 and hurts transfer too, so it carries genuine signal. Removing it would not make the model fairer, just weaker.

---

## D. Calibration and alerting

### Q14. How does isotonic calibration work and why choose it?
**Headline: it learns a monotone, piecewise-constant map from raw score to observed frequency.** Isotonic regression fits the best non-decreasing step function from LightGBM's raw probabilities to labels (pool-adjacent-violators). It preserves ranking (AUROC unchanged) and fixes the distortion from class weighting and AUPRC early stopping. We chose it over Platt scaling because it makes no sigmoid-shape assumption and we have over a million OOF rows, so overfitting is not a concern. It was fitted on dev **out-of-fold** predictions only. Results:
- dev Brier 0.0177 → 0.0166;
- test ECE 0.0027 (uncalibrated 0.0221) and Brier 0.0164.

### Q15. Why does calibration matter for a product?
**Headline: so the threshold and the displayed risk have real meaning.** A 3% risk on the dashboard means about 3 in 100 such patient-hours are positive, which clinicians can reason about. When a hospital tunes its threshold for an alert budget, calibrated probabilities make that choice interpretable. Calibration also has to be re-checked at every new site (Q19).

### Q16. Why such a low threshold (3%)? How do you handle alarm fatigue?
**Headline: the utility function prefers sensitivity, and τ = 0.03 maximised dev OOF utility, at a stated cost.**
- With 1.8% prevalence, 3% is above the base rate.
- The cost: 20.2% of hours in alert, event PPV 4.6%, 0.53 false alert events per non-septic patient-day (about 1 per 45 h per patient, about 13/day on a 24-bed unit).
- Mitigations available in the policy: **k-consecutive** hours and **sticky** suppression (the search found k = 1, s = 0 best for utility, but a hospital could trade utility for fewer alerts).
- Mitigations in the product: a **Watch** tier below the threshold, calibrated risk, and **reasons** so a clinician can dismiss an alert quickly.
- In deployment, set the threshold for an **agreed alert budget**. The trade-off curve on the Model page shows the cost: at τ = 0.05, dev sensitivity is 70.0% at 0.22 false alerts/day.

### Q17. The median lead time is 29 h. Isn't that suspiciously early?
**Headline: it's real but must be read with the alert rate.** Lead time = *t_sepsis* − first alert within a 48-h look-back. At a 3% threshold some septic patients are flagged early in their stay and remain flagged, so a 29-h lead is less specific than a 6-h one. The utility gives no credit before −12 h (it charges −0.05/h), so lead time does not inflate utility. The clinically relevant number is sensitivity within [−12, +3] h: 87.54%.

---

## E. Generalization, fairness, explainability

### Q18. What happens at a new hospital?
**Headline: up to a third of the utility is lost, mostly because the operating point is wrong.** Training on A and testing on B drops utility from 0.4405 (A internal) to 0.2963 (−32.7%). Training on B and testing on A gives 0.3188, against 0.3759 for B internal.
- AUROC falls only 0.03–0.08, so ranking largely survives.
- A's policy is too strict at B: sensitivity 84.7% → 66.3%.
- B's policy is too loose at A: false alerts 0.33 → 0.93/day.

*Show:* `reports/tables/cross_hospital.csv`, `fig_cross_hospital.png`.

### Q19. What drives the domain shift? Is it just missingness?
**Headline: no, removing missingness doesn't close the gap.** The hospitals are perfectly separable (adversarial AUROC 1.0000), mainly through documentation patterns.
- **R1 (no missingness features):** A→B utility 0.2931, essentially the same as R0's 0.2963.
- **R2 (per-site z-scoring):** 0.2882, so aligning feature distributions doesn't fix it.
- **R3 (no admin timing):** worse everywhere.

So the shift is in the feature–outcome relationship and in base rates (8.8% vs 5.7% septic), which unsupervised alignment cannot correct. The practical remedy is local recalibration and threshold tuning, then a silent prospective trial.

### Q20. Is the model fair across subgroups?
**Headline: sensitivity is similar across sex and hospital; very elderly patients do worst; subgroup estimates are noisy.**
- Test sensitivity: female 85.5%, male 88.8%; hospital A 87.7%, B 87.3%.
- Age ≥ 80: sensitivity 80.3%, AUROC 0.799 (lowest).
- The test utility gap between women and men (0.380 vs 0.455) is **not** present in dev OOF (0.423 vs 0.429).
- Age < 45 is the best band in dev (0.476) but among the worst in test (0.369).

That instability points to sampling noise in small subgroups; we did not compute per-subgroup CIs. Sex and age are inputs (as in the Challenge data), hospital is not.

### Q21. What does SHAP show, and what are its limits?
**Headline: it shows what the model relies on, not why the patient is ill.** TreeSHAP gives exact per-feature attributions in log-odds that sum to the prediction.
- **Global top features:** ICU hour, hours since FiO2, hospital-admission time, unit, hours since lactate, temperature, 12-h max temperature, BUN, WBC, shock index.
- **Stability:** top-10 sets are moderately stable across fold models (mean Jaccard 0.594).
- **Per hour:** the dashboard shows the top 5 positive contributions as text (e.g. "Lactate last measured 2h ago", "Temperature 38.4").

Limits:
- Attributions describe the model, not physiology or causality.
- Correlated features share credit arbitrarily.
- Timing features reflect clinician behaviour.
- We show only risk-raising factors, so the list explains why risk is high, not the whole picture.

### Q22. Are SHAP explanations clinically valid?
**Headline: partly. They line up with known sepsis physiology, but some are care-process proxies.** Fever, WBC, BUN/creatinine, shock index and the SIRS count all match Sepsis-3 and SIRS reasoning. "Lactate drawn 2 h ago" is clinically sensible as a marker of suspicion, but it tells the clinician something they already know. Validity would need clinician review of a sample of alerts, which is future work.

---

## F. Deployment and product

### Q23. What would real deployment require?
**Headline: local validation first, then a silent trial, then workflow integration and monitoring.**
1. Retrospective validation at the target hospital, with local recalibration and threshold tuning for an agreed alert budget.
2. Silent-mode prospective run: compare alerts with clinician recognition, actual lead time and workload.
3. Real-time data feeds with the same units and semantics; the batch/online parity test in CI.
4. Workflow: who gets the alert, what action, acknowledgement and suppression rules.
5. Monitoring of input drift, alert rate, calibration and subgroup performance.
6. Regulatory review as clinical decision-support software; model card and audit trail.

### Q24. What are the real-time constraints? Can it run every hour?
**Headline: yes, easily.** Inference is a feature computation over the patient's history plus a 303-tree ensemble and a lookup. Measured on the development laptop: a 24-bed ward query takes about 14 ms, and `POST /api/predict` scores a full 70-hour stay (cleaning, features, model, calibrator, SHAP for the last hour) in about 315 ms. Features need the patient's history (up to 24 h windows and the 72-h time-since cap), so a production service would keep a per-patient state store or recompute from the EHR feed each hour. LightGBM needs only `libgomp1` in the container.

### Q25. How is the product built, and how do you know it matches the research code?
**Headline: one shared inference module, tested end to end.**
- **Backend:** FastAPI, loading the frozen `models/final/` (booster, calibrator, policy, feature list) and, by default, `models/ensemble_final/` (GRU, GRU calibrator, `ensemble.json`) once at startup. `MODEL_KIND=lgbm` serves LightGBM alone.
- **Online path:** `src/sepsis/inference.py` runs cleaning → `build_features_from_df` → LightGBM → isotonic, and the same features → sequence rows → GRU (CPU) → isotonic, then blends 0.7 / 0.3 and applies the policy. The demo cohort was scored with the same functions.
- **Tests:** an API test posts a demo patient's raw rows and checks the risks equal the stored predictions (< 10⁻⁶) for both model kinds; ensemble online vs batch max |Δ| 4.3 × 10⁻⁷.
- **Frontend:** React/TypeScript with a replay clock and a model badge (Ensemble / LightGBM).
- **Deployment:** Docker Compose (API :8000 with CPU-only PyTorch, nginx UI :8080).
- 93 automated tests pass.

### Q26. Why did you include missed cases and false alarms in the demo?
**Headline: honesty.** The 24-bed cohort is 8 early detections, 2 missed or late septic patients, 10 quiet non-septic patients and 4 non-septic patients with false alerts, all from the lockbox test set. A demo with only successes would overstate the system. The real failure modes are what a clinician needs to see.

### Q27. What would you do with more time?
- A fresh, untouched test set (e.g. a third hospital) to measure the ensemble once, since the lockbox is spent.
- Domain adaptation or few-shot recalibration with a small amount of target-site data.
- Per-subgroup CIs and a fairness-aware threshold analysis.
- Alert-budget-constrained policy search (maximise sensitivity subject to ≤ X alerts/day).
- Clinician review of SHAP explanations.
- Prospective silent evaluation.

---

## G. The ensemble (Phase 9)

### Q28. Why an ensemble?
**Headline: the GRU is weaker alone but makes different mistakes, so averaging helps on development data.**
- Nested 5-fold CV on 32,268 development patients (each fold's calibration, weight and policy fitted on the other four): LightGBM **0.4196**, GRU (hidden 64) **0.3529**, ensemble 0.7 × LightGBM + 0.3 × GRU **0.4327**.
- Paired patient bootstrap of the nested alerts: ensemble − LightGBM **+0.0131 [+0.0072, +0.0190]**, better in 100% of 1,000 resamples. False alerts fall from 0.478 to 0.415 per patient-day at the same 85% sensitivity.
- Why it can work: LightGBM sees hand-made windows (6/12/24 h); the GRU reads the raw hourly sequence and its own memory, so their errors are only partly correlated.
- What was searched, fixed before the run: w ∈ {0.5, 0.6, 0.7, 0.8, 0.9} and two GRU variants (hidden 64; +32 LightGBM features). Every configuration is reported (`reports/tables/ensemble_dev_*.csv`). The hidden-64 variant was adopted because its gain over the plain-GRU ensemble had a CI above 0 (+0.0072 [+0.0033, +0.0109]); the +32-features variant did not (−0.0002 [−0.0039, +0.0032]).
- Honest size: +0.013 utility is small, and picking the best of three variants on the same folds adds some optimism.

*Show:* Model page → "Development comparison (nested 5-fold CV)"; `reports/tables/ensemble_dev_comparison.csv`.

### Q29. Why not test the ensemble on the lockbox?
**Headline: the lockbox was spent once, on LightGBM; opening it again to check a model we chose afterwards would turn it into a second validation set.**
- The protocol said: evaluate once (P5.T6) and never use the test set to choose anything. Scoring the ensemble on it now would tempt exactly that — if it came out lower, would we switch back? Either answer makes the 8,068 patients part of model selection.
- So the claims are kept apart: **0.4270 [0.3929, 0.4604] is the lockbox result of the LightGBM component**; the ensemble's evidence is development-only (nested utility 0.4327 vs 0.4196 for LightGBM on the same folds). The app and every document label them that way and never put them in one table.
- What the dev result does suggest (not prove): for LightGBM, the development estimates (0.4267 pooled, 0.4196 nested) were close to its lockbox result (0.4270), so development estimates have transferred well once on this data. The ensemble's +0.0131 is a development gain only; its test performance is unknown.
- The right next step is a new untouched dataset, not re-opening the old one. `MODEL_KIND=lgbm` serves the test-validated model if an examiner prefers it.

### Q30. How does the GRU stay causal?
**Headline: three locks — causal inputs, a one-directional network, and a test that tries to break it.**
1. **Inputs:** each hour's GRU input (LOCF values, measured flags, hours since last measurement, SIRS, partial SOFA, static fields, ICU hour) comes from the same causal feature builder as LightGBM, which only looks at hours ≤ *t* (prefix-equality test since P3).
2. **Network:** `nn.GRU` is unidirectional (`bidirectional=False`); the output at hour *t* is a function of inputs 0…*t* only. Padding is on the right, so it never leaks into earlier hours. Normalisation statistics come from training patients only (all 32,268 dev patients for the final model).
3. **Tests:** `tests/test_gru.py::test_gru_is_causal` changes hours ≥ 12 of a random sequence and checks outputs before hour 12 are bit-identical. `tests/test_ensemble.py::test_ensemble_online_path_is_causal` does the same on the full served path — raw rows → cleaning → features → LightGBM + GRU → calibrators → blend — by truncating a real demo stay and by raising HR, Resp, lactate and temperature after a cut-off: earlier ensemble risks do not move (< 10⁻⁶), later ones do.
- Inference runs on CPU in float32, so the risk is the same on every machine (online vs batch 4.3 × 10⁻⁷).

---

## Limitations to state unprompted (build plan §13.4)

Retrospective, de-identified data · Sepsis-3 labels from records are not perfect ground truth · no hospital C, so external validity is only simulated with A↔B · thresholds trade missed cases against alarm fatigue · SHAP explains the model, not causes · the served ensemble has no lockbox estimate (its gain is a development-data result; reasons come from the LightGBM part only) · not a medical device.
