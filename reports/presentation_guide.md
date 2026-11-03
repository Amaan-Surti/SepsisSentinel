# SepsisSentinel — Class Presentation Guide

How to explain the whole project while demonstrating the live dashboard. It covers what to say, what to click, what each thing on screen means, what happens in the backend at that moment, and why each design choice beat its alternatives.

All numbers come from `reports/tables/` and are frozen. Quote them exactly.

**One sentence to get right every time:** the served model is a LightGBM + GRU ensemble chosen on development data; its lockbox score has not been measured; the 0.4270 test result belongs to the LightGBM component alone.

---

## 0. The 30-second opening (say this before touching the laptop)

> "Sepsis kills about 11 million people a year, roughly one in five deaths worldwide, and every hour of delayed treatment makes it worse. The hard part isn't treating it, it's *noticing* it early, because it hides in hundreds of numbers that change every hour.
>
> I built **SepsisSentinel**: a system that reads an ICU patient's vitals and lab results every hour and estimates their risk of sepsis in the next few hours. It explains *why*, and it was tested honestly on 8,068 patients it had never seen. It catches **87.5%** of septic patients, typically more than a day before onset, and its risk numbers are **calibrated**: when it says 3%, it means 3%.
>
> And I'll show you where it *fails*, because that's the most important finding."

That last line hooks the audience. Most student projects only show successes.

---

## 1. Suggested flow (about 12–15 minutes)

| Min | Segment | Screen |
|---|---|---|
| 0:00 | Opening hook (§0) | Title slide |
| 0:30 | The problem and the data (§2–3) | Slides 2–4 |
| 2:30 | How the model was built (§4) | Slides 5–6 |
| 5:00 | **Live demo, Ward page** (§6.1–6.2) | Browser :8080 |
| 7:00 | **Live demo, Patient page** (§6.3) | Browser |
| 8:30 | **Live demo, threshold slider + Model page** (§6.4–6.5) | Browser |
| 10:00 | Results vs baselines, cross-hospital finding (§5) | Slides 8–10 |
| 11:30 | What's under the hood: backend and architecture (§7) | Slide 11 |
| 12:30 | Why these choices, alternatives (§8), limitations, takeaways | Slides 13–14 |
| 13:30 | Questions (§10) | – |

If time is short, cut §7 to one sentence and §8 to the summary table only.

---

## 2. The problem, in plain words

- **Sepsis** (Sepsis-3 definition, 2016): life-threatening organ dysfunction caused by the body's dysregulated response to infection. The immune system starts damaging the body's own organs.
- **Why early matters:** in septic shock, each hour of delay before effective antibiotics lowers survival (Kumar et al., 2006). Hospitals talk about a "golden hour".
- **Why it's hard:** early sepsis looks like many other things. The clues (a rising heart rate, a slight fever, a lab that drifts) are spread over hours and buried among roughly 40 measurements per patient. Nurses and doctors can't watch every number for every patient continuously.
- **Why existing rules aren't enough:**
  - **SIRS** (fever / fast heart rate / fast breathing / abnormal white cells) fires on almost everyone. On our data it alarms 1.53 times per patient-day.
  - **qSOFA** is quiet but misses three quarters of cases.
- **The other side: alarm fatigue.** ICU staff already ignore most alarms because most are false. An AI that cries wolf is worse than useless. So our goal was *early **and** trustworthy*, not just "accurate".

**Line to use:** "The goal is not to replace the doctor. It's a second pair of eyes that never gets tired, and that tells you *why* it's worried."

---

## 3. The data

**Source:** PhysioNet / Computing in Cardiology Challenge 2019, an international competition dataset. It's public and de-identified.

| | Hospital A | Hospital B | Total |
|---|---|---|---|
| Patients (ICU stays) | 20,336 | 20,000 | **40,336** |
| Hourly rows | 790,215 | 761,995 | **1,552,210** |
| Septic patients | 1,790 (8.80%) | 1,142 (5.71%) | **2,932 (7.27%)** |
| Positive hours | 2.17% | 1.41% | **1.80%** |

- **One row = one patient-hour**, with 40 variables:
  - 8 vital signs (heart rate, SpO₂, temperature, blood pressures, respiratory rate, EtCO₂);
  - 26 lab tests (lactate, white cells, creatinine, platelets…);
  - 6 demographic/admin fields (age, sex, ICU unit, time since hospital admission, hours in ICU).
- **The label** `SepsisLabel` switches to 1 **six hours before** the clinical onset time. So the model is trained to *anticipate* sepsis, not recognise it after the fact.
- **The imbalance:** only 1.8% of hours are positive. A model that never alerts is 98.2% "accurate", which is why we never report accuracy.
- **Missingness:** heart rate is recorded most hours (missing 9.9%), temperature is missing 66% of the time, and most labs are missing more than 90% of the time. Labs are only drawn when a clinician orders them.
- **The two hospitals are very different:** a classifier can tell which hospital a patient came from with **AUROC 1.0000** (perfect), mostly from *which tests were ordered* (HCO₃ missingness is the top clue). This becomes important in the cross-hospital experiment.

**On screen:** Slide 4 shows the real missingness heatmap.

---

## 4. How the model was built (the ML pipeline)

Explain it as a pipeline: **raw data → clean → split → features → model → calibrate → choose alert threshold → explain**.

### Step 1: Cleaning (`src/sepsis/cleaning.py`)
- Physically impossible values become "missing" (heart rate outside 20–300, temperature outside 25–45 °C, etc.). No variable lost more than 0.31% of its values.
- FiO₂ recorded as a percentage (e.g. 40) is converted to a fraction (0.40).
- The two unit flags become one `unit` variable (MICU / SICU / unknown).

### Step 2: Splitting, the "lockbox" (`src/sepsis/splits.py`)
- **Split by patient, never by row.** If one patient's hours appeared in both training and testing, the model could memorise that patient, which is cheating.
- 20% of patients (**8,068**) were locked away as the **test set** before any modelling and opened **exactly once** at the very end.
- The other **32,268 "development" patients** were split into 5 folds for cross-validation. Every decision (features, hyperparameters, threshold) was made using these only.

**Line:** "I treated the test set like an exam paper in a sealed envelope. I opened it once, at the end, and I didn't change anything afterwards."

### Step 3: Feature engineering, 298 features (`src/sepsis/features.py`)
The raw hourly snapshot isn't enough; the model needs to see *trends* and *context*. For every patient-hour, using **only that hour and earlier**:

| Feature family | Count | What it captures | Example |
|---|---|---|---|
| Current values (carried forward) | 34 | last known value of each measurement | `HR` = 112 |
| Measured this hour? | 34 | was a test just done | `Lactate__meas` = 1 |
| Hours since last measured | 34 | how recently someone checked | `FiO2__tsl` = 2 |
| Lab counts in last 24 h | 26 | how often a test is being repeated | `Lactate__cnt24` = 3 |
| Rolling stats (6/12/24 h) | 137 | mean, min, max, variability, slope (trend) | `Temp__max_12h` = 38.4 |
| 6-hour change | 7 | sudden shifts | `HR__diff6` = +18 |
| Change vs patient's own baseline | 11 | "abnormal *for this person*" | `MAP__dbase` = −15 |
| Clinical indices | 6 | doctor-style ratios | shock index = HR ÷ SBP |
| Clinical scores | 4 | SIRS, partial qSOFA, partial SOFA, 24-h SOFA change | `score__sirs` = 3 |
| Static | 5 | age, sex, unit, admission time, hours in ICU | |

**The "no cheating" guarantee (causality):** every feature at hour *t* uses only hours ≤ *t*, so there's no peeking into the future. This is enforced by an automated test. We compute features on a patient's full stay and on a truncated stay, and the overlapping hours must match exactly. If any feature looked ahead, the test would fail.

**Missingness as information:** "was it measured?" and "hours since last measured" are features on purpose. A lactate gets drawn *because a doctor is worried about sepsis*, so the act of ordering is a signal.

### Step 4: The model, LightGBM (`src/sepsis/train_lgbm.py`, `tune_lgbm.py`, `final_model.py`)
- **What it is:** gradient-boosted decision trees. It builds hundreds of small decision trees one after another, each fixing the mistakes of the ones before. The final prediction adds up all the trees.
- **Training:** 5-fold cross-validation on dev patients, with early stopping (stop adding trees when validation performance stops improving).
- **Tuning:** **Optuna** (Bayesian hyperparameter search) ran 40 trials over learning rate, tree size, regularisation and class weighting. The best trial improved validation AUPRC from 0.1264 to 0.1342.
- **Final model:** retrained on all 1,241,132 dev rows for 303 trees (1.1 × the average best stopping point of 275.6).

### Step 5: Calibration, isotonic regression (`src/sepsis/calibrate.py`)
- Raw boosted-tree scores aren't true probabilities (class weighting inflates them).
- **Isotonic regression** learns a monotone "correction curve" from raw score to the real observed frequency. It was fitted on *out-of-fold* predictions, so it never saw its own training data.
- **Result:** on the test set, expected calibration error is **0.0027**. When the dashboard shows 3%, about 3 in 100 such patient-hours really are septic.

### Step 6: The alert policy (`src/sepsis/alerting.py`)
A probability isn't a decision. The policy turns risk into **Alert / no alert**:
- **Threshold τ:** alert when risk ≥ τ.
- **k:** require k consecutive hours above τ.
- **Sticky:** keep the alert on for s hours after it fires.

We grid-searched 594 combinations on dev data to maximise the official utility score. The winner was **τ = 3%, k = 1, sticky = 0**.

**Why 3% sounds low but isn't:** the base rate is 1.8%, so 3% means "nearly twice as likely as a typical patient-hour", and the scoring rule punishes a missed septic patient (−2) much more than a false alarm (−0.05 per hour).

### Step 7: Explainability, SHAP (`src/sepsis/explain.py`)
- **SHAP** splits each prediction into contributions from each feature. These are Shapley values from game theory, computed exactly for trees by TreeExplainer.
- For every hour, the dashboard shows the **top 5 factors pushing risk up**, translated into plain English (e.g. `Lactate__tsl = 2` → "Lactate last measured 2h ago").
- **Top global drivers:** time in ICU, hours since FiO₂ was recorded, time since hospital admission, ICU unit, hours since lactate, temperature, BUN, white cells, shock index.

### Step 8: How it was scored, the official utility function (`src/sepsis/utility.py`)
- Septic patient: an alert from 12 h before to 3 h after onset earns credit (full credit from 6 h before). A **missed** hour near onset costs up to **−2**.
- Non-septic patient: each alert hour costs **−0.05**.
- Normalised so **never alerting = 0** and a **perfect oracle = 1**.

We also report clinical metrics a doctor would ask for: sensitivity, lead time, false alarms per patient-day, PPV, calibration.

---

## 5. The results (what to emphasise)

### Headline: lockbox test, 8,068 unseen patients, evaluated once

| Metric | Value | Plain meaning |
|---|---|---|
| **Normalized utility** | **0.4270** [0.3929–0.4604] | 43% of the way from "do nothing" to "perfect oracle" |
| AUROC | **0.8475** | ranks a septic hour above a non-septic one 85% of the time |
| AUPRC | **0.1369** | **7.6×** better than chance (chance = 1.79%) |
| Patient sensitivity | **87.54%** | about 9 in 10 septic patients alerted in the useful window |
| Median lead time | **29.0 h** | first alert typically more than a day before onset |
| False alerts | **0.53 / patient-day** | about one false alarm per patient every 45 hours |
| Calibration error (ECE) | **0.0027** | risk numbers are trustworthy |

**Line:** "The development estimate was 0.4267 and the test result was 0.4270, a difference of 0.0003. That's what an honest, non-overfitted model looks like." (These are the LightGBM model's numbers.)

### What the app serves: the LightGBM + GRU ensemble (development data only)

After the lockbox test I compared LightGBM with a unidirectional GRU on the development folds, using nested 5-fold cross-validation (each fold's calibration, blend weight and policy fitted on the other four folds). The search space was fixed before running it, and every configuration is reported.

| Development, nested CV (not lockbox) | Utility | Sensitivity | False alerts/day |
|---|---|---|---|
| LightGBM | 0.4196 | 85.46% | 0.478 |
| GRU (hidden 64) | 0.3529 | 78.13% | 0.400 |
| **Ensemble 0.7 × LightGBM + 0.3 × GRU** | **0.4327** | **85.00%** | **0.415** |

Paired bootstrap: ensemble − LightGBM **+0.0131 [+0.0072, +0.0190]**.

**Line:** "The recurrent network on its own is clearly worse, but as a second opinion it helps a little, with fewer false alarms at the same sensitivity. That's a development-data result. I did **not** test the ensemble on the sealed test set, because that set was used once, for LightGBM. Opening it again to check a model I picked afterwards would make it a validation set. So the app serves the ensemble, and the test-set number stays with LightGBM."

### Versus baselines (same test patients)

| Model | Utility | Sensitivity | False alerts/day |
|---|---|---|---|
| Never alert | 0.0000 | 0% | 0 |
| SIRS ≥ 2 (classic rule) | 0.0507 | 82.8% | 1.53 |
| qSOFA ≥ 2 | 0.0122 | 25.9% | 0.45 |
| Logistic regression | 0.2438 | 67.1% | 0.25 |
| **SepsisSentinel** | **0.4270** | **87.5%** | 0.53 |

- **+75%** utility over logistic regression and **8.4×** the SIRS rule.
- SIRS catches almost as many patients, but at three times the false alarms.

### The ablation: missingness is a signal
Removing the "was it measured / how long ago" features drops utility from 0.4262 to **0.4039**, the biggest loss of any feature group. Removing rolling statistics, deltas or clinical scores each costs less than 0.006.

### The headline finding: it doesn't travel well between hospitals
| Setting | Utility |
|---|---|
| Train on A, test on A | 0.4405 |
| **Train on A, test on B** | **0.2963 (−32.7%)** |
| Train on B, test on B | 0.3759 |
| **Train on B, test on A** | **0.3188** (false alarms 0.33 → **0.93/day**) |

- The model still *ranks* patients reasonably (AUROC drops only 0.03–0.08). What breaks is the **alert threshold**: hospital A's threshold is too strict at B (sensitivity falls to 66%), and B's is too loose at A (false alarms triple).
- Removing missingness features or z-scoring each hospital did **not** fix it.

**Line:** "The test score at your own hospital is where validation *starts*, not where it ends. Real-world sepsis tools have failed in exactly this way."

---

## 6. Live demo script: what's on screen and what's happening behind it

**Setup:** http://localhost:8080 (Docker), browser at 100% zoom, replay at hour 0, speed 2×, threshold 3.0%, ground truth **off**. The UI is a gallery: the **ward is a wall of numbered exhibits**; opening a bed **mounts it on a white object label**, and the rest of the wall dims. The numbered **tour rail** along the bottom (stops 1–6) is your route; anyone who walks up mid-story can see which stop you are on.

### 6.1 The frame (top and bottom of every page)

| On screen | What it means | Behind the scenes |
|---|---|---|
| Dark strip "Decision support on retrospective data. Not a medical device." | Safety disclaimer, part of the sticky header | Always rendered; also returned by `/api/config` |
| **Ensemble** badge next to the name | Which model is served (LightGBM + GRU ensemble, or LightGBM with `MODEL_KIND=lgbm`) | From `/api/config` (`model_kind`, `model_version`) |
| **Ward · Tour · Evidence** | The three rooms | React Router routes `/`, `/tour`, `/evidence`; a bed is `/bed/<id>` |
| **ICU hour 0 / 69** | The replay clock. All 24 patients are shown as if they were admitted at the same moment; 69 is the longest stay − 1 | The clock lives in the browser (`ReplayContext.tsx`); the backend is stateless and answers "what did the world look like at hour *t*?" |
| **Reset / ‹ 1 h / Play / 1 h ›**, **Speed** | Move through time; keyboard: Space plays/pauses, ← / → step one hour | Every change of *t* triggers a fresh request; `/?t=16&truth=1` deep-links to an hour |
| **Alert threshold** slider (1%–50%) + "reset to policy 3.0%" | The alert threshold τ; starts at the policy value 3% | Debounced 200 ms, then sent as `?threshold=`; the backend **recomputes alerts** for all beds at that threshold |
| **Show ground truth (demo)** | Reveals the recorded sepsis onset (a teaching feature; a real system wouldn't have it) | Sends `show_truth=true`; only then does the API include onset and outcome |
| **Tour rail** (bottom): 1 A false alarm · 2 Caught early · 3 Caught very early · 4 Missed · 5 The threshold · 6 The evidence | Click a stop: it pauses, sets the hour, ground truth and policy threshold, and opens the right screen. The current stop is shown "mounted" (white) | `src/tour.ts` holds only bed, hour and route; every sentence about a bed is computed from the API |

### 6.2 The ward wall (`/`): "the nurse's station view"

**Say:** "These are 24 real ICU patients from the sealed test set. The model never trained on any of them. I'm replaying their stays hour by hour."

| Column / element | Meaning |
|---|---|
| Summary (Alert / Watch / OK / Discharged) | Shape, count, word and rule (e.g. "risk ≥ 3.0%") |
| **Bed** number | The exhibit number, identical on every screen |
| **Status** | **Alert** (triangle, risk ≥ threshold), **Watch** (dot, risk ≥ half the threshold), **OK** (bar), **Discharged after hour N**. Always shape + word |
| Age / sex · unit | MICU = medical ICU, SICU = surgical ICU, "—" = not recorded |
| **Calibrated risk** bar + % | Probability right now (2 decimals); the white tick marks the threshold |
| **Last 12 h** | Sparkline of the last 12 hours of risk; dashed line = threshold |
| HR, MAP, Temp, Resp, SpO₂, WBC | Last recorded values ("—" = not measured yet) |
| **In alert** | How long the alert has been continuously on |

**Start the tour: stop 1, A false alarm.** Bed 15 is in **Alert** at hour 0 and never develops sepsis (in LightGBM mode Bed 20 alerts too). "That's a false alarm, and I've kept it in on purpose. Under the ensemble these beds come out as 6 early catches, 2 flagged outside the 4–12 h window, 2 misses, 8 quiet and 6 with a false alarm. I'm showing you the real behaviour, not a highlight reel."

**Why are risks so small (1–4%)?** "Because they're *calibrated*. Only 1.8% of hours are septic, so a 3.5% risk is genuinely high."

**Behind the scenes (per request, ~14 ms):** the backend computes `hour = min(t, stay − 1)` for each bed, looks up the **pre-computed calibrated risk** (scored once by the served model with the same code as `/api/predict`), re-applies the **alert policy at the requested threshold**, works out the level, trend, last vitals and alert duration, and returns the beds sorted by risk, discharged last.

### 6.3 A bed, mounted on its label (`/bed/:id`): "why is the model worried?"

**Stop 2, Caught early** opens **Bed 03** at hour 16 with ground truth on.

| Element | Meaning |
|---|---|
| **Bed 03** at display scale + tombstone | "Man, 52 · ICU unit not recorded · ICU hour 16 of 24 · Calibrated risk now 3.64% (3.55% in LightGBM mode) · first alert at hour 16 · in alert for 1 h" |
| **WHAT HAPPENED** (third tombstone line, ground truth on) | "Sepsis onset at ICU hour 22: the first alert came 6 h before it." The same line closes the label |
| Catalogue line | Exhibit number, held-out test patient id, served model, hospital admission time |
| Strip on the left | The wall, collapsed and dimmed; the open bed is the one mounted in white |
| **Label** tab | Risk trajectory (served risk, plus the LightGBM and GRU parts as thin lines, threshold, shaded alert hours, "now", onset), **What the model saw** (top 5 SHAP reasons, noted as coming from the LightGBM part), and all eight vitals as small charts |
| **Every measurement** tab (34) | All 8 vitals and 26 labs as small charts in clinical groups (vitals, blood gas, perfusion, chemistry, liver, blood count), then a **flowsheet**: every measurement at every hour (bold = measured that hour; tick the box to see carried-forward values), plus the risk and status rows |
| **Hour by hour** tab | Newest first, one entry per hour: risk and change since the previous hour, the LightGBM and GRU parts, status, everything measured that hour, and that hour's top reasons. Filters: all hours, alert hours, hours with new lab results |

**Say:** "He has a fever of 38.4, and the team just adjusted his oxygen. The model picks up both. Notice it's partly reacting to *what the clinicians are doing*: ordering tests is a signal of concern." Step with → a few times: the reasons change ("FiO2 last measured 3h ago", "Temperature 38.7").

**Behind the scenes:** `GET /api/patients/p003071?t=16&threshold=0.03&show_truth=true` returns every vital and lab for hours 0…*t* (all 34), the risk, alert and per-hour LightGBM/GRU series, the stored top-5 SHAP reasons for **every** hour (computed by TreeExplainer when the cohort was scored), and the onset hour, but only because `show_truth` is on.

**Stop 3, Caught very early** (Bed 16, hour 30): first alert at hour 28, onset at hour 42 (14 h). Useful, but very early alerts also cost attention. **Stop 4, Missed** (Bed 07, hour 28): onset at hour 28 with no shading anywhere. "This one it missed. On the full test set it misses about 1 in 8."

### 6.4 Stop 5, the threshold: the clinical trade-off

The ward at hour 40 (active beds):
- Drag down to **1.5%**: 5 alerts. "Catch more, earlier, but more false alarms."
- Back to **3%** (reset link): 2 alerts.
- Up to **10%**: 0 alerts. "Quiet ward, but we'd miss people."

**Say:** "Where to set this is a *clinical and staffing* decision, not a purely technical one. The model gives calibrated probabilities so a hospital can choose its own alarm budget." **Behind the scenes:** the risks don't change; only the policy (threshold → alert) is re-applied, which is why it's instant.

### 6.5 Stop 6, the evidence (`/evidence`): two rooms

| Element | Meaning | Source |
|---|---|---|
| Served-model note | "The app serves the LightGBM + GRU ensemble… its lockbox score has not been measured" | `/api/config` |
| **Lockbox test (LightGBM)** room (green wall, white labels) | Headline sentence; E1 test metrics with CIs; E2 baselines; E3 calibration and E4 ROC/PR figures (click for full size) | `final_test_metrics.csv`, `final_test_baselines.csv`, `/api/figures` |
| **Development comparison (nested 5-fold CV)** annex (slate-blue wall, dashed edge, "Development data · not lockbox") | E5 LightGBM vs GRU vs ensemble (every configuration), E6 blend-weight search, E7 paired bootstrap, E8 served configuration | `reports/tables/ensemble_dev_*.csv` via `dev_cv` |
| LightGBM, development data | E9 threshold trade-off curve (dotted = policy, solid = slider), E10 cross-hospital (R0; R1–R3 behind a disclosure), E11 model card | `threshold_curve.json`, `cross_hospital.csv`, `/api/model-card` |

**Say:** "Nothing here is recomputed. The single LightGBM test run hangs in one room, the development comparison in another. They're never in one table. The numbers in my slides, my report and this page are literally the same files."

---

## 7. What's under the hood: architecture and backend

```
OFFLINE (Python, run once)
 PhysioNet files → io → cleaning → splits → features (298) → LightGBM + Optuna
   → isotonic calibrator → policy search → models/final/  (frozen: lgbm.txt,
     calibrator.joblib, policy.json, feature_list.json, threshold_curve.json)
 features → sequences → GRU (GPU, dev folds) → nested weight/variant search on dev
   → models/ensemble_final/ (gru.pt, gru_calibrator.joblib, ensemble.json)
 Lockbox test patients → demo_cohort.py → data/demo/ (24 beds: rows, risks, SHAP reasons)

ONLINE (Docker Compose)
 Browser ──► nginx :8080 (serves React app, proxies /api) ──► FastAPI :8000
                                                         (loads models/final, models/ensemble_final
                                                          and data/demo once; GRU on CPU)
```

### Backend (FastAPI, `backend/app/`)
- **`settings.py`:** where the files live and which model to serve (`MODEL_KIND=ensemble`, the default, or `lgbm`), all overridable via environment variables.
- **`service.py`, `SepsisService`:** at startup (FastAPI *lifespan*) it loads the booster, calibrator, policy, feature list, threshold curve, a SHAP TreeExplainer, the GRU and its calibrator and `ensemble.json` (weight + policy), and the demo tables **once**, so requests are fast.
- **`schemas.py`:** Pydantic models that validate every request and response. A bad threshold (e.g. 1.5) or a negative hour gets a **422**; an unknown patient gets a **404**.
- **`main.py`, endpoints:**

| Endpoint | What it does |
|---|---|
| `GET /api/health` | liveness + served model kind and version (`ensemble-w0.7-lgbm-final-303r-<hash>+gru-<hash>`) |
| `GET /api/config` | served model, policy threshold, k, sticky, replay length, disclaimer |
| `GET /api/ward?t&threshold` | all 24 beds at hour *t* |
| `GET /api/patients/{id}?t&threshold&show_truth` | one patient's trajectory + reasons |
| `POST /api/predict` | **the real online path:** raw hourly rows → clean → build features → LightGBM → calibrator, and features → GRU (CPU) → calibrator → 0.7/0.3 blend → policy → SHAP (LightGBM) for the last hour |
| `GET /api/metrics` | `lockbox_test` (LightGBM, frozen) and `dev_cv` (nested CV comparison) as separate sections, plus cross-hospital tables |
| `GET /api/model-card` | model card |
| `GET /api/figures/{name}` | whitelisted result figures |

- **The key engineering guarantee: one code path.** `src/sepsis/inference.py` is used both to build the demo cohort and by `POST /api/predict`. An automated test sends a demo patient's raw rows to `/api/predict` and checks that the risks match the stored ones to within 10⁻⁶, for both model kinds; the ensemble's online path matches a batched pass over the precomputed features to 4.3 × 10⁻⁷. This rules out *training/serving skew*, the most common silent failure when ML goes to production.
- **Speed:** a ward query takes about 14 ms. Scoring a whole 70-hour stay from raw data: LightGBM 242 ms vs ensemble 261 ms in the same process (the CPU GRU adds about 18 ms); over HTTP on the laptop both are around 300 ms (see STATUS.md for the measured runs).

### Frontend (React + TypeScript + Recharts, `frontend/src/`)
- `api.ts`: typed fetch helpers. `state/ReplayContext.tsx`: clock, speed, threshold (debounced) and show-truth, shared by all pages.
- Pages: `WardPage` (the wall), `BedPage` (strip + mounted label with Label / Every measurement / Hour by hour tabs), `TourPage`, `EvidencePage`. Components: `RiskBar`, `Sparkline` and `MiniSeries` (hand-drawn SVG; 34 of them on a bed page stay cheap at 5× replay), `RiskChart` (Recharts), `ReasonsPanel`, `Status` (shape + word), `TourRail`, `Disclaimer`. Variable names, units and groups for all 34 measurements live in `src/variables.ts`.
- If the backend dies, it shows "Cannot reach the API at /api — is the backend running?" with a Retry button.

### Deployment (Docker)
- `backend/Dockerfile`: Python 3.11-slim + `libgomp1` (needed by LightGBM) + CPU-only PyTorch 2.6.0 for the GRU (no CUDA libraries; torch headers and package test suites removed).
- `frontend/Dockerfile`: two stages, Node 20 builds the app and nginx 1.27 serves it and proxies `/api`.
- `docker-compose.yml`: one command (`docker compose up --build`), with health checks; the UI waits until the API is healthy.

### Quality
- **93 automated tests**, including causality (features, the GRU, and the full ensemble online path), the utility scorer against hand-computed cases, the alert policy, online-vs-batch parity, and the API under both model kinds. Ruff linting is clean.

---

## 8. Why I chose what I chose: alternatives

**Be honest about status.** "Tested" means we actually ran it and have numbers. "Not tested" means the choice is justified by reasoning and the literature, not by our own experiment. Examiners respect this distinction.

### 8.1 Summary table

| Decision | What I used | Alternatives | Tested? | Why mine |
|---|---|---|---|---|
| Core model | **LightGBM** (gradient-boosted trees) | Logistic regression | ✅ (B3) | LogReg utility 0.2438 vs **0.4270** |
| | | Clinical rules SIRS / qSOFA | ✅ (B1, B2) | 0.0507 / 0.0122 |
| | | XGBoost, CatBoost | ❌ | Same model family; LightGBM is faster on 1.2M rows (histogram + leaf-wise growth) and handles NaN natively. Expect similar accuracy |
| | | Random forest | ❌ | Usually less accurate than boosting on tabular data; worse probability estimates |
| | | GRU alone | ❌ | Dev nested utility 0.3529 (hidden 64) vs LightGBM 0.4196 |
| | | **LightGBM + GRU ensemble** | ✅ served | Dev nested utility 0.4327; Δ +0.0131 [+0.0072, +0.0190]; no lockbox estimate |
| Hyperparameters | **Optuna** (TPE, 40 trials) | Defaults | ✅ | Defaults: utility 0.4183 vs tuned **0.4262** (dev) |
| | | Grid / random search | ❌ | Bayesian search finds good settings in fewer trials |
| Missing values | **Keep NaN + LOCF + "measured?" / "hours since" features** | Drop the missingness features | ✅ (ablation) | Utility 0.4262 → 0.4039 without them |
| | | Mean / median imputation, MICE, backward fill | ❌ | Imputation erases the "a test was ordered" signal; backward fill **leaks the future** |
| Feature groups | **All 10 families (298)** | Without rolling stats / deltas / scores | ✅ (ablation) | Each removal lowers utility (−0.0028 to −0.0057) |
| Calibration | **Isotonic regression** | No calibration | ✅ | Test ECE 0.0221 → **0.0027** |
| | | Platt (sigmoid) scaling, temperature scaling | ❌ | Assume a fixed curve shape; isotonic is flexible and we have >1M OOF rows, so overfitting isn't a risk |
| Alert rule | **Threshold + k-consecutive + sticky, searched for max utility** | Fixed 50% threshold | (implied) | With 1.8% prevalence, 50% would almost never fire |
| | | k = 2/3 persistence, 6-h sticky | ✅ (grid of 594) | k = 1, s = 0 maximised utility |
| Primary metric | **Official Challenge utility** | Accuracy | – | 98.2% for never alerting, so meaningless |
| | | AUROC only | reported | Ignores timing and the alert threshold |
| | | F1 | ❌ | Ignores *when* you alert; utility rewards early and penalises late |
| Validation | **Patient-level split + 5-fold CV + once-only lockbox + patient bootstrap CIs** | Random row split | ❌ (deliberately avoided) | Same patient in train and test means leakage and inflated scores |
| | | Single train/test split | ❌ | No uncertainty estimate; easy to overfit by re-testing |
| Explainability | **SHAP TreeExplainer** | LightGBM built-in importance (gain/split) | ❌ | Global only, no per-patient explanation, biased |
| | | LIME | ❌ | Approximate, unstable; TreeSHAP is exact and fast for trees |
| | | Permutation importance | ❌ | Global only; slow |
| Generalization test | **Train one hospital, test the other (R0–R3)** | Just report the pooled test score | ✅ | The pooled score hides a 32.7% drop |
| | | Per-hospital standardisation (R2), drop missingness (R1) | ✅ | Neither closed the gap |
| Backend | **FastAPI** | Flask, Django | ❌ | FastAPI gives automatic validation (Pydantic), automatic docs at `/docs`, async, and typing |
| | | Streamlit / Gradio | ❌ | Quick to build, but you can't separate API from UI or show a real deployable service |
| Frontend | **React + TypeScript + Recharts** | Streamlit, Dash, plain HTML | ❌ | Full control of a clinical UI (replay clock, synced charts); TypeScript catches errors at build time |
| Deployment | **Docker Compose** | "Run two terminals", cloud PaaS | ❌ | One command, reproducible on any machine, health checks |

### 8.2 The key ones explained (talking points)

**Why LightGBM and not deep learning?**
- The data is tabular, irregular and mostly missing. Boosted trees are the state of the art there, and most top 2019 Challenge entries used them.
- It handles missing values natively (learns which branch NaN should go down).
- It trains in minutes on a laptop, with no GPU.
- SHAP is exact for trees, so every alert can be explained.
- It's easy to calibrate.
- My hand-built temporal features (rolling windows, slopes, time-since) give trees the "memory" an RNN would learn.
- Evidence on this data (development, nested CV): a GRU alone scores 0.3529 vs LightGBM's 0.4196. Averaged with LightGBM it adds a little (0.4327), so the app serves the ensemble, with the GRU as a second opinion.

**Why not XGBoost or CatBoost?** Same family of algorithm. LightGBM's histogram-based, leaf-wise trees are typically the fastest on millions of rows. I'd expect very similar accuracy; the choice is about speed and memory, not a big accuracy gap.

**Why logistic regression as the main ML baseline?** It's the standard, interpretable clinical model. If a complex model can't beat it clearly, the complexity isn't justified. Ours beats it by 75%.

**Why isotonic, not Platt scaling?** Platt fits an S-shaped curve, which assumes a particular distortion. Isotonic learns *any* monotone correction. With over a million out-of-fold predictions there's no overfitting risk, and the result is ECE 0.0027.

**Why the utility score, not accuracy or F1?** Sepsis prediction is about *timing*. An alert two days late is worthless; an alert six hours early is gold. Accuracy and F1 treat all hours the same. The utility function was designed by clinicians for this exact problem, and it lets us compare with the international Challenge setup.

**Why keep missing values as signal instead of imputing?** In an ICU, *missing* isn't random. A lab is drawn because someone is worried. The ablation proves it: removing missingness features causes the biggest drop of any group.

**Why a patient-level split and a once-only lockbox?** Splitting by rows would put hour 10 of a patient in training and hour 11 in testing, so the model would recognise the patient. Testing many times and picking the best is a subtle form of cheating. One sealed test set, opened once, is the only honest estimate.

---

## 9. Selling points: the lines that land

1. **"Honest by design."** Sealed test set opened once; dev 0.4267 vs test 0.4270; misses and false alarms deliberately left in the demo.
2. **"It explains itself."** Every alert comes with the top 5 reasons in plain English, updated every hour.
3. **"When it says 3%, it means 3%."** Calibration error 0.0027.
4. **"Early."** 87.5% of septic patients flagged in the useful window; median first alert 29 hours before onset.
5. **"Beats the rules doctors use today."** 8.4× SIRS, +75% over logistic regression.
6. **"I found where it breaks."** A 32.7% drop when moving hospitals, and *why*: the threshold, not the ranking. That finding is more valuable than another decimal of accuracy.
7. **"It's a real product, not a notebook."** API, dashboard, Docker, 93 automated tests, and the same code path for training and serving, proven by a test.
8. **"I kept the claims apart."** The ensemble was chosen on development data with a search fixed in advance; the sealed test number stays with the model it measured.

---

## 10. Likely class questions (short answers)

Full answers are in `reports/viva_qa.md`.

| Question | Short answer |
|---|---|
| "Why is the risk only 3%? That seems low." | It's calibrated; the base rate is 1.8%, so 3% is high. Uncalibrated scores look scarier but mean nothing. |
| "Isn't AUPRC 0.137 bad?" | Chance is 0.018, so we're 7.6× chance. Rare-event AUPRC is always low. |
| "How do you know there's no data leakage?" | Patient-level splits, features use only past hours (automated prefix test), all tuning on dev data, test opened once. |
| "Why does 'lactate last measured' matter?" | Ordering a lactate signals clinical suspicion. The model learns from clinician behaviour too, which is useful but means it's partly hospital-specific. |
| "Could this be used in a real hospital?" | Not as-is. It would need local re-tuning, a silent trial, workflow design and regulatory approval. The cross-hospital result is exactly why. |
| "Why did it miss Bed 07?" | His risk never exceeded 1.3%; the signals weren't there in his data. That's the 12.5% it misses. Lowering the threshold would catch more at the cost of alarms. |
| "29 hours early sounds too good." | The low threshold means some patients are flagged early and stay flagged. The scoring rule gives no credit that early, so sensitivity in the −12 h to +3 h window (87.5%) is the fairer number. |
| "How does it compare to the Challenge winners?" | Not directly comparable: their test included an unseen third hospital. Our own A→B experiment (0.2963) shows how much harder that is. |
| "What would you do next?" | Measure the ensemble once on a new untouched dataset, alert-budget-constrained thresholds, few-shot recalibration for a new hospital, clinician review of explanations. |
| "Why not test the ensemble on the lockbox?" | It was opened once, for LightGBM. Testing a model chosen afterwards would turn it into a validation set; the ensemble's evidence is development-only and labelled that way. |

---

## 11. Demo-day checklist

- [ ] Docker Desktop running → `docker compose up --build` → both containers healthy (`docker compose ps`)
- [ ] http://localhost:8080 open, 100% zoom; badge reads **Ensemble**; replay at hour 0, speed 2×, threshold 3.0%, ground truth **off**
- [ ] Slides open: `reports/slides/slides.pptx` (or `.pdf`)
- [ ] Key beds memorised:
  - **Bed 03** alerts at h16, onset h22 (main story)
  - **Bed 07** missed, onset h28
  - **Bed 15** false alarm at h0 (Bed 20 too in LightGBM mode)
  - **Bed 16** first alert at h28, onset h42 (backup; LightGBM mode: h30)
- [ ] Fallback screen recording ready in case the laptop or Docker fails
- [ ] Close with the disclaimer: *"Decision support on retrospective data. Not a medical device."*
