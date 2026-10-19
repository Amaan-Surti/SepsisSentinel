# SepsisSentinel — Demo Video Script (3½ minutes)

Audience: clinicians and examiners. Tone: calm and factual. Every event below was checked against `data/demo/` (24 lockbox-test patients, frozen model, policy τ = 0.03).

## Before recording (setup checklist)

- [ ] `docker compose up --build` (or two terminals: `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000` and `cd frontend; npm run dev`)
- [ ] Chrome at 1280×720 or larger, zoom 100%, open http://localhost:8080 (Docker) or http://localhost:5173 (dev)
- [ ] Replay at hour 0, speed **2×**, threshold **3.0%**, "Show ground truth" **off**
- [ ] Close other tabs and notifications; screen recorder at 1080p

## Script

| Time | Screen / action | Narration |
|---|---|---|
| **0:00–0:20** | Ward page at hour 0. Point to the dark banner. | "This is SepsisSentinel, an early-warning system for sepsis. It's replaying 24 real ICU patients from a test set the model never saw, hour by hour. It's a research demonstration on retrospective data, not a medical device." |
| **0:20–0:45** | Hover the columns: risk bar, 12-h trend, vitals, level chip. Point to **Bed 15** and **Bed 20** (both *Alert* at hour 0). | "Each row is a bed, sorted by calibrated sepsis risk. A risk of 3% really means about 3 in 100 such patient-hours turn out septic; the model is calibrated, with a test calibration error of 0.003. Orange means Alert, over the 3% threshold; amber means Watch. Right at hour 0 two beds are alerting, and neither patient develops sepsis. Those are false alarms, and I'm leaving them in deliberately." |
| **0:45–1:05** | Click **Play** (2×). Let it run; Bed 06 flickers into Alert (hours 2–13). | "Let's run the clock. Risks move as new vitals and labs arrive. The system costs about one false alert per patient every 45 hours, and that trade-off was chosen on purpose." |
| **1:05–1:20** | At **hour 16**, **Bed 03** turns *Alert*. Click **Pause**. | "Hour 16: Bed 03, a 52-year-old man, crosses the threshold. Let's look." |
| **1:20–1:55** | Click Bed 03. Patient page: risk chart rising over the dashed threshold, orange shaded alert hours. Point to **Why this risk**. | "Here's his trajectory. The model gives reasons for every hour, using SHAP: an FiO2 was just recorded, and temperature is 38.4 with the 6- and 12-hour maximum at the same level. These explain what the model is reacting to, not what's causing his illness. A recent oxygen setting tells us the team is already watching him closely." |
| **1:55–2:15** | Toggle **Show ground truth**. The "Sepsis onset" line appears at **hour 22**. | "Now I'll reveal what happened. Sepsis onset, by Sepsis-3 criteria, was at hour 22, six hours after the alert. In the real world that's six hours to draw cultures, start antibiotics and fluids." |
| **2:15–2:35** | Back to Ward. Click **Bed 07** (show-truth still on): onset hour 28, no shaded alert hours. | "It doesn't catch everyone. Bed 07 became septic at hour 28 and never crossed the threshold. On the full test set the system flags 87.5% of septic patients in the useful window, so about one in eight is missed." |
| **2:35–3:05** | Drag the **threshold slider** down to ~1.5%: more beds turn Alert. Then up to ~10%: almost none. Open the **Model** page and point at the trade-off chart. | "The threshold is a clinical choice. Lower it and you catch more patients earlier, but with more alarms; raise it and alarms drop but you miss more. The Model page shows the trade-off curve and the frozen test results: utility 0.427, AUROC 0.85, sensitivity 87.5%, and a median first alert 29 hours before onset. It's well ahead of the SIRS and qSOFA rules." |
| **3:05–3:25** | Scroll to **Cross-hospital** table. | "One important caveat: a model trained at one hospital lost up to a third of its utility at the other, mainly because the right threshold differs between sites. Any real deployment would need local tuning and a silent trial first." |
| **3:25–3:35** | Scroll to the footer disclaimer. | "SepsisSentinel is decision support on retrospective data, not a medical device. Thank you." |

## Backup moments (if something goes off-script)

- **Bed 16** (septic, onset hour 42): single alert at **hour 30**, 12 h before onset. Reasons: lactate drawn 2 h ago, temperature 38.2. Then quiet until hour 43. Useful to show an alert that comes and goes.
- **Bed 11** (septic, onset hour 23): alerting continuously from **hour 17**.
- **Bed 19** (septic, onset hour 48): highest risk in the cohort (17.2%), alerting from hour 43.
- **Bed 12**: second missed septic patient (onset hour 44, maximum risk 2.6%).

## Fallback

Record a **60-second screen capture** in advance (hour 0 → Play → pause at Bed 03 → reasons → show truth → Model page). If the laptop, Docker or the browser fails during the live demo, play the recording and continue the narration from the table above.
