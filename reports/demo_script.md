# SepsisSentinel — Demo Video Script (3½ minutes)

Audience: clinicians and examiners. Tone: calm and factual. Every event below was checked against `data/demo/demo_predictions.parquet` (24 lockbox-test patients scored by the served LightGBM + GRU ensemble, policy τ = 0.03). The top bar shows an **Ensemble** badge. With `MODEL_KIND=lgbm` the app serves LightGBM alone (`demo_predictions_lgbm.parquet`); the beats below hold for both except where noted.

## Before recording (setup checklist)

- [ ] `docker compose up --build` (or two terminals: `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000` and `cd frontend; npm run dev`)
- [ ] Chrome at 1280×720 or larger, zoom 100%, open http://localhost:8080 (Docker) or http://localhost:5173 (dev)
- [ ] Replay at hour 0, speed **2×**, threshold **3.0%**, "Show ground truth" **off**; badge reads **Ensemble**
- [ ] Shortcuts: Space = play/pause, ← / → = one hour; `/?t=16` opens the replay at hour 16
- [ ] Close other tabs and notifications; screen recorder at 1080p

## Script

| Time | Screen / action | Narration |
|---|---|---|
| **0:00–0:20** | Ward page at hour 0. Point to the dark banner. | "This is SepsisSentinel, an early-warning system for sepsis. It's replaying 24 real ICU patients from a test set the model never saw, hour by hour. As the banner says: decision support on retrospective data, not a medical device." |
| **0:20–0:45** | Point to the summary chips, then the columns: risk bar, 12-h trend, vitals, status chip. Point to **Bed 15** (*Alert* at hour 0). | "Each row is a bed, sorted by calibrated sepsis risk. A risk of 3% means about 3 in 100 such patient-hours turn out septic. Alert, with a triangle, means over the 3% threshold; Watch, with a dot, means over half of it. Right at hour 0 one bed is alerting, Bed 15, and that patient never develops sepsis. It's a false alarm, and I'm leaving it in deliberately." *(LightGBM mode: Beds 15 and 20.)* |
| **0:45–1:05** | Press **Space** to play (2×). Let it run; Bed 06 flickers in and out of Alert from hour 2. | "Let's run the clock. Risks move as new vitals and labs arrive. On the test set the LightGBM model costs about one false alert per patient every 45 hours, and that trade-off was chosen on purpose." |
| **1:05–1:20** | At **hour 16**, **Bed 03** turns *Alert*. Press **Space** to pause. | "Hour 16: Bed 03, a 52-year-old man, crosses the threshold. Let's look." |
| **1:20–1:55** | Click Bed 03. Patient page: risk chart rising over the dashed threshold, shaded alert hours. Point to **Why this risk** and its note. | "Here's his trajectory. The served model averages two models, a gradient-boosted tree model and a recurrent network. The reasons come from the tree part, using SHAP: an FiO2 was just recorded, and temperature is 38.4 with the 6- and 12-hour maximum at the same level. These explain what the model is reacting to, not what's causing his illness. A recent oxygen setting tells us the team is already watching him closely." |
| **1:55–2:15** | Toggle **Show ground truth**. The "Sepsis onset" line appears at **hour 22**. | "Now I'll reveal what happened. Sepsis onset, by Sepsis-3 criteria, was at hour 22, six hours after the alert. In the real world that's six hours to draw cultures, start antibiotics and fluids." |
| **2:15–2:35** | Back to Ward. Click **Bed 07** (show-truth still on): onset hour 28, no shaded alert hours. | "It doesn't catch everyone. Bed 07 became septic at hour 28 and never crossed the threshold. On the full test set the system flags 87.5% of septic patients in the useful window, so about one in eight is missed." |
| **2:35–3:05** | Drag the **threshold slider** down to ~1.5%: more beds turn Alert. Then up to ~10%: almost none. Open the **Model** page: point at the white **Lockbox test (LightGBM)** band, then the dashed **Development comparison** band. | "The threshold is a clinical choice: lower catches more patients earlier with more alarms; higher means fewer alarms and more misses. On the held-out test set, opened once, the LightGBM model scored utility 0.427, sensitivity 87.5%, with a median first alert 29 hours before onset — well ahead of the SIRS and qSOFA rules. The app now serves an ensemble of that model and a recurrent network. It was chosen on development data, where it beat LightGBM by 0.013 utility with fewer false alarms. We have not tested it on the held-out set, and the page keeps those numbers apart." |
| **3:05–3:25** | Scroll to **Cross-hospital** table. | "One important caveat: a model trained at one hospital lost up to a third of its utility at the other, mainly because the right threshold differs between sites. Any real deployment would need local tuning and a silent trial first." |
| **3:25–3:35** | Scroll to the footer disclaimer. | "SepsisSentinel is decision support on retrospective data, not a medical device. Thank you." |

## Backup moments (if something goes off-script)

- **Bed 16** (septic, onset hour 42): first alert at **hour 28**, 14 h before onset, on and off until onset. At hour 30 the reasons are lactate drawn 2 h ago and temperature 38.2. *(LightGBM mode: a single alert at hour 30, then quiet until hour 43.)*
- **Bed 11** (septic, onset hour 23): alerting from **hour 17** (one quiet hour at 18).
- **Bed 19** (septic, onset hour 48): highest risk in the cohort (24.6% at hour 49), alerting from hour 42. *(LightGBM mode: 17.2%, from hour 43.)*
- **Bed 12**: second missed septic patient (onset hour 44, maximum risk 2.4%; LightGBM 2.6%).

## Fallback

Record a **60-second screen capture** in advance (hour 0 → Play → pause at Bed 03 → reasons → show truth → Model page). The deep link `/?t=16&truth=1` jumps straight to the Bed 03 moment. If the laptop, Docker or the browser fails during the live demo, play the recording and continue the narration from the table above.
