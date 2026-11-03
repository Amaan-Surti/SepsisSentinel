# UI review — impeccable critique + audit (P9, 2026-11-03)

Target: the P7 dashboard (master, Docker image 1.0) at 1280×720 and 1440×900, Ward / Patient (Bed 16, hour 30, ground truth on) / Model.
Method: impeccable `critique` with two isolated sub-agents (A: design review; B: detector + browser evidence) and a separate `audit` sub-agent. Before/after screenshots: `reports/screenshots/before/*.jpg` and `reports/screenshots/p9/*.jpg` (the P9 after state; `reports/screenshots/*.jpg` now show the Gallery Label redesign below) (captured with `reports/screenshots/capture.py`, headless Chrome over CDP).

Goal from the brief: a calmer, clearer clinical dashboard that a class can read from a projector — every number unchanged and traceable to the API, status never by colour alone, the disclaimer on every page, usable at 1280×720 and 1440×900, keyboard accessible, WCAG AA.

## Scores before the changes

| Assessment | Score | Band |
|---|---|---|
| Critique (Nielsen, 10 heuristics) | 24 / 40 | Acceptable |
| Audit (a11y, performance, theming, responsive, integrity) | 12 / 20 | Acceptable |
| Detector (`impeccable detect`) | 3 warnings, all `side-tab` on the ward summary counters | — |

Heuristics: status 3, real world 3, control 3, consistency 2, error prevention 3, recognition 2, flexibility 1, minimalism 2, error recovery 2, help 3.
Audit: accessibility 2, performance 3, theming 2, responsive 2, integrity 3.

## Findings and what was done

| # | Finding (source) | Severity | Change (impeccable command) |
|---|---|---|---|
| 1 | Not readable from a projector: 14 px base, 10–12 px labels, ticks and units (9.8 px minimum), 8 px risk bars, 13 % alert shading (A, B, audit P2-5) | P0 | 16 px base, 13 px floor for labels, 12–13 px chart ticks, 28 px clock, 10 px risk bars, 3 px risk line, 2 px threshold line, 20 % alert shading, larger sparklines (`typeset`, `polish`) |
| 2 | Required disclaimer only in the footer; banner wording different and scrolls away (A, B) | P1 | Banner now starts with “Decision support on retrospective data. Not a medical device.” verbatim and is part of the sticky header on every page; footer repeats it (`clarify`) |
| 3 | Contrast below 4.5:1 — OK chip 3.8, discharged rows 2.3, orange text 3.7, active nav 4.3 (B, audit P1-4) | P1 | Ink tokens: `--alert-ink`, `--watch-ink`, `--ok-ink`, `--teal-ink`; discharged rows use `--muted` on a grey row with the word “Discharged” (`polish`) |
| 4 | `body { min-width: 1200px }` blocks reflow at 200 % zoom / 1024-wide projectors (audit P1-1) | P1 | Removed; top bar wraps, tables scroll inside their card, grids use `auto-fit`, two-column layouts collapse below 1180 px (`layout`) |
| 5 | Ward rows were focusable `<tr>`s with an `aria-label` that hid the cells and said “ALERT” for discharged beds (audit P1-2) | P1 | Bed name is a real link; row `aria-label` removed; `<caption>` and `scope="col"` added (`harden`) |
| 6 | Clicking the “Threshold” label fired the “policy” reset button (audit P1-3) | P1 | Button moved outside the `<label>`; slider has `aria-valuetext` (“3.0%”) (`harden`) |
| 7 | “3.0 % Alert” next to “3.0 % Watch” (1-decimal rounding); ICU-hour column repeated the clock; vitals without units (A) | P1 | Risk shown to 2 decimals everywhere; ICU-hr column removed (discharged beds say “Discharged after hour N”); units in headers; “Alert since / for 1 h” → “In alert / 1 h” (`clarify`) |
| 8 | Model page had no hierarchy: same metrics three times, 16-row cross-hospital table, full model card; operating point off the curve (A) | P2 | Headline sentence (values from the API) replaces the KPI tiles; “Lockbox test (LightGBM)” band; separate dashed, tinted “Development comparison (nested 5-fold CV)” band tagged “Development data · not lockbox”; R1–R3 and the model card behind disclosures (`layout`, `distill`) |
| 9 | No shortcuts, no deep links (A) | P2 | Space = play/pause, ← / → = step one hour; `?t=<hour>&truth=1` deep links (`harden`) |
| 10 | `h1` → `h3` skips, no page titles, charts silent to screen readers, `aria-label` on role-less elements (audit P2-1–P2-3) | P2 | Card titles are `h2`; per-route `document.title`; risk chart has a generated text summary; `role="group"` on the controls (`harden`) |
| 11 | Chart colours copied into JS; ~27 hex literals outside tokens (audit P2-6) | P2 | Charts read the CSS tokens (`src/chartColors.ts`); new tokens for every tint (`polish`) |
| 12 | Unicode glyphs as icons, hard-offset brand shadow using the alert colour, coloured side stripes (craft floor; detector `side-tab`) | P3 | Authored SVG status shapes (triangle / circle / bar) and transport icons; plain brand mark; stripes replaced by tinted summary chips with a shape, count, word and rule |
| 13 | Model page figures loaded eagerly at full size (audit P2-7) | P3 | `loading="lazy"` with intrinsic width/height; context value memoised |

Not changed, on purpose: the threshold slider keeps its 0.01–0.50 range (the policy τ = 0.03 lies below the plan's 0.05 minimum; recorded in STATUS.md since P7); fonts still load from Google Fonts (self-hosting would mean downloading font files — left as a follow-up); the light theme stays (projector use).

---

# Redesign: Gallery Label (2026-11-03, branch `feature/ui-redesign`)

After the P9 refinements the author asked for "a completely new better UI", to be shown live on his laptop at an exhibition, with **every** feature and every detail: all vitals and all labs for every hour, and an insight for every hour. Run through impeccable's new-work flow:

1. **Product truth:** `PRODUCT.md` written from a short interview (exhibition booth, visitors arrive mid-story, structure may change, must not look like a startup SaaS product).
2. **Direction round:** seven candidate worlds from the visitors' world (observation chart, weather forecast, Swiss pharma information design, hospital wayfinding, museum exhibit labels, journal visual abstract, departure board); the concept-seed roll assigned **museum exhibit labels**; four catalogue challengers were declined and two kept as alternates. Five sketches were shown on the decision page; the author picked the assigned **Gallery Label** sketch. Contract: `.impeccable/surfaces/frontend-src-main-tsx.md` (seed `seed:1cb1bf224b161401`).
3. **Build (code-led):** ward wall, bed label with *Label / Every measurement / Hour by hour* tabs (all 34 measurements; per-hour risk, LightGBM and GRU parts, status, measurements and reasons), six-stop tour, evidence rooms. API extended: `/api/patients/{id}` now returns all 8 vitals and 26 labs, `reasons_by_hour`, `risk_components` and `hosp_adm_time`.
4. **Finish review** (fresh agent following impeccable's finish-reviewer procedure; the shipped reviewer agent type is not available in this harness): disposition **fix** with eight material findings. Round 1 resolved five (status clipped at 1280, "now" painted in the Watch colour, invisible flowsheet group headings, nested flowsheet scroll, vital tiles wrapping); round 2 resolved the reveal below the fold at 1280×720, the undersized calibration figure and a tombstone-orphan regression. One capture was rejected as invalid (lazy image not loaded) and recaptured.
5. **Open:** fonts still load from Google Fonts; self-hosting them (for an offline exhibition laptop) needs the owner's approval to download the font files. Until then the page falls back to Arial without internet.
6. **Detector:** `impeccable detect` 0 findings. `DESIGN.md` + `.impeccable/design.json` record the built world.

Screenshots: `reports/screenshots/{ward,patient,measurements,hours,tour,model,model_full}.jpg` (Gallery Label); `reports/screenshots/p9/` (the refined P9 UI); `reports/screenshots/before/` (P7).
