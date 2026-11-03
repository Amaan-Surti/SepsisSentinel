# UI review — impeccable critique + audit (P9, 2026-11-03)

Target: the P7 dashboard (master, Docker image 1.0) at 1280×720 and 1440×900, Ward / Patient (Bed 16, hour 30, ground truth on) / Model.
Method: impeccable `critique` with two isolated sub-agents (A: design review; B: detector + browser evidence) and a separate `audit` sub-agent. Before/after screenshots: `reports/screenshots/before/*.jpg` and `reports/screenshots/*.jpg` (captured with `reports/screenshots/capture.py`, headless Chrome over CDP).

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
