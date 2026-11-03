---
version: 1
slug: "frontend-src-main-tsx"
primary_target: "frontend/src/main.tsx"
related_targets: ["frontend/src"]
---

# Surface brief: SepsisSentinel dashboard (frontend/)

Scope: the whole replay dashboard (ward, bed record, tour, evidence). Mode: Operate.
Audience and job: exhibition visitors at the presenter's laptop, arriving mid-story; the presenter drives the replay and answers examiners. Task: see which beds the model flags at this hour, open one, understand why, reveal what happened, and see the evidence.
Content: every feature of the P9 dashboard plus all 34 hourly measurements (8 vitals, 26 labs) and hour-by-hour insights (risk, LightGBM and GRU components, alert state, top reasons) for every hour.
Constraints: numbers only from the API; lockbox (LightGBM) and development (ensemble) results never share a table; status by shape + word; disclaimer on every page; WCAG AA; keyboard; 1280×720 to 1920×1080; React 18 + TS + Recharts, no heavy deps.
Memorable moment: the object label's last line, WHAT HAPPENED, revealing onset beside the alert that came hours earlier.
Unresolved: self-hosting the fonts for offline use (needs the user's approval to download font files).
Recorded adaptations (finish review, 2026-11-03): (1) the Ward route is the full gallery wall at full width (every bed with risk, trend, six vitals, status) because the user asked for every detail; the mounted label lives on the bed route, where the wall collapses to a dimmed strip. (2) The current tour stop is mounted (label white) instead of brass, which would collide with the ochre Watch state. (3) The evidence page's development results hang in a separate annex with its own slate-blue wall and dashed edge, so lockbox and development numbers can never read as one room. (4) The WHAT HAPPENED line also appears as the third tombstone line so it is on the first screen.

## Direction contract

THESIS: Every bed is a numbered exhibit on a gallery wall; the open bed is mounted on an object label. Refuses the dark neon ICU-monitor dashboard and the white SaaS card grid.

OWN-WORLD: Deep verdigris gallery wall (#203E3A family) as the shell and ground; warm-white mounted label boards only for the exhibit in focus; ink near-black; state colours vermilion (Alert) and ochre (Watch) used only for state; the presenter's current tour stop mounted in label white; a slate-blue annex wall for development-only evidence. Atkinson Hyperlegible Next for all text on one strict five-step scale, Atkinson Hyperlegible Mono for codes and catalogue numbers. Exhibit numerals large and heavy.

STORY: A visitor sees a ranked wall of numbered beds and one mounted label, reads who the patient is and what the model saw, watches the risk cross the threshold, then reads WHAT HAPPENED. An examiner opens the full record (every measurement, every hour) and the evidence room.

FIRST VIEWPORT: Dark disclaimer strip; wall-green top bar with brand, model badge, nav (Ward · Tour · Evidence) and the replay clock and controls at right. Left 380 px: the ward wall, numbered exhibits with status shape + word and risk. Right: the mounted label (Bed number at display scale, tombstone line, risk trajectory, what the model saw, WHAT HAPPENED). Bottom: numbered tour rail, current stop in brass.

FORM: Gallery Label (museum object labels), candidate 5 of 7 in my ordered list, assigned by the roll. seed:1cb1bf224b161401. Raises: strict five-step type scale (specimen); one catalogue number per bed everywhere (catalogue sleeve); focus dims the wall (streaming wall); only the focused exhibit is mounted (acetate manual).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
