---
name: SepsisSentinel
description: Sepsis early-warning replay of 24 held-out ICU stays, hung as numbered exhibits on a gallery wall.
colors:
  wall: "#203e3a"
  wall-deep: "#152b28"
  wall-raised: "#2a524c"
  wall-line: "#3c6660"
  wall-ink: "#eef4f2"
  wall-muted: "#aac5be"
  label: "#f8f8f4"
  label-2: "#eeefe8"
  ink: "#131918"
  ink-muted: "#4d5956"
  rule: "#d5d9d0"
  alert: "#c8452c"
  alert-ink: "#9c3218"
  alert-soft: "#fbe4dc"
  alert-on-wall: "#ffa68e"
  watch: "#d9a520"
  watch-ink: "#6f4e09"
  watch-soft: "#f6eac8"
  watch-on-wall: "#f0c75a"
  gru: "#6b5aa6"
  annex: "#27324d"
  annex-line: "#8d9bc6"
  annex-label: "#eef1f8"
  annex-ink: "#233056"
typography:
  display:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, Arial, sans-serif"
    fontSize: "clamp(56px, 9vh, 80px)"
    fontWeight: 800
    lineHeight: 0.9
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, Arial, sans-serif"
    fontSize: "32px"
    fontWeight: 800
    lineHeight: 1
  title:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, Arial, sans-serif"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.35
  body:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    lineHeight: 1.45
  catalogue:
    fontFamily: "Atkinson Hyperlegible Mono, Consolas, monospace"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "0.02em"
rounded:
  sm: "8px"
  md: "14px"
  pill: "999px"
spacing:
  inline: "8px"
  stack: "14px"
  block: "18px"
  section: "24px"
  gutter: "28px"
  gutter-narrow: "16px"
components:
  button:
    backgroundColor: "transparent"
    textColor: "{colors.wall-ink}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
    height: "40px"
  button-hover:
    backgroundColor: "{colors.wall-raised}"
    textColor: "{colors.wall-ink}"
  button-primary:
    backgroundColor: "{colors.label}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
    height: "40px"
  button-on-label:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
  button-on-label-hover:
    backgroundColor: "{colors.label-2}"
  nav-link:
    textColor: "{colors.wall-muted}"
    rounded: "{rounded.sm}"
    padding: "7px 14px"
  nav-link-active:
    backgroundColor: "{colors.label}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "7px 14px"
  label-board:
    backgroundColor: "{colors.label}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "26px 30px 30px"
  annex-label:
    backgroundColor: "{colors.annex-label}"
    textColor: "{colors.annex-ink}"
    rounded: "{rounded.md}"
    padding: "20px 22px"
  chip-alert:
    backgroundColor: "{colors.alert-soft}"
    textColor: "{colors.alert-ink}"
    rounded: "{rounded.pill}"
    padding: "5px 16px"
  chip-watch:
    backgroundColor: "{colors.watch-soft}"
    textColor: "{colors.watch-ink}"
    rounded: "{rounded.pill}"
    padding: "5px 16px"
  chip-ok:
    backgroundColor: "{colors.label-2}"
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.pill}"
    padding: "5px 16px"
  status-alert-on-wall:
    textColor: "{colors.alert-on-wall}"
  status-watch-on-wall:
    textColor: "{colors.watch-on-wall}"
  status-alert-on-label:
    textColor: "{colors.alert-ink}"
  status-watch-on-label:
    textColor: "{colors.watch-ink}"
  tour-stop:
    backgroundColor: "transparent"
    textColor: "{colors.wall-ink}"
    rounded: "{rounded.pill}"
    padding: "4px 16px 4px 4px"
  tour-stop-current:
    backgroundColor: "{colors.label}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "4px 16px 4px 4px"
  strip-bed-focus:
    backgroundColor: "{colors.label}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 10px"
  room-tag-lockbox:
    backgroundColor: "{colors.label}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "4px 14px"
  room-tag-annex:
    backgroundColor: "{colors.annex-label}"
    textColor: "{colors.annex-ink}"
    rounded: "{rounded.pill}"
    padding: "4px 14px"
---

# Design System: SepsisSentinel

## Overview

**Creative North Star: "The Gallery Label"**

The whole dashboard is a gallery. The shell is a painted verdigris wall on which every ICU bed hangs as a numbered exhibit; only the exhibit in focus is mounted on a warm-white object label, the way a museum mounts the one piece you are standing in front of. The label reads like a museum tombstone: a heavy bed number at display scale, a tombstone line (who, which unit, which hour, the calibrated risk now), a monospaced catalogue line, and at the foot the line the visitor waits for, WHAT HAPPENED, revealing onset beside the alert that came hours earlier.

Density follows the gallery's two distances. From a metre away the wall is legible as numbers, status shapes and words; up close the label carries everything an examiner asks for, every measurement and every hour, behind tabs on the same board. The system refuses the dark neon ICU-monitor dashboard and the white SaaS card grid: there are no glowing traces, no gradient tiles, no rows of identical white cards on grey. Colour is architectural (wall, label, annex) except for two state colours that appear only where they mean state.

Evidence hangs in rooms. The single lockbox test of LightGBM hangs in an outlined room on the main wall; development results for the served ensemble hang in a separate annex with its own slate-blue wall and dashed edge, so the two kinds of number can never be read as one room.

**Key Characteristics:**
- Verdigris wall shell; warm-white label board only for the focused exhibit or an evidence plate.
- Vermilion (Alert) and ochre (Watch) are state colours only, each with an on-wall and an on-label ink variant.
- Status is always shape plus word: triangle Alert, circle Watch, bar OK, the word Discharged.
- One family, Atkinson Hyperlegible Next, on a strict five-step scale (14 · 17 · 22 · 32 · clamp 56–80); Atkinson Hyperlegible Mono for catalogue numbers and codes.
- One catalogue number per bed, identical on the wall, the strip, the label and the tour.
- Opening a bed collapses the wall to a dimmed strip; the focused bed is the only mounted item.
- A numbered tour rail runs along the foot of every page; the current stop is mounted in label white.

## Colors

A deep, low-chroma verdigris architecture with a warm-white mount, a slate-blue annex, and two saturated state colours kept strictly for state.

### Primary
- **Verdigris Wall** (`wall`): the ground of every page, the top bar, and the open ward table. All page-level headings sit on it in white.
- **Deep Verdigris** (`wall-deep`): the disclaimer strip, the tour rail, and empty risk-bar tracks; the darker band that frames the wall top and bottom.
- **Raised Verdigris** (`wall-raised`): hover on wall rows, wall buttons and nav links; the numeral disc of an inactive tour stop.
- **Verdigris Line** (`wall-line`): every rule, row divider and outline drawn on the wall (control borders, summary tiles, the lockbox room edge, tour items).
- **Wall Ink** (`wall-ink`): body text and sparkline strokes on the wall; also the focus outline colour on the wall.
- **Lichen Muted** (`wall-muted`): secondary text on the wall (column heads, captions, field labels, discharged rows), and the OK risk-bar fill.

### Secondary (state only)
- **Vermilion** (`alert`): Alert fill on label boards: risk-bar fill, SHAP contribution bars, and the chart's shaded alert hours (at 0.18 opacity).
- **Vermilion Ink** (`alert-ink`): Alert text and the dashed threshold line on a label; the raise arrow in "What the model saw".
- **Vermilion Wash** (`alert-soft`): background of the Alert chip on a label.
- **Vermilion On Wall** (`alert-on-wall`): Alert status word and glyph, risk fill, sparkline threshold and alert dot on the wall; error state text.
- **Ochre** (`watch`): Watch risk-bar fill on a label.
- **Ochre Ink** (`watch-ink`): Watch text on a label.
- **Ochre Wash** (`watch-soft`): background of the Watch chip.
- **Ochre On Wall** (`watch-on-wall`): Watch status and risk fill on the wall; text selection highlight.

### Tertiary
- **Slate Annex** (`annex`): the wall of the development annex on the evidence page.
- **Annex Line** (`annex-line`): the annex's dashed 2.5px edge.
- **Annex Label** (`annex-label`): the plate colour for development-only evidence and its room tag.
- **Annex Ink** (`annex-ink`): text on annex plates.
- **GRU Violet** (`gru`): one chart series only, the dotted GRU component line in the risk chart and its legend key.

### Neutral
- **Label White** (`label`): the mounted object label, evidence plates, the active nav link, the primary Play button, the focused bed in the strip, the current tour stop, the lockbox room tag.
- **Label Shade** (`label-2`): secondary fill on a label: empty mini-series, selected flowsheet column, current hour row, OK chip, reason pills, tab hover.
- **Ink** (`ink`): text on labels, the label's heavy head and foot rules, the sepsis-onset line in the chart.
- **Ink Muted** (`ink-muted`): secondary text on labels, chart axes, the LightGBM component line, OK risk fill on a label.
- **Label Rule** (`rule`): hairlines and borders inside a label (tables, mini cards, tab underline, chart grid).

### Named Rules
**The State-Only Rule.** Vermilion and ochre mean Alert and Watch and nothing else. No decorative use, no brand use, no "highlight" use. If a thing is not in Alert or Watch, it does not get those hues.

**The Two Inks Rule.** Every state colour has a wall variant and a label variant: `alert-on-wall` / `watch-on-wall` on verdigris, `alert-ink` / `watch-ink` (text) and `alert` / `watch` (fills) on a label. Never put the on-wall variant on a label or the reverse; contrast fails both ways.

**The One Mount Rule.** Label white marks the one thing in focus: the open bed, the current tour stop, the active page, the primary action (Play). Never use it as a generic card background on the wall for unfocused items. Evidence plates are the exception because each plate is itself an exhibit.

## Typography

**Display Font:** Atkinson Hyperlegible Next (with Atkinson Hyperlegible, Arial, sans-serif)
**Body Font:** Atkinson Hyperlegible Next (same stack)
**Label/Mono Font:** Atkinson Hyperlegible Mono (with Consolas, monospace)

**Character:** A single low-vision-designed humanist family carries everything from the 80px bed number to 14px axis ticks, so the wall reads at arm's length in a bright hall; the mono cut is the catalogue card's typewriter. Loaded weights: 400, 700, 800 (Next) and 400, 600 (Mono).

### Hierarchy
- **Display** (800, `clamp(56px, 9vh, 80px)`, 0.9, -0.02em): the bed number on the mounted label only. The step tracks viewport height so a 720px-tall laptop keeps the risk chart in view.
- **Headline** (800, 32px, 1): page titles, room titles, the replay clock value, bed numbers on the wall table, ward summary counts.
- **Title** (800, 22px, 1.35): brand name, section titles on a label, fact values, mini-series current values, hour numbers. The tombstone line uses this size at 400 with 800 for the figures.
- **Body** (400, 17px, 1.45): running text, controls, table cells, tab labels (700). Notes and legends cap at 110ch; evidence headlines and markdown at 80ch.
- **Label** (700, 14px): column heads, field labels, units (400), captions and sub-lines (400), tour-rail heading. The chart axis ticks also use 14px.
- **Catalogue** (Mono 400, 14px, 0.02em): the exhibit catalogue line (set in capitals by content), SHAP feature codes, evidence plate numbers E1–E11 (600).

### Named Rules
**The Five Steps Rule.** Only five sizes exist: 14, 17, 22, 32 and the display clamp. A new element picks one of them; there is no 12, 15, 18, 20 or 24.

**The Tabular Figures Rule.** Every number that can change with the replay clock (risk, vitals, metrics, hours) is set with tabular figures so columns do not shimmer as the clock plays.

**The Run-In Key Rule.** The only letter-spaced capitals in body type are the inline key `WHAT HAPPENED` (14px, 800, 0.06em) that opens the reveal sentence on the same line. It is never a separate eyebrow above a heading.

## Layout

A single page frame (max 1880px, centred, 28px side gutters, 24px top) under a sticky two-band header: the deep disclaimer strip, then the wall-green top bar with brand, model badge, nav (Ward · Tour · Evidence) and the replay controls pushed right. A fixed tour rail sits along the bottom of every page; the footer reserves 84px beneath its content so the rail never covers it.

- **Ward:** the full gallery wall at full width: a summary row of outlined count tiles, then one table of all beds ranked by risk (catalogue number, status, risk bar with threshold tick, 12-hour sparkline, six vitals, hours in alert).
- **Bed:** a two-column exhibit grid, a 300px wall strip (sticky, scrolls on its own) beside the mounted label. Inside the label, the overview is chart + 380px "What the model saw" column, then a fluid grid of vital mini-series (min 220px).
- **Evidence:** rooms stacked vertically, each with a two-column grid of plates (18px gap).
- **Rhythm:** 8px inline gaps, 14px stack, 18px between plates, 24px between sections, 28px gutters.

Responsive behaviour: at 1366px and below the ward table tightens (narrower risk bar, sparkline 92px); at 1240px the bed overview and evidence plates go to one column and the strip narrows to 240px; at 900px the strip moves below the label, the header stops being sticky, gutters drop to 16px and the label head stacks. At 760px viewport height or less, the tour rail, label padding and tabs compact so the threshold crossing on the risk chart stays above the rail at 1280×720.

### Named Rules
**The Focus Dims The Wall Rule.** When a bed is open, the wall collapses to a strip at 0.86 opacity; only the focused bed is mounted in label white. Hover lifts a strip item to full opacity.

**The One Catalogue Number Rule.** A bed has one number, the same two digits on the wall table, the strip, the label's display numeral, the catalogue line and the tour. Never introduce a second identifier in display type.

## Elevation & Depth

The wall is flat; depth comes from mounting. Everything on the wall is drawn with 1–1.5px verdigris lines and tonal hover fills. The label board alone is lifted off the wall with one soft, dark ambient shadow, so mounting reads as physical. Inside the label, hierarchy is carried by heavy ink rules (1.5px under the label head and above the facts, 2.5px above WHAT HAPPENED), not by shadows.

### Shadow Vocabulary
- **Mount** (`box-shadow: 0 14px 36px rgba(6, 18, 16, 0.35)`): the object label and the lockbox evidence plates. Nothing else on the wall casts a shadow.
- **Tab underline** (`box-shadow: inset 0 -3px 0 var(--ink)`): the selected label tab; a structural stroke, not elevation.
- **Current column** (`box-shadow: inset 2px 0 0 var(--ink), inset -2px 0 0 var(--ink)`): brackets the current replay hour in the flowsheet.

### Named Rules
**The Mounted Not Floating Rule.** Only mounted label boards cast the Mount shadow. Wall items, tiles, rows and tour stops are flat and outlined; annex plates are flat on their slate wall.

## Shapes

Gently rounded boards and controls on a disciplined three-step radius: 14px for label boards, evidence plates and tour items; 8px for buttons, selects, nav links, strip items, mini-series cards and figures; full pills for chips, badges, room tags, tour stops and reason pills. Status glyphs are drawn on a 16-unit grid with a 1.75 stroke: a filled triangle (Alert), a filled circle (Watch), a 2.25 bar (OK). Risk bars are 10px tall with a 5px radius and a 2px × 18px threshold tick that overshoots the track. The annex is distinguished by its dashed 2.5px edge, the lockbox room by a solid 1.5px verdigris line.

## Components

### Buttons
Wall vocabulary: quiet, outlined, and large enough to hit from a standing position.
- **Shape:** gently rounded (8px), 1.5px border, minimum height 40px, 6px × 12px padding, icon + word with 6px gap.
- **Default (on wall):** transparent with `wall-line` border and `wall-ink` text; hover fills `wall-raised` and lightens the border to `wall-muted`. Disabled at 0.4 opacity.
- **Primary:** label white with ink text at 800, used for Play/Pause and for the go-to-stop action on non-current Tour page items; hover goes to pure white.
- **On a label:** ink text with `rule` border; hover fills `label-2`.
- **Link button:** borderless, underlined, 14px `wall-muted` ("reset to policy").
- **Focus:** 3px solid outline offset 2px, `wall-ink` on the wall and `wall` on labels.

### Chips
- **Style:** pill (999px), 5px × 16px, 800 weight, glyph + word. Alert: `alert-soft` / `alert-ink`. Watch: `watch-soft` / `watch-ink`. OK and Discharged: `label-2` / `ink-muted`.
- **State:** chips live on labels only; on the wall the same status is rendered as bare glyph + word in the on-wall ink.

### Cards / Containers
- **Object label:** `label` board, 14px radius, 26px 30px 30px padding (20px 18px 24px under 900px), Mount shadow. Head = display numeral + tombstone + catalogue line, closed by a 1.5px ink rule; tabs (Label · Every measurement · Hour by hour) underline in ink.
- **Evidence plate (lockbox):** the same board at 20px 22px padding, numbered E1–E11 in mono.
- **Annex plate:** `annex-label` board, `annex-ink` text, no shadow, on the slate annex wall.
- **Mini-series card:** white field, 1px `rule` border, 8px radius; dashed on `label-2` when the measurement has not been taken yet.

### Inputs / Fields
- **Select:** the button vocabulary (transparent, 1.5px `wall-line`, 8px radius, 40px).
- **Threshold slider and ground-truth checkbox:** native controls tinted with `accent-color: label`; checkbox 20px. The current threshold value sits beside the field label in white 17px.

### Navigation
- **Top nav:** three links, 700 weight, `wall-muted`, 7px × 14px, 8px radius; hover white on `wall-raised`; the active page is mounted in label white with ink text (Ward stays active on bed routes).
- **Wall strip:** an ordered list of beds, each a three-column row (22px numeral, status, risk); the focused bed is mounted.
- **Mobile:** nav and controls wrap; the header unsticks under 900px.

### Status (Signature)
Shape + word, always. Triangle + "Alert", circle + "Watch", bar + "OK", the word "Discharged" with no glyph. Colour is a third, redundant cue: on-wall inks on verdigris, ink variants on labels.

### Tour Rail (Signature)
A fixed deep-verdigris band on every page, like a gallery audio-guide route: "Tour" then numbered pill stops (32px numeral disc in `wall-raised` + title). The current stop is mounted: label white pill, ink numeral disc with label-white digit. On the Tour page the same stops are 14px-radius outlined items, the current one mounted.

### Risk Bar and Sparkline (Signature)
Wall-table instruments. The risk bar's full scale adapts to the threshold (max of 0.2 or 2τ) with a white tick at τ; fill is muted for OK, ochre for Watch, vermilion for Alert. The 12-hour sparkline is a 2px `wall-ink` line with a dashed vermilion threshold and an end dot that turns vermilion at or above τ.

### Risk Chart
Recharts, colours read live from the CSS tokens. Served risk is a 3.5px line in `wall` (the verdigris returns as ink on the label); LightGBM part dashed `ink-muted`; GRU part dotted `gru`; alert hours shaded `alert` at 0.18; threshold a 2px dashed `alert-ink` line labelled in 700; "now" a dotted muted line; sepsis onset a 3px ink line labelled in 800, shown only with ground truth on.

## Do's and Don'ts

### Do:
- **Do** hang every page on the verdigris wall (`#203e3a`) and mount only the item in focus on label white (`#f8f8f4`).
- **Do** render every status as glyph + word; use `alert-on-wall` / `watch-on-wall` on the wall and `alert-ink` / `watch-ink` on labels.
- **Do** pick type sizes only from 14 / 17 / 22 / 32 / clamp(56px, 9vh, 80px), with tabular figures for every live number.
- **Do** keep one catalogue number per bed everywhere, and set catalogue lines and codes in Atkinson Hyperlegible Mono.
- **Do** hang development (ensemble) evidence in the slate annex with its dashed `annex-line` edge, and lockbox evidence in the outlined room on the main wall.
- **Do** keep the disclaimer strip at the top and the tour rail at the foot of every page.
- **Do** read chart colours from the CSS tokens (`chartColors.ts`) rather than hard-coding them.

### Don't:
- **Don't** use vermilion or ochre for anything that is not Alert or Watch state.
- **Don't** show status by colour alone, and don't add a third state hue.
- **Don't** put lockbox and development numbers on the same plate, table or room.
- **Don't** build a dark neon ICU-monitor look (glows, black backgrounds, saturated traces) or a white SaaS card grid.
- **Don't** cast the Mount shadow from anything but a mounted label board.
- **Don't** mount unfocused items in label white, and don't use brass or gold for the current tour stop; it would collide with ochre Watch.
- **Don't** add letter-spaced eyebrows or kickers above headings; the only run-in key is WHAT HAPPENED.

<!-- Fonts are self-hosted from frontend/public/fonts (variable woff2, latin + latin-ext, SIL OFL 1.1 licences beside them), declared in frontend/src/fonts.css with font-display: swap; the two latin files are preloaded in index.html. No runtime font requests leave the machine. -->
