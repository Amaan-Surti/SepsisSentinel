# Slides

`slides.md` is a 14-slide deck written in [Marp](https://marp.app) Markdown. Speaker notes and layout guidance are the HTML comments under each slide (Marp shows them as presenter notes). Images are referenced relatively from `../figures/` and `../screenshots/`.

Pre-rendered exports: `slides.pptx` and `slides.pdf` (marp-cli 3.4.0).

## Render

```powershell
# from reports/slides/
npx @marp-team/marp-cli@3.4.0 slides.md --pptx --allow-local-files   # PowerPoint
npx @marp-team/marp-cli@3.4.0 slides.md --pdf  --allow-local-files   # PDF
npx @marp-team/marp-cli@3.4.0 slides.md --html --allow-local-files   # HTML with presenter view
```

Or open `slides.md` in VS Code with the "Marp for VS Code" extension and export.

## Slide map (build plan §P8.T3)

| # | Slide | Plan item |
|---|---|---|
| 1 | Title | – |
| 2 | Why sepsis, why early | motivation |
| 3 | Data: by the numbers | "By the numbers" (exact rows, septic patients, % positive rows) |
| 4 | Missing values | "Missing values" (real missingness figure) |
| 5 | Method | pipeline and leakage test |
| 6 | Evaluation | utility function and clinical metrics |
| 7 | Lockbox test | "Evaluation" (test results + CIs, replacing targets) |
| 8 | Results vs baselines | new |
| 9 | Ablation: missingness is a signal | new |
| 10 | Cross-hospital finding | new |
| 11 | Product | "One patient's stay" (real demo patient, Bed 16) |
| 12 | Live demo | new |
| 13 | Limitations and ethics | §13.4 |
| 14 | Takeaways | – |

All numbers match `reports/tables/` and `reports/report.md`. Timing: about 11 minutes of talk + 3 minutes of demo.
