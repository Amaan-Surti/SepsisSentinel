"""Results assembler — P8.T1.

Builds reports/results.md from every table in reports/tables/ and every figure in reports/figures/,
in the order of the final report. Tables or figures not listed in SECTIONS are appended in an
appendix so nothing is silently dropped. Re-run whenever results change:

    python -m sepsis.report
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Table:
    name: str
    caption: str
    prep: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    note: str = ""


@dataclass(frozen=True)
class Figure:
    name: str
    caption: str


def _top(col: str, n: int, ascending: bool = False) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df.sort_values(col, ascending=ascending).head(n)


def _head(n: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df.head(n)


def _adv(df: pd.DataFrame) -> pd.DataFrame:
    auroc = df.loc[df["metric"] == "adversarial_auroc", "value"].iloc[0]
    feats = df.dropna(subset=["feature"])[["feature", "importance_gain"]].head(15)
    head = pd.DataFrame({"feature": ["(adversarial AUROC)"], "importance_gain": [auroc]})
    return pd.concat([head, feats], ignore_index=True)


SECTIONS: List[tuple[str, str, list]] = [
    (
        "1. Data",
        "PhysioNet/CinC Challenge 2019 training sets A and B after physiological-range cleaning.",
        [
            Table("dataset_summary.csv", "Cohort summary by hospital."),
            Table("cleaning_report.csv", "Values set to NaN by physiological sanity ranges."),
            Table(
                "missingness_by_var.csv",
                "Share of hours with no measurement, by variable and hospital (15 most missing).",
                _top("pct_missing_overall", 15),
            ),
            Table(
                "ks_tests_A_vs_B.csv",
                "Kolmogorov–Smirnov tests, hospital A vs B (10 largest statistics).",
                _top("ks_stat", 10),
            ),
            Table(
                "adversarial_validation.csv",
                "Adversarial validation: LightGBM predicting hospital from per-patient summaries.",
                _adv,
            ),
            Figure("fig_class_imbalance.png", "Septic patients and positive hours by hospital."),
            Figure("fig_stay_length.png", "ICU length of stay distribution."),
            Figure("fig_onset_timing.png", "Timing of sepsis onset within the ICU stay."),
            Figure("fig_missingness_heatmap.png", "Missingness by variable and hospital."),
            Figure("fig_measurement_frequency.png", "Measurement frequency by variable."),
            Figure("fig_trajectories_aligned.png", "Vital-sign trajectories aligned on onset."),
            Figure("fig_adversarial_top20.png", "Top-20 features separating hospital A from B."),
        ],
    ),
    (
        "2. Development (5-fold out-of-fold on the dev split)",
        "All fitting, tuning, calibration and policy choice used the 32,268 dev patients only.",
        [
            Table("baselines_dev_oof.csv", "Baselines B0–B3, dev out-of-fold."),
            Table("lgbm_dev_oof.csv", "LightGBM default vs Optuna-tuned, dev out-of-fold."),
            Table(
                "optuna_trials.csv",
                "Optuna trials (top 10 of 40 by validation AUPRC).",
                lambda df: df.sort_values("value", ascending=False)
                .head(10)
                .drop(
                    columns=[
                        c
                        for c in (
                            "datetime_start",
                            "datetime_complete",
                            "system_attrs_fixed_params",
                        )
                        if c in df.columns
                    ]
                ),
            ),
            Table("ablations.csv", "Feature-group ablations (tuned params, dev out-of-fold)."),
            Figure("fig_ablations.png", "Utility by ablation variant."),
        ],
    ),
    (
        "3. Calibration and alert policy",
        "Isotonic regression and the alert-policy grid were fitted on calibrated dev OOF scores.",
        [
            Table(
                "policy_grid.csv",
                "Alert-policy grid on calibrated dev OOF (top 10 of all configurations).",
                _top("utility_norm", 10),
            ),
            Figure("fig_calibration_dev.png", "Reliability diagram on dev OOF, before/after."),
            Figure(
                "fig_policy_tradeoff.png", "Utility, sensitivity and false alerts vs threshold."
            ),
        ],
    ),
    (
        "4. Lockbox test (8,068 patients, evaluated once)",
        "95% CIs from 1,000 bootstrap resamples of patients (seed 42).",
        [
            Table("final_test_metrics.csv", "Final model on the lockbox test set."),
            Table("final_test_baselines.csv", "Baselines on the lockbox test set."),
            Table("final_test_subgroups.csv", "Final model by subgroup on the lockbox test set."),
            Figure("fig_roc_pr_test.png", "ROC and precision–recall curves (test)."),
            Figure("fig_calibration_test.png", "Reliability diagram (test)."),
            Figure("fig_lead_time_hist_test.png", "Lead time of first alert before onset (test)."),
        ],
    ),
    (
        "5. Cross-hospital generalization",
        "R0 full features; R1 no missingness; R2 per-hospital standardization; R3 no admin timing.",
        [
            Table("cross_hospital.csv", "Internal CV vs zero-shot transfer between hospitals."),
            Figure("fig_cross_hospital.png", "Internal vs external utility by variant."),
            Figure("fig_score_distributions_A_vs_B.png", "Risk score distributions, A vs B."),
        ],
    ),
    (
        "6. Explainability and subgroups (dev)",
        "SHAP TreeExplainer on the final model; subgroup metrics on calibrated dev OOF.",
        [
            Table("shap_global.csv", "Global mean |SHAP| (top 20).", _head(20)),
            Table("shap_stability.csv", "Pairwise Jaccard of top-10 SHAP features across folds."),
            Table("subgroups_dev.csv", "Subgroup metrics on calibrated dev OOF."),
            Figure("fig_shap_bar_top20.png", "Top-20 features by mean |SHAP|."),
            Figure("fig_shap_beeswarm_top20.png", "SHAP beeswarm, top-20 features."),
        ],
    ),
]


def _fmt(x: object) -> str:
    """Format a cell: floats to 4 significant decimals, NaN as an em dash."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float):
        if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-4):
            return f"{x:.3g}"
        return f"{x:.4f}".rstrip("0").rstrip(".") if abs(x) >= 100 else f"{x:.4f}"
    return str(x).replace("|", "\\|")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _table_block(t: Table, tables_dir: Path, n: int) -> str:
    df = pd.read_csv(tables_dir / t.name)
    total = len(df)
    shown = t.prep(df) if t.prep else df
    note = f" Showing {len(shown)} of {total} rows." if len(shown) < total else ""
    return (
        f"**Table {n}.** {t.caption}{note} *Source: `tables/{t.name}`*\n\n{df_to_markdown(shown)}\n"
    )


def _figure_block(f: Figure, n: int) -> str:
    return f"![{f.caption}](figures/{f.name})\n\n**Figure {n}.** {f.caption}\n"


def build_results_markdown(reports_dir: Path) -> str:
    """Assemble the Markdown text for reports/results.md."""
    tables_dir, figures_dir = reports_dir / "tables", reports_dir / "figures"
    tables = sorted(p.name for p in tables_dir.glob("*.csv"))
    figures = sorted(p.name for p in figures_dir.glob("*.png"))
    used_t, used_f = set(), set()
    t_n = f_n = 0
    parts = [
        "# SepsisSentinel — Results\n",
        "Auto-generated by `python -m sepsis.report` from `reports/tables/` and "
        "`reports/figures/`. Do not edit by hand; re-run after any change.\n",
    ]
    for heading, intro, items in SECTIONS:
        parts.append(f"## {heading}\n\n{intro}\n")
        for item in items:
            if isinstance(item, Table) and item.name in tables:
                t_n += 1
                parts.append(_table_block(item, tables_dir, t_n))
                used_t.add(item.name)
            elif isinstance(item, Figure) and item.name in figures:
                f_n += 1
                parts.append(_figure_block(item, f_n))
                used_f.add(item.name)
            else:
                logger.warning("Missing artifact for %s", item.name)

    extra_t = [t for t in tables if t not in used_t]
    extra_f = [f for f in figures if f not in used_f]
    if extra_t or extra_f:
        parts.append("## Appendix: other artifacts\n")
        for name in extra_t:
            t_n += 1
            prep = _top("utility_norm", 5) if name.startswith("policy_grid_") else _head(20)
            parts.append(
                _table_block(Table(name, name.removesuffix(".csv") + ".", prep), tables_dir, t_n)
            )
        for name in extra_f:
            f_n += 1
            parts.append(_figure_block(Figure(name, name.removesuffix(".png") + "."), f_n))
    logger.info("Assembled %d tables and %d figures", t_n, f_n)
    return "\n".join(parts)


def main() -> None:
    """CLI entrypoint: python -m sepsis.report."""
    parser = argparse.ArgumentParser(description="Assemble reports/results.md")
    parser.add_argument("--reports", type=str, default="reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    reports_dir = Path(args.reports)
    out = reports_dir / "results.md"
    out.write_text(build_results_markdown(reports_dir), encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
