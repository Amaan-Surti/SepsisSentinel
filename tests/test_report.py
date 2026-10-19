"""Tests for the results assembler (P8.T1)."""

import pandas as pd

from sepsis.report import build_results_markdown, df_to_markdown


def test_df_to_markdown_formats_cells():
    df = pd.DataFrame({"metric": ["auroc", "a|b"], "value": [0.84750671, float("nan")]})
    md = df_to_markdown(df).splitlines()
    assert md[0] == "| metric | value |"
    assert md[2] == "| auroc | 0.8475 |"
    assert md[3] == "| a\\|b | — |"


def test_build_includes_listed_and_unlisted_artifacts(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "figures").mkdir()
    pd.DataFrame({"metric": ["utility_norm"], "value": [0.427]}).to_csv(
        tmp_path / "tables" / "final_test_metrics.csv", index=False
    )
    pd.DataFrame({"x": [1, 2]}).to_csv(tmp_path / "tables" / "something_new.csv", index=False)
    (tmp_path / "figures" / "fig_roc_pr_test.png").write_bytes(b"png")
    (tmp_path / "figures" / "fig_extra.png").write_bytes(b"png")
    md = build_results_markdown(tmp_path)
    assert "## 4. Lockbox test" in md
    assert "| utility_norm | 0.4270 |" in md
    assert "figures/fig_roc_pr_test.png" in md
    assert "## Appendix: other artifacts" in md
    assert "something_new" in md and "figures/fig_extra.png" in md
