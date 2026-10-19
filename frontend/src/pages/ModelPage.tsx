import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError, Metrics, ModelCard, pct, Row } from "../api";
import { Markdown } from "../components/Markdown";
import { C } from "../components/RiskChart";
import { useReplay } from "../state/ReplayContext";

const METRIC_LABEL: Record<string, [string, (x: number) => string]> = {
  utility_norm: ["Normalized utility (primary)", (x) => x.toFixed(4)],
  auroc: ["AUROC", (x) => x.toFixed(4)],
  auprc: ["AUPRC (chance = 1.8%)", (x) => x.toFixed(4)],
  brier: ["Brier score", (x) => x.toFixed(4)],
  ece: ["Expected calibration error", (x) => x.toFixed(4)],
  patient_sensitivity: ["Patient sensitivity", (x) => pct(x, 2)],
  median_lead_time_h: ["Median lead time before onset", (x) => `${x.toFixed(1)} h`],
  false_alerts_per_patient_day: ["False alerts per patient-day", (x) => x.toFixed(3)],
  event_ppv: ["Alert-event PPV", (x) => pct(x, 2)],
  alert_hour_rate: ["Share of hours in alert", (x) => pct(x, 1)],
};

const BASELINE_LABEL: Record<string, string> = {
  B0_never_alert: "B0 · Never alert",
  B1_sirs_rule: "B1 · SIRS ≥ 2",
  B2_qsofa_rule: "B2 · qSOFA ≥ 2 (partial)",
  B3_logistic_regression: "B3 · Logistic regression",
};

const n = (r: Row, k: string) => Number(r[k]);

export function ModelPage() {
  const { threshold } = useReplay();
  const [m, setM] = useState<Metrics | null>(null);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctl = new AbortController();
    Promise.all([api.metrics(ctl.signal), api.modelCard(ctl.signal)])
      .then(([mm, cc]) => {
        setM(mm);
        setCard(cc);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => ctl.abort();
  }, []);

  if (error) return <div className="state error">{error}</div>;
  if (!m || !card) return <div className="state">Loading model results…</div>;

  const test = Object.fromEntries(m.test_metrics.map((r) => [String(r.metric), r]));
  const curve = m.threshold_curve.map((c) => ({
    threshold: c.threshold,
    utility: c.utility_norm,
    sensitivity: c.patient_sensitivity,
    fa: c.false_alerts_per_patient_day,
  }));
  const curveMin = curve[0]?.threshold ?? 0;
  const ours: Row = {
    baseline: "SepsisSentinel",
    utility_norm: test.utility_norm.value,
    auroc: test.auroc.value,
    patient_sensitivity: test.patient_sensitivity.value,
    false_alerts_per_patient_day: test.false_alerts_per_patient_day.value,
    median_lead_time_h: test.median_lead_time_h.value,
  };

  return (
    <main className="page model">
      <div className="page-head">
        <div>
          <h1>Model performance</h1>
          <p className="sub">
            Lockbox test set: 8,068 patients, evaluated once. 95% CIs from 1,000 patient-level bootstrap resamples. Policy: τ = {m.policy.threshold},
            k = {m.policy.k}, sticky = {m.policy.sticky}.
          </p>
        </div>
      </div>

      <div className="kpis">
        {(["utility_norm", "auroc", "patient_sensitivity", "median_lead_time_h", "false_alerts_per_patient_day"] as const).map((k) => (
          <div className="kpi card" key={k}>
            <span className="kpi-label">{METRIC_LABEL[k][0]}</span>
            <span className="kpi-value mono">{METRIC_LABEL[k][1](n(test[k], "value"))}</span>
            <span className="kpi-ci mono">
              95% CI {METRIC_LABEL[k][1](n(test[k], "ci_lower"))} – {METRIC_LABEL[k][1](n(test[k], "ci_upper"))}
            </span>
          </div>
        ))}
      </div>

      <div className="two-col">
        <section className="card">
          <h3>Test metrics</h3>
          <table className="data">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Value</th>
                <th className="num">95% CI</th>
              </tr>
            </thead>
            <tbody>
              {m.test_metrics.map((r) => {
                const [label, f] = METRIC_LABEL[String(r.metric)] ?? [String(r.metric), (x: number) => x.toFixed(4)];
                return (
                  <tr key={String(r.metric)}>
                    <td>{label}</td>
                    <td className="num mono">{f(n(r, "value"))}</td>
                    <td className="num mono">{r.ci_lower == null ? "—" : `${f(n(r, "ci_lower"))} – ${f(n(r, "ci_upper"))}`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>

        <section className="card">
          <h3>Versus baselines (same test set)</h3>
          <table className="data">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Utility</th>
                <th className="num">AUROC</th>
                <th className="num">Sensitivity</th>
                <th className="num">False alerts/day</th>
              </tr>
            </thead>
            <tbody>
              {[...m.baselines, ours].map((r) => (
                <tr key={String(r.baseline)} className={r.baseline === "SepsisSentinel" ? "highlight" : ""}>
                  <td>{BASELINE_LABEL[String(r.baseline)] ?? "SepsisSentinel · LightGBM"}</td>
                  <td className="num mono">{n(r, "utility_norm").toFixed(4)}</td>
                  <td className="num mono">{n(r, "auroc").toFixed(3)}</td>
                  <td className="num mono">{pct(n(r, "patient_sensitivity"), 2)}</td>
                  <td className="num mono">{n(r, "false_alerts_per_patient_day").toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="card">
        <h3>Threshold trade-off (dev out-of-fold, k = {m.policy.k}, sticky = {m.policy.sticky})</h3>
        <p className="sub">
          Lower thresholds catch more septic patients earlier but raise false alerts. The stored curve covers τ = {curveMin.toFixed(2)}–0.95; the chosen
          policy τ = {m.policy.threshold} (dev utility {m.policy.utility_norm.toFixed(4)}) sits just left of it.
          {threshold < curveMin ? ` Slider (${pct(threshold, 1)}) is below the curve range.` : ` Marker: slider at ${pct(threshold, 1)}.`}
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={curve} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="threshold" type="number" domain={[0, 0.95]} tickCount={11} stroke={C.muted} fontSize={12} />
            <YAxis yAxisId="l" domain={[0, 1]} stroke={C.muted} fontSize={12} width={40} />
            <YAxis yAxisId="r" orientation="right" stroke={C.alert} fontSize={12} width={40} />
            <Tooltip formatter={(v: number, k: string) => [v.toFixed(3), k]} labelFormatter={(x) => `Threshold ${x}`} />
            <Legend />
            <ReferenceLine yAxisId="l" x={m.policy.threshold} stroke={C.text} strokeDasharray="2 3" label={{ value: "policy", position: "top", fontSize: 11, fill: C.text }} />
            {threshold >= curveMin && <ReferenceLine yAxisId="l" x={threshold} stroke={C.watch} strokeWidth={2} />}
            <Line yAxisId="l" dataKey="utility" name="Normalized utility" stroke={C.teal} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line yAxisId="l" dataKey="sensitivity" name="Patient sensitivity" stroke={C.text} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="r" dataKey="fa" name="False alerts / patient-day (right)" stroke={C.alert} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <div className="two-col">
        <section className="card">
          <h3>Calibration (test)</h3>
          <img className="fig" src={api.figureUrl("fig_calibration_test.png")} alt="Reliability diagram on the test set; ECE 0.0027" />
        </section>
        <section className="card">
          <h3>ROC and precision–recall (test)</h3>
          <img className="fig" src={api.figureUrl("fig_roc_pr_test.png")} alt="ROC and PR curves on the test set" />
        </section>
      </div>

      <section className="card">
        <h3>Cross-hospital generalization</h3>
        <p className="sub">Train on one hospital, test on the other. R0 = full features; R1 = no missingness; R2 = per-hospital standardization; R3 = no admission timing.</p>
        <table className="data">
          <thead>
            <tr>
              <th>Variant</th>
              <th>Direction</th>
              <th className="num">Utility</th>
              <th className="num">AUROC</th>
              <th className="num">Sensitivity</th>
              <th className="num">False alerts/day</th>
            </tr>
          </thead>
          <tbody>
            {m.cross_hospital.map((r) => (
              <tr key={`${r.variant}-${r.direction}`} className={String(r.direction).includes("->") ? "external" : ""}>
                <td>{String(r.variant)}</td>
                <td>{String(r.direction).replace("->", "→")}</td>
                <td className="num mono">{n(r, "utility_norm").toFixed(4)}</td>
                <td className="num mono">{n(r, "auroc").toFixed(3)}</td>
                <td className="num mono">{pct(n(r, "patient_sensitivity"), 1)}</td>
                <td className="num mono">{n(r, "false_alerts_per_patient_day").toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card model-card">
        <h3>Model card · {card.model_version}</h3>
        <Markdown source={card.markdown} />
      </section>
    </main>
  );
}
