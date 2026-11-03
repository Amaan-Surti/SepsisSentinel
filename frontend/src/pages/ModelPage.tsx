import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError, DevCV, Metrics, ModelCard, pct, Row } from "../api";
import { C, TICK } from "../chartColors";
import { Markdown } from "../components/Markdown";
import { useReplay } from "../state/ReplayContext";
import { usePageTitle } from "../usePageTitle";

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

const GRU_NAME: Record<string, string> = { plain: "GRU", h64: "GRU, hidden 64", top32: "GRU + 32 LightGBM features" };
const VARIANT_NAME: Record<string, string> = { plain: "Plain GRU", h64: "GRU hidden 64", top32: "GRU + 32 LightGBM features" };
const ROW_ORDER = ["lgbm", "gru", "ensemble"];

const n = (r: Row, k: string) => Number(r[k]);
const signed = (x: number) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(4)}`;

/** "LightGBM", "GRU, hidden 64", "Ensemble with GRU hidden 64", … */
function name(model: string, variant: string): string {
  if (model === "lgbm") return "LightGBM";
  if (model === "gru") return GRU_NAME[variant] ?? `GRU (${variant})`;
  return `Ensemble with ${variant === "plain" ? "GRU" : VARIANT_NAME[variant] ?? variant}`;
}

const modelName = (r: Row) => name(String(r.model), String(r.gru_variant ?? "-"));

function DevSection({ dev }: { dev: DevCV }) {
  const sel = dev.selection;
  const nested = dev.comparison
    .filter((r) => r.scheme === "nested")
    .sort((a, b) => ROW_ORDER.indexOf(String(a.model)) - ROW_ORDER.indexOf(String(b.model)));
  const weights = dev.weights.filter((r) => r.gru_variant === sel.chosen_variant);
  const served = dev.final.find((r) => r.scheme === "nested");
  const bootLabel = (c: string) =>
    c
      .split(" - ")
      .map((side) => {
        const [m, v] = side.split("/");
        return name(m, v ?? "-");
      })
      .join(" − ");
  return (
    <section className="band band-dev" aria-labelledby="dev-title">
      <div className="band-head">
        <h2 id="dev-title">{dev.label}</h2>
        <span className="band-tag tag-dev">Development data · not lockbox</span>
      </div>
      <p className="band-note">{dev.note} These numbers are not comparable with the lockbox test above.</p>

      <div className="two-col">
        <section className="card">
          <h3>LightGBM vs GRU vs ensemble</h3>
          <p className="sub">Nested out-of-fold results on 32,268 development patients. Highlighted: the served configuration.</p>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col" className="num">Utility</th>
                  <th scope="col" className="num">AUROC</th>
                  <th scope="col" className="num">Sensitivity</th>
                  <th scope="col" className="num">False alerts/day</th>
                  <th scope="col" className="num">Lead time</th>
                </tr>
              </thead>
              <tbody>
                {nested.map((r) => {
                  const isServed = r.model === "ensemble" && r.gru_variant === sel.chosen_variant;
                  return (
                    <tr key={`${r.model}-${r.gru_variant}`} className={isServed ? "highlight" : ""}>
                      <td>
                        {modelName(r)}
                        {isServed && <span className="served-tag">served</span>}
                      </td>
                      <td className="num num-tab">{n(r, "utility_norm").toFixed(4)}</td>
                      <td className="num num-tab">{n(r, "auroc").toFixed(4)}</td>
                      <td className="num num-tab">{pct(n(r, "patient_sensitivity"), 2)}</td>
                      <td className="num num-tab">{n(r, "false_alerts_per_patient_day").toFixed(3)}</td>
                      <td className="num num-tab">{n(r, "median_lead_time_h").toFixed(1)} h</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <h3>Blend weight search</h3>
          <p className="sub">
            LightGBM share w ∈ {"{"}
            {sel.weights.join(", ")}
            {"}"}, fixed before the run. Each fold's weight was picked on the other four folds.
          </p>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col" className="num">w (LightGBM)</th>
                  <th scope="col" className="num">Utility</th>
                  <th scope="col" className="num">Sensitivity</th>
                  <th scope="col" className="num">False alerts/day</th>
                  <th scope="col" className="num">Lead time</th>
                  <th scope="col" className="num">Folds picking it</th>
                </tr>
              </thead>
              <tbody>
                {weights.map((r) => (
                  <tr key={String(r.weight_lgbm)} className={n(r, "weight_lgbm") === sel.pooled_weight ? "highlight" : ""}>
                    <td className="num num-tab">{n(r, "weight_lgbm").toFixed(1)}</td>
                    <td className="num num-tab">{n(r, "nested_utility_norm").toFixed(4)}</td>
                    <td className="num num-tab">{pct(n(r, "nested_patient_sensitivity"), 2)}</td>
                    <td className="num num-tab">{n(r, "nested_false_alerts_per_patient_day").toFixed(3)}</td>
                    <td className="num num-tab">{n(r, "nested_median_lead_time_h").toFixed(1)} h</td>
                    <td className="num num-tab">{n(r, "folds_selecting")} of 5</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <div className="two-col">
        <section className="card">
          <h3>Paired patient bootstrap</h3>
          <p className="sub">Difference in nested utility, 1,000 patient resamples, 95% CI.</p>
          <ul className="boot">
            {dev.bootstrap.map((b) => {
              const lo = n(b, "ci_low");
              const verdict = lo > 0 ? "CI above 0" : n(b, "ci_high") < 0 ? "CI below 0" : "CI includes 0";
              return (
                <li key={String(b.comparison)}>
                  <span className="boot-label">{bootLabel(String(b.comparison))}</span>
                  <span className="boot-value num-tab">
                    {signed(n(b, "diff"))} <span className="unit">[{signed(lo)}, {signed(n(b, "ci_high"))}]</span>
                  </span>
                  <span className="boot-verdict">
                    {verdict} · better in {pct(n(b, "prob_better"), 1)} of resamples
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
        <section className="card">
          <h3>Served configuration</h3>
          <dl className="config-list">
            <div>
              <dt>GRU variant kept</dt>
              <dd>{VARIANT_NAME[sel.chosen_variant] ?? sel.chosen_variant}</dd>
            </div>
            <div>
              <dt>Blend</dt>
              <dd className="num-tab">
                {sel.pooled_weight.toFixed(1)} × LightGBM + {(1 - sel.pooled_weight).toFixed(1)} × GRU
              </dd>
            </div>
            <div>
              <dt>Alert policy</dt>
              <dd className="num-tab">
                τ = {sel.pooled_policy.threshold}, k = {sel.pooled_policy.k}, sticky = {sel.pooled_policy.sticky}
              </dd>
            </div>
            {served && (
              <div>
                <dt>GRU epochs (all dev patients)</dt>
                <dd className="num-tab">{n(served, "gru_epochs")}</dd>
              </div>
            )}
          </dl>
          <p className="sub why">{sel.reason}.</p>
          <p className="sub why">The ensemble has not been evaluated on the lockbox test set.</p>
        </section>
      </div>
    </section>
  );
}

export function ModelPage() {
  const { threshold, config } = useReplay();
  const [m, setM] = useState<Metrics | null>(null);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  usePageTitle("Model");

  useEffect(() => {
    const ctl = new AbortController();
    Promise.all([api.metrics(ctl.signal), api.modelCard(ctl.signal)])
      .then(([mm, cc]) => {
        setM(mm);
        setCard(cc);
        setError(null);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => ctl.abort();
  }, [attempt]);

  if (error)
    return (
      <div className="state error" role="alert">
        {error} <button onClick={() => setAttempt((a) => a + 1)}>Retry</button>
      </div>
    );
  if (!m || !card) return <div className="state" role="status">Loading model results…</div>;

  const lb = m.lockbox_test;
  const test = Object.fromEntries(lb.metrics.map((r) => [String(r.metric), r]));
  const v = (k: string) => METRIC_LABEL[k][1](n(test[k], "value"));
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
  };
  const r0 = m.cross_hospital.filter((r) => String(r.variant).startsWith("R0"));
  const rOther = m.cross_hospital.filter((r) => !String(r.variant).startsWith("R0"));
  const crossRows = (rows: Row[]) =>
    rows.map((r) => (
      <tr key={`${r.variant}-${r.direction}`} className={String(r.direction).includes("->") ? "external" : ""}>
        <td>{String(r.variant)}</td>
        <td>{String(r.direction).replace("->", " → ")}</td>
        <td className="num num-tab">{n(r, "utility_norm").toFixed(4)}</td>
        <td className="num num-tab">{n(r, "auroc").toFixed(3)}</td>
        <td className="num num-tab">{pct(n(r, "patient_sensitivity"), 1)}</td>
        <td className="num num-tab">{n(r, "false_alerts_per_patient_day").toFixed(2)}</td>
      </tr>
    ));
  const crossHead = (
    <thead>
      <tr>
        <th scope="col">Variant</th>
        <th scope="col">Train → test</th>
        <th scope="col" className="num">Utility</th>
        <th scope="col" className="num">AUROC</th>
        <th scope="col" className="num">Sensitivity</th>
        <th scope="col" className="num">False alerts/day</th>
      </tr>
    </thead>
  );

  return (
    <main className="page model">
      <div className="page-head">
        <div>
          <h1>Model performance</h1>
          <p className="sub">
            Two kinds of result, kept apart: the one-time lockbox test of the LightGBM model, and the development comparison that chose
            the served model.
          </p>
        </div>
      </div>
      {config?.model_kind === "ensemble" && (
        <p className="served-note" role="note">
          The app serves the <b>LightGBM + GRU ensemble</b>, chosen on development data. Its lockbox score has not been measured; the
          lockbox result below belongs to the LightGBM component alone.
        </p>
      )}

      <section className="band band-lockbox" aria-labelledby="lockbox-title">
        <div className="band-head">
          <h2 id="lockbox-title">{lb.label}</h2>
          <span className="band-tag tag-lockbox">8,068 held-out patients · evaluated once</span>
        </div>
        <p className="headline">
          On the lockbox, LightGBM caught <b>{v("patient_sensitivity")}</b> of septic patients, a median <b>{v("median_lead_time_h")}</b>{" "}
          before onset, at <b>{v("false_alerts_per_patient_day")}</b> false alerts per patient-day. Normalized utility{" "}
          <b>{v("utility_norm")}</b> (95% CI {METRIC_LABEL.utility_norm[1](n(test.utility_norm, "ci_lower"))}–
          {METRIC_LABEL.utility_norm[1](n(test.utility_norm, "ci_upper"))}).
        </p>
        <p className="band-note">
          {lb.model}. 95% CIs from 1,000 patient-level bootstrap resamples. Policy τ = {lb.policy.threshold}, k = {lb.policy.k}, sticky ={" "}
          {lb.policy.sticky}.
        </p>

        <div className="two-col">
          <section className="card">
            <h3>Test metrics</h3>
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  <th scope="col" className="num">Value</th>
                  <th scope="col" className="num">95% CI</th>
                </tr>
              </thead>
              <tbody>
                {lb.metrics.map((r) => {
                  const [label, f] = METRIC_LABEL[String(r.metric)] ?? [String(r.metric), (x: number) => x.toFixed(4)];
                  return (
                    <tr key={String(r.metric)}>
                      <td>{label}</td>
                      <td className="num num-tab">{f(n(r, "value"))}</td>
                      <td className="num num-tab">{r.ci_lower == null ? "—" : `${f(n(r, "ci_lower"))} – ${f(n(r, "ci_upper"))}`}</td>
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
                  <th scope="col">Model</th>
                  <th scope="col" className="num">Utility</th>
                  <th scope="col" className="num">AUROC</th>
                  <th scope="col" className="num">Sensitivity</th>
                  <th scope="col" className="num">False alerts/day</th>
                </tr>
              </thead>
              <tbody>
                {[...lb.baselines, ours].map((r) => (
                  <tr key={String(r.baseline)} className={r.baseline === "SepsisSentinel" ? "highlight" : ""}>
                    <td>{BASELINE_LABEL[String(r.baseline)] ?? "SepsisSentinel · LightGBM"}</td>
                    <td className="num num-tab">{n(r, "utility_norm").toFixed(4)}</td>
                    <td className="num num-tab">{n(r, "auroc").toFixed(3)}</td>
                    <td className="num num-tab">{pct(n(r, "patient_sensitivity"), 2)}</td>
                    <td className="num num-tab">{n(r, "false_alerts_per_patient_day").toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>

        <div className="two-col">
          <section className="card">
            <h3>Calibration (test)</h3>
            <img
              className="fig"
              src={api.figureUrl("fig_calibration_test.png")}
              alt={`Reliability diagram on the test set; ECE ${v("ece")}`}
              loading="lazy"
              width={2100}
              height={2400}
            />
          </section>
          <section className="card">
            <h3>ROC and precision–recall (test)</h3>
            <img className="fig" src={api.figureUrl("fig_roc_pr_test.png")} alt="ROC and PR curves on the test set" loading="lazy" width={3600} height={1500} />
          </section>
        </div>
      </section>

      {m.dev_cv && <DevSection dev={m.dev_cv} />}

      <section className="card">
        <h2>Threshold trade-off · LightGBM, development out-of-fold</h2>
        <p className="sub">
          Lower thresholds catch more septic patients earlier but raise false alerts. The stored curve spans τ = {curveMin.toFixed(2)}–0.95
          (k = {lb.policy.k}, sticky = {lb.policy.sticky}); the dotted line marks the LightGBM policy τ = {lb.policy.threshold}
          {lb.policy.utility_norm != null && ` (dev utility ${lb.policy.utility_norm.toFixed(4)})`}.
          {threshold >= curveMin && ` The amber line is the slider (${pct(threshold, 1)}).`}
        </p>
        <figure className="chart" role="img" aria-label="Normalized utility, patient sensitivity and false alerts per patient-day against the alert threshold">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={curve} margin={{ top: 16, right: 24, bottom: 18, left: 4 }}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis
                dataKey="threshold"
                type="number"
                domain={[0, 0.95]}
                ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
                stroke={C.muted}
                fontSize={TICK.main}
                label={{ value: "Alert threshold τ", position: "insideBottomRight", offset: -12, fill: C.muted, fontSize: TICK.main }}
              />
              <YAxis yAxisId="l" domain={[0, 1]} stroke={C.muted} fontSize={TICK.main} width={44} />
              <YAxis yAxisId="r" orientation="right" stroke={C.alertInk} fontSize={TICK.main} width={44} />
              <Tooltip formatter={(x: number, k: string) => [x.toFixed(3), k]} labelFormatter={(x) => `Threshold ${x}`} />
              <Legend wrapperStyle={{ fontSize: TICK.main, color: C.text }} />
              <ReferenceLine yAxisId="l" x={lb.policy.threshold} stroke={C.text} strokeDasharray="2 3" strokeWidth={1.5} />
              {threshold >= curveMin && <ReferenceLine yAxisId="l" x={threshold} stroke={C.watch} strokeWidth={2.5} />}
              <Line yAxisId="l" dataKey="utility" name="Normalized utility" stroke={C.teal} strokeWidth={3} dot={false} isAnimationActive={false} />
              <Line yAxisId="l" dataKey="sensitivity" name="Patient sensitivity" stroke={C.text} strokeWidth={2} strokeDasharray="8 4" dot={false} isAnimationActive={false} />
              <Line yAxisId="r" dataKey="fa" name="False alerts / patient-day (right axis)" stroke={C.alertInk} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </figure>
      </section>

      <section className="card">
        <h2>Cross-hospital generalization · LightGBM, development data</h2>
        <p className="sub">Train on one hospital, test on the other. R0 uses the full feature set; shaded rows are the transfer (external) direction.</p>
        <div className="table-scroll">
          <table className="data">
            {crossHead}
            <tbody>{crossRows(r0)}</tbody>
          </table>
        </div>
        {rOther.length > 0 && (
          <details className="disclosure">
            <summary>Robustness variants R1–R3 (no missingness, per-hospital standardization, no admission timing)</summary>
            <div className="table-scroll">
              <table className="data">
                {crossHead}
                <tbody>{crossRows(rOther)}</tbody>
              </table>
            </div>
          </details>
        )}
      </section>

      <section className="card model-card">
        <details className="disclosure">
          <summary>
            <h2>Model card</h2>
            <span className="unit">{card.model_version}</span>
          </summary>
          <Markdown source={card.markdown} />
        </details>
      </section>
    </main>
  );
}
