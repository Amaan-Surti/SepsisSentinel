import { ModelKind, Reason } from "../api";
import { RaiseIcon } from "./Icons";

const SOURCE_NOTE: Record<ModelKind, string> = {
  ensemble: "Reasons come from the LightGBM part of the ensemble; the GRU part has no per-feature explanation.",
  lgbm: "Reasons come from the LightGBM model.",
};

export function ReasonsPanel({ reasons, hour, kind }: { reasons: Reason[]; hour: number; kind: ModelKind }) {
  const max = Math.max(...reasons.map((r) => r.shap), 1e-9);
  return (
    <section className="card reasons" aria-labelledby="reasons-title">
      <h2 id="reasons-title">Why this risk</h2>
      <p className="sub">Factors raising the risk most at hour {hour}</p>
      <p className="source-note">{SOURCE_NOTE[kind]}</p>
      {reasons.length === 0 ? (
        <p className="empty">No factor is pushing risk up at this hour.</p>
      ) : (
        <ol>
          {reasons.map((r) => (
            <li key={r.feature}>
              <RaiseIcon className="raise" size={16} />
              <div className="reason-body">
                <span className="reason-text">
                  {r.text}
                  <span className="visually-hidden"> (raises risk)</span>
                </span>
                <span className="reason-bar" aria-hidden="true">
                  <span style={{ width: `${(r.shap / max) * 100}%` }} />
                </span>
                <code className="reason-feat">
                  {r.feature} · SHAP +{r.shap.toFixed(3)}
                </code>
              </div>
            </li>
          ))}
        </ol>
      )}
      <p className="caveat">SHAP explains the model's behaviour, not the patient's physiology. Timing features reflect when clinicians ordered tests.</p>
    </section>
  );
}
