import { Reason } from "../api";

export function ReasonsPanel({ reasons, hour }: { reasons: Reason[]; hour: number }) {
  const max = Math.max(...reasons.map((r) => r.shap), 1e-9);
  return (
    <section className="card reasons">
      <h3>Why this risk</h3>
      <p className="sub">Top factors raising the model's risk at hour {hour} (SHAP, log-odds)</p>
      {reasons.length === 0 ? (
        <p className="empty">No factor is pushing risk up at this hour.</p>
      ) : (
        <ol>
          {reasons.map((r) => (
            <li key={r.feature}>
              <span className="glyph up" aria-label="raises risk">
                ▲
              </span>
              <div className="reason-body">
                <span className="reason-text">{r.text}</span>
                <span className="reason-bar">
                  <span style={{ width: `${(r.shap / max) * 100}%` }} />
                </span>
                <code className="reason-feat">
                  {r.feature} · +{r.shap.toFixed(3)}
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
