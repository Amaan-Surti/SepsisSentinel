import { Level, pct } from "../api";

/** Horizontal risk bar; full scale adapts to the threshold so small calibrated risks stay readable. */
export function RiskBar({ risk, threshold, level }: { risk: number; threshold: number; level: Level }) {
  const scale = Math.max(0.2, threshold * 2);
  const w = Math.min(1, risk / scale) * 100;
  const tick = Math.min(1, threshold / scale) * 100;
  return (
    <div className="riskbar" title={`Risk ${pct(risk, 2)} · threshold ${pct(threshold, 1)}`}>
      <div className="riskbar-track" aria-hidden="true">
        <div className={`riskbar-fill fill-${level.toLowerCase()}`} style={{ width: `${w}%` }} />
        <div className="riskbar-tick" style={{ left: `${tick}%` }} />
      </div>
      <span className="riskbar-value num-tab">{pct(risk, 2)}</span>
    </div>
  );
}
