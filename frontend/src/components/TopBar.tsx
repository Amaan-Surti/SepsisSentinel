import { NavLink, useLocation } from "react-router-dom";
import { pct } from "../api";
import { SPEEDS, useReplay } from "../state/ReplayContext";
import { BackIcon, ForwardIcon, PauseIcon, PlayIcon, ResetIcon } from "./Icons";

const BADGE: Record<string, string> = { ensemble: "Ensemble", lgbm: "LightGBM" };

export function TopBar() {
  const r = useReplay();
  const { pathname } = useLocation();
  const max = r.config?.max_hour ?? 0;
  const offPolicy = r.config != null && Math.abs(r.threshold - r.config.threshold) > 1e-9;
  return (
    <header className="top">
      <div className="brand">
        <span className="brand-name">SepsisSentinel</span>
        {r.config && (
          <span className="badge" title={`Served model: ${r.config.model_label} (${r.config.model_version})`}>
            <span className="vh">Served model: </span>
            {BADGE[r.config.model_kind] ?? r.config.model_label}
          </span>
        )}
        <nav className="nav" aria-label="Pages">
          <NavLink to="/" className={({ isActive }) => (isActive || pathname.startsWith("/bed/") ? "active" : "")} end>
            Ward
          </NavLink>
          <NavLink to="/evidence">Evidence</NavLink>
        </nav>
      </div>

      <div className="controls" role="group" aria-label="Replay controls">
        <div className="clock">
          <span className="clock-label">ICU hour</span>
          <span className="clock-value num">{r.t}</span>
          <span className="clock-max">/ {max}</span>
        </div>
        <div className="btn-group">
          <button onClick={r.reset} title="Back to hour 0">
            <ResetIcon />
            Reset
          </button>
          <button onClick={() => r.step(-1)} disabled={r.t <= 0} title="Back 1 hour (← key)" aria-label="Back 1 hour">
            <BackIcon />1 h
          </button>
          <button className="primary play" onClick={r.togglePlay} disabled={!r.config} title="Play or pause (Space)">
            {r.playing ? <PauseIcon /> : <PlayIcon />}
            {r.playing ? "Pause" : "Play"}
          </button>
          <button onClick={() => r.step(1)} disabled={r.t >= max} title="Forward 1 hour (→ key)" aria-label="Forward 1 hour">
            1 h<ForwardIcon />
          </button>
        </div>
        <label className="field">
          <span>Speed</span>
          <select value={r.speedMs} onChange={(e) => r.setSpeedMs(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s.ms} value={s.ms}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <div className="field">
          <span className="threshold-head">
            <label htmlFor="threshold">Alert threshold</label>
            <b className="num">{pct(r.threshold, 1)}</b>
            {offPolicy && (
              <button className="link" onClick={() => r.setThreshold(r.config!.threshold)} title="Back to the policy threshold">
                reset to policy {pct(r.config!.threshold, 1)}
              </button>
            )}
          </span>
          <input
            id="threshold"
            type="range"
            min={0.01}
            max={0.5}
            step={0.005}
            value={r.threshold}
            onChange={(e) => r.setThreshold(Number(e.target.value))}
            aria-valuetext={pct(r.threshold, 1)}
          />
        </div>
        <label className="toggle">
          <input type="checkbox" checked={r.showTruth} onChange={(e) => r.setShowTruth(e.target.checked)} />
          <span>Show ground truth (demo)</span>
        </label>
      </div>
    </header>
  );
}
