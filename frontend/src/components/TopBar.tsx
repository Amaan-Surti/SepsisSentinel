import { NavLink } from "react-router-dom";
import { pct } from "../api";
import { SPEEDS, useReplay } from "../state/ReplayContext";

export function TopBar() {
  const r = useReplay();
  const max = r.config?.max_hour ?? 0;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true" />
        <span className="brand-name">SepsisSentinel</span>
        <nav className="nav">
          <NavLink to="/" end>
            Ward
          </NavLink>
          <NavLink to="/model">Model</NavLink>
        </nav>
      </div>

      <div className="controls" aria-label="Replay controls">
        <div className="clock">
          <span className="clock-label">ICU hour</span>
          <span className="clock-value mono">
            {r.t}
            <span className="clock-max"> / {max}</span>
          </span>
        </div>
        <div className="btn-group">
          <button onClick={r.reset} title="Reset to hour 0">
            ⏮ Reset
          </button>
          <button onClick={() => r.step(-1)} disabled={r.t <= 0} title="Step back 1 hour">
            −1h
          </button>
          <button className="primary play" onClick={r.togglePlay} disabled={!r.config} aria-pressed={r.playing}>
            {r.playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          <button onClick={() => r.step(1)} disabled={r.t >= max} title="Step forward 1 hour">
            +1h
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
        <label className="field threshold">
          <span>
            Threshold <b className="mono">{pct(r.threshold, 1)}</b>
            {r.config && Math.abs(r.threshold - r.config.threshold) > 1e-9 && (
              <button className="link" onClick={() => r.setThreshold(r.config!.threshold)} title="Back to the policy threshold">
                policy {pct(r.config.threshold, 1)}
              </button>
            )}
          </span>
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.005}
            value={r.threshold}
            onChange={(e) => r.setThreshold(Number(e.target.value))}
            aria-label="Alert threshold"
          />
        </label>
        <label className="toggle">
          <input type="checkbox" checked={r.showTruth} onChange={(e) => r.setShowTruth(e.target.checked)} />
          <span>Show ground truth (demo)</span>
        </label>
      </div>
    </header>
  );
}
