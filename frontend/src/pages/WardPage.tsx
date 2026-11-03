import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError, Level, pct, sexName, unitName, WardBed } from "../api";
import { StatusGlyph } from "../components/Icons";
import { LevelChip } from "../components/LevelChip";
import { RiskBar } from "../components/RiskBar";
import { Sparkline } from "../components/Sparkline";
import { useReplay } from "../state/ReplayContext";
import { usePageTitle } from "../usePageTitle";

const fmt = (x: number | null | undefined, d = 0) => (x == null ? "—" : x.toFixed(d));

const SUMMARY: { level: Level; label: string; rule: (tau: string, half: string) => string }[] = [
  { level: "ALERT", label: "Alert", rule: (tau) => `risk ≥ ${tau}` },
  { level: "WATCH", label: "Watch", rule: (_, half) => `risk ≥ ${half}` },
  { level: "OK", label: "OK", rule: (_, half) => `below ${half}` },
];

export function WardPage() {
  const { t, appliedThreshold, config, retryConfig } = useReplay();
  const [beds, setBeds] = useState<WardBed[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const navigate = useNavigate();
  usePageTitle("Ward");

  useEffect(() => {
    if (!config) return;
    const ctl = new AbortController();
    api
      .ward(t, appliedThreshold, ctl.signal)
      .then((b) => {
        setBeds(b);
        setError(null);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => ctl.abort();
  }, [t, appliedThreshold, config, attempt]);

  if (error)
    return (
      <div className="state error" role="alert">
        {error} <button onClick={() => (config ? setAttempt((a) => a + 1) : retryConfig())}>Retry</button>
      </div>
    );
  if (!beds) return <div className="state" role="status">Loading ward…</div>;

  const active = beds.filter((b) => b.status === "active");
  const count = (l: Level) => active.filter((b) => b.level === l).length;
  const tau = pct(appliedThreshold, 1);
  const half = pct(appliedThreshold / 2, 2);

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>ICU ward</h1>
          <p className="sub">
            Hour {t} of the replay · {active.length} of {beds.length} beds occupied · sorted by calibrated risk
          </p>
        </div>
        <ul className="summary" aria-label="Ward summary">
          {SUMMARY.map((s) => (
            <li key={s.level} className={`summary-item summary-${s.level.toLowerCase()}`}>
              <StatusGlyph level={s.level} size={14} />
              <b className="num-tab">{count(s.level)}</b>
              <span className="summary-text">
                <span className="summary-label">{s.label}</span>
                <span className="summary-rule">{s.rule(tau, half)}</span>
              </span>
            </li>
          ))}
          <li className="summary-item summary-discharged">
            <b className="num-tab">{beds.length - active.length}</b>
            <span className="summary-text">
              <span className="summary-label">Discharged</span>
              <span className="summary-rule">stay ended</span>
            </span>
          </li>
        </ul>
      </div>

      <div className="card table-card">
        <div className="table-scroll">
          <table className="ward">
            <caption className="visually-hidden">
              Demo ICU beds at replay hour {t}, highest calibrated risk first. Select a bed for its trajectory.
            </caption>
            <thead>
              <tr>
                <th scope="col">Bed</th>
                <th scope="col">Age / sex</th>
                <th scope="col">Unit</th>
                <th scope="col">Risk</th>
                <th scope="col">Last 12 h</th>
                <th scope="col" className="num">
                  HR <span className="unit">bpm</span>
                </th>
                <th scope="col" className="num">
                  MAP <span className="unit">mmHg</span>
                </th>
                <th scope="col" className="num">
                  Temp <span className="unit">°C</span>
                </th>
                <th scope="col" className="num">
                  WBC <span className="unit">×10³/µL</span>
                </th>
                <th scope="col">Status</th>
                <th scope="col" className="num">
                  In alert
                </th>
              </tr>
            </thead>
            <tbody>
              {beds.map((b) => {
                const off = b.status === "discharged";
                const href = `/patient/${b.patient_id}`;
                return (
                  <tr
                    key={b.patient_id}
                    className={`${off ? "discharged" : ""} level-${off ? "off" : b.level.toLowerCase()}`}
                    onClick={() => navigate(href)}
                  >
                    <td className="bed">
                      <Link to={href}>{b.display_name}</Link>
                    </td>
                    <td>
                      {b.age} {sexName(b.gender)}
                    </td>
                    <td>{unitName(b.unit)}</td>
                    <td>
                      <RiskBar risk={b.risk} threshold={appliedThreshold} level={off ? "OK" : b.level} />
                    </td>
                    <td>
                      <Sparkline values={b.risk_trend} threshold={appliedThreshold} muted={off} />
                    </td>
                    <td className="num num-tab">{fmt(b.vitals.HR)}</td>
                    <td className="num num-tab">{fmt(b.vitals.MAP)}</td>
                    <td className="num num-tab">{fmt(b.vitals.Temp, 1)}</td>
                    <td className="num num-tab">{fmt(b.vitals.WBC, 1)}</td>
                    <td>{off ? <span className="disch-label">Discharged after hour {b.hour}</span> : <LevelChip level={b.level} />}</td>
                    <td className="num num-tab">{!off && b.alert && b.alert_since != null ? `${b.hour - b.alert_since + 1} h` : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      <p className="legend">
        Vitals are the last recorded value (labs such as WBC stay “—” until first measured). Watch = risk at least half the alert
        threshold. Keys: Space plays or pauses, ← and → step one hour.
      </p>
    </main>
  );
}
