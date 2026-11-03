import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError, Level, pct, sexName, unitName, WardBed } from "../api";
import { StatusGlyph } from "../components/Icons";
import { RiskBar } from "../components/RiskBar";
import { Sparkline } from "../components/Sparkline";
import { bedNumber, Status } from "../components/Status";
import { useReplay } from "../state/ReplayContext";
import { usePageTitle } from "../usePageTitle";

const fmt = (x: number | null | undefined, d = 0) => (x == null ? "—" : x.toFixed(d));

/** Ward beds at hour t, shared by the wall and the exhibit strip. */
export function useWard() {
  const { t, appliedThreshold, config } = useReplay();
  const [beds, setBeds] = useState<WardBed[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
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
  return { beds, error, retry: () => setAttempt((a) => a + 1) };
}

const SUMMARY: { level: Level; word: string; rule: (tau: string, half: string) => string }[] = [
  { level: "ALERT", word: "Alert", rule: (tau) => `risk ≥ ${tau}` },
  { level: "WATCH", word: "Watch", rule: (_, half) => `risk ≥ ${half}` },
  { level: "OK", word: "OK", rule: (_, half) => `below ${half}` },
];

export function WardPage() {
  const { t, appliedThreshold } = useReplay();
  const { beds, error, retry } = useWard();
  const navigate = useNavigate();
  usePageTitle("Ward");

  if (error)
    return (
      <div className="state error" role="alert">
        {error} <button onClick={retry}>Retry</button>
      </div>
    );
  if (!beds) return <div className="state" role="status">Loading the ward…</div>;

  const active = beds.filter((b) => b.status === "active");
  const count = (l: Level) => active.filter((b) => b.level === l).length;
  const tau = pct(appliedThreshold, 1);
  const half = pct(appliedThreshold / 2, 2);

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>The ward at hour {t}</h1>
          <p>
            {active.length} of {beds.length} beds occupied · every bed is a real held-out ICU stay · highest calibrated risk first
          </p>
        </div>
        <ul className="summary" aria-label="Ward summary">
          {SUMMARY.map((s) => (
            <li key={s.level} className={`st-${s.level.toLowerCase()}`}>
              <StatusGlyph level={s.level} size={16} />
              <b className="num">{count(s.level)}</b>
              <span>
                <span className="status">{s.word}</span>
                <span className="rule">{s.rule(tau, half)}</span>
              </span>
            </li>
          ))}
          <li className="st-off">
            <b className="num">{beds.length - active.length}</b>
            <span>
              <span className="status">Discharged</span>
              <span className="rule">stay ended</span>
            </span>
          </li>
        </ul>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="wall-table">
          <caption className="vh">Demo ICU beds at replay hour {t}, highest calibrated risk first. Select a bed to open its record.</caption>
          <thead>
            <tr>
              <th scope="col">Bed</th>
              <th scope="col">Status</th>
              <th scope="col">Age / sex · unit</th>
              <th scope="col">Calibrated risk</th>
              <th scope="col">Last 12 h</th>
              <th scope="col" className="right">
                HR <span className="unit">bpm</span>
              </th>
              <th scope="col" className="right">
                MAP <span className="unit">mm Hg</span>
              </th>
              <th scope="col" className="right">
                Temp <span className="unit">°C</span>
              </th>
              <th scope="col" className="right">
                Resp <span className="unit">/min</span>
              </th>
              <th scope="col" className="right">
                SpO₂ <span className="unit">%</span>
              </th>
              <th scope="col" className="right">
                WBC <span className="unit">×10³/µL</span>
              </th>
              <th scope="col" className="right">
                In alert
              </th>
            </tr>
          </thead>
          <tbody>
            {beds.map((b) => {
              const off = b.status === "discharged";
              const href = `/bed/${b.patient_id}`;
              return (
                <tr key={b.patient_id} className={off ? "discharged" : ""} onClick={() => navigate(href)}>
                  <td>
                    <Link to={href} className="exhibit-no">
                      {bedNumber(b.display_name)}
                      <small>bed</small>
                    </Link>
                  </td>
                  <td>{off ? <span className="status st-off">Discharged after hour {b.hour}</span> : <Status level={b.level} />}</td>
                  <td className="nowrap">
                    {b.age} {sexName(b.gender)} · {b.unit === 1 || b.unit === 2 ? unitName(b.unit) : "—"}
                  </td>
                  <td>
                    <RiskBar risk={b.risk} threshold={appliedThreshold} level={off ? "OK" : b.level} />
                  </td>
                  <td>
                    <Sparkline values={b.risk_trend} threshold={appliedThreshold} />
                  </td>
                  <td className="right num">{fmt(b.vitals.HR)}</td>
                  <td className="right num">{fmt(b.vitals.MAP)}</td>
                  <td className="right num">{fmt(b.vitals.Temp, 1)}</td>
                  <td className="right num">{fmt(b.vitals.Resp)}</td>
                  <td className="right num">{fmt(b.vitals.O2Sat)}</td>
                  <td className="right num">{fmt(b.vitals.WBC, 1)}</td>
                  <td className="right num">{!off && b.alert && b.alert_since != null ? `${b.hour - b.alert_since + 1} h` : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="legend">
        Vitals are the last recorded value; labs such as WBC stay “—” until first measured. Watch means risk of at least half the alert
        threshold. Keys: Space plays or pauses, ← and → step one hour.
      </p>
    </main>
  );
}
