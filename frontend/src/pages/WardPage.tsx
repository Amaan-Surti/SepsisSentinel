import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError, sexName, unitName, WardBed } from "../api";
import { LevelChip } from "../components/LevelChip";
import { RiskBar } from "../components/RiskBar";
import { Sparkline } from "../components/Sparkline";
import { useReplay } from "../state/ReplayContext";

const fmt = (x: number | null | undefined, d = 0) => (x == null ? "—" : x.toFixed(d));

export function WardPage() {
  const { t, appliedThreshold, config } = useReplay();
  const [beds, setBeds] = useState<WardBed[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

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
  }, [t, appliedThreshold, config]);

  if (error) return <div className="state error">{error}</div>;
  if (!beds) return <div className="state">Loading ward…</div>;

  const active = beds.filter((b) => b.status === "active");
  const count = (l: string) => active.filter((b) => b.level === l).length;

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>ICU ward · replay</h1>
          <p className="sub">
            {active.length} of {beds.length} beds occupied at ICU hour {t}. Sorted by current calibrated risk.
          </p>
        </div>
        <div className="summary" aria-label="Ward summary">
          <div className="summary-item alert">
            <b>{count("ALERT")}</b> Alert
          </div>
          <div className="summary-item watch">
            <b>{count("WATCH")}</b> Watch
          </div>
          <div className="summary-item ok">
            <b>{count("OK")}</b> OK
          </div>
          <div className="summary-item muted">
            <b>{beds.length - active.length}</b> Discharged
          </div>
        </div>
      </div>

      <div className="card table-card">
        <table className="ward">
          <thead>
            <tr>
              <th>Bed</th>
              <th>Age/Sex</th>
              <th>Unit</th>
              <th className="num">ICU hr</th>
              <th>Risk</th>
              <th>12-h trend</th>
              <th className="num">HR</th>
              <th className="num">MAP</th>
              <th className="num">Temp</th>
              <th className="num">WBC</th>
              <th>Level</th>
              <th>Alert since</th>
            </tr>
          </thead>
          <tbody>
            {beds.map((b) => {
              const off = b.status === "discharged";
              return (
                <tr
                  key={b.patient_id}
                  className={`${off ? "discharged" : ""} level-${b.level.toLowerCase()}`}
                  tabIndex={0}
                  onClick={() => navigate(`/patient/${b.patient_id}`)}
                  onKeyDown={(e) => e.key === "Enter" && navigate(`/patient/${b.patient_id}`)}
                  aria-label={`${b.display_name}, ${b.level}${off ? ", discharged" : ""}`}
                >
                  <td className="bed">{b.display_name}</td>
                  <td>
                    {b.age} {sexName(b.gender)}
                  </td>
                  <td>{unitName(b.unit)}</td>
                  <td className="num mono">{off ? "disch." : b.hour}</td>
                  <td>
                    <RiskBar risk={b.risk} threshold={appliedThreshold} level={b.level} />
                  </td>
                  <td>
                    <Sparkline values={b.risk_trend} threshold={appliedThreshold} />
                  </td>
                  <td className="num mono">{fmt(b.vitals.HR)}</td>
                  <td className="num mono">{fmt(b.vitals.MAP)}</td>
                  <td className="num mono">{fmt(b.vitals.Temp, 1)}</td>
                  <td className="num mono">{fmt(b.vitals.WBC, 1)}</td>
                  <td>{off ? <span className="disch-label">Discharged</span> : <LevelChip level={b.level} />}</td>
                  <td className="mono">{!off && b.alert && b.alert_since != null ? `for ${b.hour - b.alert_since + 1} h` : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="legend">
        <b>Alert</b>: calibrated risk ≥ threshold. <b>Watch</b>: risk ≥ ½ threshold. Vitals show the last recorded value. Click a bed for its
        trajectory and explanation.
      </p>
    </main>
  );
}
