import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError, PatientDetail, pct, sexName, unitName } from "../api";
import { LabsChart } from "../components/LabsChart";
import { LevelChip } from "../components/LevelChip";
import { ReasonsPanel } from "../components/ReasonsPanel";
import { RiskChart } from "../components/RiskChart";
import { VitalsGrid } from "../components/VitalsGrid";
import { useReplay } from "../state/ReplayContext";

const OUTCOME: Record<string, string> = {
  septic_early: "Septic — flagged early",
  septic_missed_or_late: "Septic — missed or late",
  nonseptic_quiet: "Not septic — no alerts",
  nonseptic_false_alert: "Not septic — had false alerts",
};

export function PatientPage() {
  const { id = "" } = useParams();
  const { t, appliedThreshold, showTruth, config } = useReplay();
  const [p, setP] = useState<PatientDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!config) return;
    const ctl = new AbortController();
    api
      .patient(id, t, appliedThreshold, showTruth, ctl.signal)
      .then((d) => {
        setP(d);
        setError(null);
      })
      .catch((e) => {
        if (e.name === "AbortError") return;
        setError(e instanceof ApiError && e.status === 404 ? `No demo patient with id "${id}".` : e.message);
      });
    return () => ctl.abort();
  }, [id, t, appliedThreshold, showTruth, config]);

  if (error)
    return (
      <div className="state error">
        {error} <Link to="/">Back to ward</Link>
      </div>
    );
  if (!p) return <div className="state">Loading patient…</div>;

  const risk = p.risk[p.risk.length - 1];
  const firstAlert = p.alert.indexOf(true);
  return (
    <main className="page">
      <div className="patient-head card">
        <Link to="/" className="back">
          ← Ward
        </Link>
        <h1>{p.display_name}</h1>
        <dl className="facts">
          <div>
            <dt>Age / sex</dt>
            <dd>
              {p.age} {sexName(p.gender)}
            </dd>
          </div>
          <div>
            <dt>Unit</dt>
            <dd>{unitName(p.unit)}</dd>
          </div>
          <div>
            <dt>ICU hour</dt>
            <dd className="mono">
              {p.hour} <small>of {p.stay_hours - 1}</small>
              {p.status === "discharged" && <small> · discharged</small>}
            </dd>
          </div>
          <div>
            <dt>Risk now</dt>
            <dd className="mono">{pct(risk, 2)}</dd>
          </div>
          <div>
            <dt>First alert</dt>
            <dd className="mono">{firstAlert >= 0 ? `hour ${p.hours[firstAlert]}` : "none yet"}</dd>
          </div>
        </dl>
        <LevelChip level={p.level} />
      </div>

      {showTruth && p.outcome && (
        <div className={`truth ${p.septic ? "truth-septic" : ""}`}>
          <b>Ground truth (demo only):</b> {OUTCOME[p.outcome] ?? p.outcome}
          {p.onset_hour != null && (
            <>
              {" "}
              · Sepsis onset at ICU hour <b className="mono">{p.onset_hour}</b>
              {firstAlert >= 0 && p.hours[firstAlert] < p.onset_hour && (
                <> · first alert {p.onset_hour - p.hours[firstAlert]} h before onset</>
              )}
            </>
          )}
        </div>
      )}

      <div className="patient-grid">
        <section className="card">
          <h3>Calibrated sepsis risk</h3>
          <p className="sub">Shaded: hours in alert at the current threshold ({pct(p.threshold, 1)}).</p>
          <RiskChart p={p} showTruth={showTruth} />
        </section>
        <ReasonsPanel reasons={p.reasons} hour={p.hour} />
      </div>

      <section className="card">
        <h3>Vital signs</h3>
        <VitalsGrid p={p} showTruth={showTruth} />
      </section>
      <section className="card">
        <h3>Laboratory values</h3>
        <p className="sub">Dots mark hours when the test was resulted.</p>
        <LabsChart p={p} showTruth={showTruth} />
      </section>
    </main>
  );
}
