import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError, PatientDetail, pct, sexName, WardBed } from "../api";
import { BackIcon } from "../components/Icons";
import { MiniSeries } from "../components/MiniSeries";
import { ReasonsPanel } from "../components/ReasonsPanel";
import { RiskChart } from "../components/RiskChart";
import { bedNumber, levelAt, Status, StatusChip } from "../components/Status";
import { useReplay } from "../state/ReplayContext";
import { usePageTitle } from "../usePageTitle";
import { ALL_VARIABLES, fmtNormal, fmtValue, GROUPS } from "../variables";
import { useWard } from "./WardPage";

const UNIT_LONG: Record<number, string> = { 1: "Medical ICU", 2: "Surgical ICU" };
const SEX_LONG: Record<number, string> = { 0: "Woman", 1: "Man" };

type Tab = "label" | "measurements" | "hours";

function usePatient(id: string) {
  const { t, appliedThreshold, showTruth, config } = useReplay();
  const [p, setP] = useState<PatientDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    setP(null);
    setError(null);
  }, [id]);
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
        setError(e instanceof ApiError && e.status === 404 ? `No demo bed with id "${id}".` : e.message);
      });
    return () => ctl.abort();
  }, [id, t, appliedThreshold, showTruth, config, attempt]);
  return { p, error, retry: () => setAttempt((a) => a + 1) };
}

/** The wall, collapsed to a strip while an exhibit is open; everything but the focused bed is dimmed. */
function Strip({ beds, focus }: { beds: WardBed[] | null; focus: string }) {
  return (
    <nav className="strip" aria-label="All beds">
      <h2>The ward · highest risk first</h2>
      <ol>
        {(beds ?? []).map((b) => {
          const off = b.status === "discharged";
          return (
            <li key={b.patient_id}>
              <Link to={`/bed/${b.patient_id}`} className={b.patient_id === focus ? "focus" : ""} aria-current={b.patient_id === focus ? "page" : undefined}>
                <span className="n">{bedNumber(b.display_name)}</span>
                <Status level={b.level} off={off} size={12} />
                <span className="num">{pct(b.risk, 2)}</span>
              </Link>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/** The WHAT HAPPENED sentence and the exact case, from the hours replayed so far. */
function whatHappened(p: PatientDetail): { line: JSX.Element; label: string } | null {
  if (!p.outcome) return null;
  const firstAlert = p.alert.indexOf(true);
  const alertHours = p.alert.filter(Boolean).length;
  if (p.septic && p.onset_hour != null) {
    const fa = firstAlert >= 0 ? p.hours[firstAlert] : null;
    const onset = <b>{p.onset_hour}</b>;
    if (fa == null)
      return {
        line: <>Sepsis onset at ICU hour {onset}: no alert {p.hour >= p.onset_hour ? "came before it" : "so far"}.</>,
        label: p.hour >= p.onset_hour ? "Septic, missed" : "Septic, not flagged yet",
      };
    const lead = p.onset_hour - fa;
    if (lead <= 0)
      return { line: <>Sepsis onset at ICU hour {onset}: the first alert came <span className="nowrap"><b>{-lead} h</b> after it.</span></>, label: "Septic, flagged late" };
    return {
      line: <>Sepsis onset at ICU hour {onset}: the first alert came <span className="nowrap"><b>{lead} h</b> before it.</span></>,
      label: lead > 12 ? "Septic, flagged more than 12 h before onset" : lead < 4 ? "Septic, flagged less than 4 h before onset" : "Septic, flagged 4–12 h before onset",
    };
  }
  return {
    line:
      alertHours > 0 ? (
        <>
          Never developed sepsis: the <b>{alertHours}</b> alert hour{alertHours === 1 ? "" : "s"} so far {alertHours === 1 ? "was a false alarm" : "were false alarms"}.
        </>
      ) : (
        <>Never developed sepsis, and no alert so far.</>
      ),
    label: alertHours > 0 ? "Not septic, had false alerts" : "Not septic, no alerts",
  };
}

function Reveal({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const w = showTruth ? whatHappened(p) : null;
  if (!w)
    return (
      <p className="reveal hidden">
        <span className="k">WHAT HAPPENED</span>Ground truth is hidden. Turn on “Show ground truth (demo)” to reveal this patient's outcome.
      </p>
    );
  return (
    <p className="reveal">
      <span className="k">WHAT HAPPENED</span>
      <span>{w.line}</span>
      <span className="sub" style={{ margin: 0 }}>
        {w.label}
      </span>
    </p>
  );
}

function LabelTab({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const { config } = useReplay();
  const onset = showTruth ? p.onset_hour ?? null : null;
  return (
    <>
      <div className="overview">
        <section aria-labelledby="risk-title">
          <h3 id="risk-title">Calibrated sepsis risk, hour by hour</h3>
          <p className="sub">Shaded hours are in alert at the current threshold ({pct(p.threshold, 1)}).</p>
          <RiskChart p={p} showTruth={showTruth} weight={config?.weight_lgbm ?? null} height={window.innerHeight <= 760 ? 250 : 330} />
        </section>
        <ReasonsPanel reasons={p.reasons} hour={p.hour} kind={config?.model_kind ?? "lgbm"} />
      </div>
      <section aria-labelledby="vitals-now" style={{ marginTop: 22 }}>
        <h3 id="vitals-now">Vital signs</h3>
        <p className="sub">All eight hourly vitals; the full set of 34 measurements is under “Every measurement”.</p>
        <div className="now-vitals">
          {GROUPS[0].vars.map((v) => (
            <MiniSeries key={v.key} v={v} values={p.vitals[v.key] ?? []} hours={p.hours} xMax={p.stay_hours - 1} onset={onset} />
          ))}
        </div>
      </section>
      <Reveal p={p} showTruth={showTruth} />
    </>
  );
}

function MeasurementsTab({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const [carry, setCarry] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const series = (key: string) => p.vitals[key] ?? p.labs[key] ?? [];
  const onset = showTruth ? p.onset_hour ?? null : null;
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [p.hour]);
  const measuredCount = ALL_VARIABLES.filter((v) => series(v.key).some((x) => x != null)).length;
  return (
    <>
      {GROUPS.map((g) => (
        <section className="group" key={g.id} aria-labelledby={`g-${g.id}`}>
          <div className="group-head">
            <h3 id={`g-${g.id}`}>{g.title}</h3>
            <span>{g.vars.length} measurements</span>
          </div>
          <div className="minis">
            {g.vars.map((v) => (
              <MiniSeries key={v.key} v={v} values={series(v.key)} hours={p.hours} xMax={p.stay_hours - 1} onset={onset} />
            ))}
          </div>
        </section>
      ))}
      <section className="flow-wrap" aria-labelledby="flow-title">
        <div className="group-head">
          <h3 id="flow-title">Flowsheet: every measurement, every hour</h3>
          <span>
            {measuredCount} of {ALL_VARIABLES.length} measured so far · hours 0–{p.hour}
          </span>
        </div>
        <label className="filter">
          <input type="checkbox" checked={carry} onChange={(e) => setCarry(e.target.checked)} /> Show the last value carried forward (grey) where
          nothing new was measured
        </label>
        <div className="flow-scroll" ref={scroller} tabIndex={0} aria-label="Flowsheet, scrolls sideways">
          <table className="flow">
            <thead>
              <tr>
                <th scope="col">Measurement</th>
                {p.hours.map((h) => (
                  <th scope="col" key={h} className={h === p.hour ? "cur" : ""}>
                    h{h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Risk</th>
                {p.risk.map((r, i) => (
                  <td key={i} className={p.hours[i] === p.hour ? "cur" : ""}>
                    {pct(r, 2)}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Status</th>
                {p.risk.map((r, i) => {
                  const l = levelAt(r, p.alert[i], p.threshold);
                  return (
                    <td key={i} className={p.hours[i] === p.hour ? "cur" : ""}>
                      {l === "ALERT" ? "Alert" : l === "WATCH" ? "Watch" : "OK"}
                    </td>
                  );
                })}
              </tr>
              {GROUPS.map((g) => [
                <tr className="grp" key={`h-${g.id}`}>
                  <th scope="rowgroup" colSpan={p.hours.length + 1}>
                    <span className="grp-title">{g.title}</span>
                  </th>
                </tr>,
                ...g.vars.map((v) => {
                  const s = series(v.key);
                  let lastSeen: number | null = null;
                  return (
                    <tr key={v.key}>
                      <th scope="row">
                        {v.label} {v.unit && <span className="unit">{v.unit}</span>}
                        <span className="flow-normal">normal {fmtNormal(v)}</span>
                      </th>
                      {s.map((x, i) => {
                        if (x != null) lastSeen = x;
                        const cur = p.hours[i] === p.hour ? "cur" : "";
                        if (x != null) return <td key={i} className={cur}><b>{fmtValue(x, v.digits)}</b></td>;
                        return (
                          <td key={i} className={`${cur} ${carry && lastSeen != null ? "carry" : ""}`}>
                            {carry && lastSeen != null ? fmtValue(lastSeen, v.digits) : "·"}
                          </td>
                        );
                      })}
                    </tr>
                  );
                }),
              ])}
            </tbody>
          </table>
        </div>
        <p className="caveat">Bold = measured that hour; “·” = not measured. Units as documented for the PhysioNet 2019 Challenge data.</p>
      </section>
    </>
  );
}

type HourFilter = "all" | "alert" | "results";

function HoursTab({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const [filter, setFilter] = useState<HourFilter>("all");
  const insights = useMemo(
    () =>
      p.hours.map((h, i) => {
        const measured = ALL_VARIABLES.flatMap((v) => {
          const x = (p.vitals[v.key] ?? p.labs[v.key] ?? [])[i];
          return x == null ? [] : [{ v, x }];
        });
        const labs = measured.filter((m) => p.labs[m.v.key] !== undefined);
        return {
          h,
          i,
          risk: p.risk[i],
          delta: i > 0 ? p.risk[i] - p.risk[i - 1] : null,
          level: levelAt(p.risk[i], p.alert[i], p.threshold),
          lgbm: p.risk_components?.lgbm[i] ?? null,
          gru: p.risk_components?.gru[i] ?? null,
          measured,
          newLabs: labs.length,
          reasons: p.reasons_by_hour[i] ?? [],
        };
      }),
    [p],
  );
  const shown = insights
    .filter((x) => (filter === "alert" ? x.level === "ALERT" : filter === "results" ? x.newLabs > 0 : true))
    .reverse();
  return (
    <section aria-labelledby="hours-title">
      <div className="group-head">
        <h3 id="hours-title">Hour by hour</h3>
        <span>Newest first · what was measured, what the model said, and why</span>
      </div>
      <div className="filter" role="radiogroup" aria-label="Which hours">
        {(
          [
            ["all", `All ${insights.length} hours`],
            ["alert", `Alert hours (${insights.filter((x) => x.level === "ALERT").length})`],
            ["results", `Hours with new lab results (${insights.filter((x) => x.newLabs > 0).length})`],
          ] as [HourFilter, string][]
        ).map(([k, label]) => (
          <label key={k}>
            <input type="radio" name="hours-filter" checked={filter === k} onChange={() => setFilter(k)} /> {label}
          </label>
        ))}
      </div>
      <ol className="hours">
        {shown.map((x) => (
          <li key={x.h} className={`hour ${x.h === p.hour ? "current" : ""}`}>
            <div className="h">
              Hour {x.h}
              <small>{x.h === p.hour ? "now" : `${p.hour - x.h} h ago`}</small>
              {showTruth && p.onset_hour === x.h && <small className="onset">Sepsis onset</small>}
            </div>
            <div>
              <div className="rk">
                {pct(x.risk, 2)}
                <small>
                  {x.delta == null ? "first hour" : `${x.delta >= 0 ? "+" : "−"}${(Math.abs(x.delta) * 100).toFixed(2)} pts vs hour ${x.h - 1}`}
                  {x.lgbm != null && x.gru != null && (
                    <>
                      <br />
                      LightGBM {pct(x.lgbm, 2)} · GRU {pct(x.gru, 2)}
                    </>
                  )}
                </small>
              </div>
              <Status level={x.level} />
            </div>
            <div>
              <p className="measured">
                {x.measured.length === 0 ? (
                  "Nothing new measured this hour."
                ) : (
                  <>
                    Measured ({x.measured.length}):{" "}
                    {x.measured.map((m, j) => (
                      <span key={m.v.key}>
                        {j > 0 && " · "}
                        {m.v.label} <b>{fmtValue(m.x, m.v.digits)}</b>
                      </span>
                    ))}
                  </>
                )}
              </p>
              {x.reasons.length > 0 && (
                <div className="reasons" aria-label="Top reasons this hour">
                  {x.reasons.map((r) => (
                    <span key={r.feature}>{r.text}</span>
                  ))}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function BedPage() {
  const { id = "" } = useParams();
  const { showTruth, config } = useReplay();
  const { beds } = useWard();
  const { p, error, retry } = usePatient(id);
  const [tab, setTab] = useState<Tab>("label");
  usePageTitle(p ? p.display_name : "Bed");

  if (error)
    return (
      <div className="state error" role="alert">
        {error} <button onClick={retry}>Retry</button> <Link to="/">Back to the ward</Link>
      </div>
    );
  if (!p) return <div className="state" role="status">Loading the bed…</div>;

  const risk = p.risk[p.risk.length - 1];
  const firstAlert = p.alert.indexOf(true);
  const off = p.status === "discharged";
  const n = bedNumber(p.display_name);
  const adm = p.hosp_adm_time;
  const tabs: [Tab, string, string][] = [
    ["label", "Label", ""],
    ["measurements", "Every measurement", `${ALL_VARIABLES.length}`],
    ["hours", "Hour by hour", `${p.hours.length}`],
  ];

  return (
    <main className="page">
      <div className="exhibit">
        <Strip beds={beds} focus={id} />
        <article className="label" aria-labelledby="bed-title">
          <div className="label-top">
            <Link to="/" className="back">
              <BackIcon size={14} />
              The ward
            </Link>
            <StatusChip level={p.level} off={off} />
          </div>
          <div className="label-head">
            <h1 id="bed-title" className="bed-no">
              Bed {n}
            </h1>
            <div>
              <p className="tomb">
                {SEX_LONG[p.gender] ?? sexName(p.gender)}, {p.age} · {UNIT_LONG[p.unit] ?? "ICU unit not recorded"} · ICU hour <b>{p.hour}</b> of{" "}
                {p.stay_hours - 1}
                {off && " · discharged"}
                <br />
                Calibrated risk now <b>{pct(risk, 2)}</b> · {firstAlert >= 0 ? <>first alert at hour <b>{p.hours[firstAlert]}</b></> : "no alert yet"}
                {p.alert_since != null && (
                  <>
                    {" "}
                    · <span className="nowrap">in alert for <b>{p.hour - p.alert_since + 1} h</b></span>
                  </>
                )}
              </p>
              {showTruth && whatHappened(p) && (
                <p className="tomb-reveal">
                  <span className="k">WHAT HAPPENED</span> {whatHappened(p)!.line}
                </p>
              )}
              <p className="catalogue">
                EXHIBIT {n} · HELD-OUT TEST PATIENT {p.patient_id.toUpperCase()} · {config?.model_label.toUpperCase()}
                {adm != null && ` · ADMITTED TO HOSPITAL ${adm < -0.5 ? `${Math.round(-adm)} H BEFORE ICU` : "AT ICU ADMISSION"}`}
              </p>
            </div>
          </div>

          <div className="tabs" role="tablist" aria-label="Exhibit views">
            {tabs.map(([k, label, count]) => (
              <button key={k} role="tab" id={`tab-${k}`} aria-selected={tab === k} aria-controls={`panel-${k}`} onClick={() => setTab(k)}>
                {label}
                {count && <span className="count">{count}</span>}
              </button>
            ))}
          </div>
          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
            {tab === "label" && <LabelTab p={p} showTruth={showTruth} />}
            {tab === "measurements" && <MeasurementsTab p={p} showTruth={showTruth} />}
            {tab === "hours" && <HoursTab p={p} showTruth={showTruth} />}
          </div>
        </article>
      </div>
    </main>
  );
}
