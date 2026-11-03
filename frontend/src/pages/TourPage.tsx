import { useEffect, useState } from "react";
import { api, ApiError, Metrics, pct } from "../api";
import { bedNumber } from "../components/Status";
import { useGoToStop } from "../components/TourRail";
import { useReplay } from "../state/ReplayContext";
import { TOUR, TourStop } from "../tour";
import { usePageTitle } from "../usePageTitle";

/** One computed sentence per stop, from the API at that stop's replay state. */
async function factFor(s: TourStop, tau: number, signal: AbortSignal): Promise<string> {
  if (s.route === "bed" && s.bed) {
    const p = await api.patient(s.bed, s.t, tau, true, signal);
    const fa = p.alert.indexOf(true);
    const alertTxt = fa >= 0 ? `first alert at hour ${p.hours[fa]}` : `no alert by hour ${p.hour}`;
    const truth = p.septic && p.onset_hour != null ? `sepsis onset at hour ${p.onset_hour}` : "never developed sepsis";
    return `Bed ${bedNumber(p.display_name)} · ${p.age} ${p.gender === 1 ? "M" : "F"} · ${alertTxt} · ${truth}`;
  }
  if (s.route === "ward") {
    const counts = await Promise.all(
      [0.015, tau, 0.1].map(async (x) => {
        const beds = await api.ward(s.t, x, signal);
        return `${beds.filter((b) => b.status === "active" && b.alert).length} at ${pct(x, 1)}`;
      }),
    );
    return `Active beds in alert at hour ${s.t}: ${counts.join(" · ")}`;
  }
  const m: Metrics = await api.metrics(signal);
  const u = m.lockbox_test.metrics.find((r) => r.metric === "utility_norm");
  const dev = m.dev_cv?.selection.nested_utility;
  return `Lockbox (LightGBM, once): utility ${Number(u?.value).toFixed(4)}${dev != null ? ` · development, nested CV (served ensemble): ${dev.toFixed(4)}, not comparable` : ""}`;
}

export function TourPage() {
  const { config, stop } = useReplay();
  const go = useGoToStop();
  const [facts, setFacts] = useState<Record<number, string>>({});
  const [error, setError] = useState<string | null>(null);
  usePageTitle("Tour");

  useEffect(() => {
    if (!config) return;
    const ctl = new AbortController();
    Promise.all(TOUR.map(async (s) => [s.n, await factFor(s, config.threshold, ctl.signal)] as const))
      .then((pairs) => setFacts(Object.fromEntries(pairs)))
      .catch((e) => {
        if (e.name !== "AbortError") setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => ctl.abort();
  }, [config]);

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>The tour</h1>
          <p>Six stops through the story, in order. Each stop sets the replay hour, ground truth and threshold, then opens the right screen.</p>
        </div>
      </div>
      {error && (
        <p className="state error" role="alert">
          {error}
        </p>
      )}
      <ol className="tour-list">
        {TOUR.map((s) => (
          <li key={s.n} className={`tour-item ${stop === s.n ? "current" : ""}`}>
            <span className="n" aria-hidden="true">
              {s.n}
            </span>
            <div>
              <h2>
                <span className="vh">Stop {s.n}: </span>
                {s.title}
              </h2>
              <p>{s.prompt}</p>
              <p className="fact">{facts[s.n] ?? "Loading the facts for this stop…"}</p>
            </div>
            <button className={stop === s.n ? "" : "primary"} onClick={() => go(s)}>
              {stop === s.n ? "Open again" : "Go to stop"}
            </button>
          </li>
        ))}
      </ol>
    </main>
  );
}
