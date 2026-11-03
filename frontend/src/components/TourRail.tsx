import { useNavigate } from "react-router-dom";
import { useReplay } from "../state/ReplayContext";
import { TOUR, TourStop } from "../tour";

/** Apply a tour stop: set the replay hour, ground truth and policy threshold, then go there. */
export function useGoToStop() {
  const r = useReplay();
  const navigate = useNavigate();
  return (s: TourStop) => {
    if (r.playing) r.togglePlay();
    r.setStop(s.n);
    r.setT(s.t);
    r.setShowTruth(s.truth);
    if (r.config) r.setThreshold(r.config.threshold);
    navigate(s.route === "bed" && s.bed ? `/bed/${s.bed}` : s.route === "evidence" ? "/evidence" : "/");
  };
}

/** Numbered stops along the bottom of every page, like a gallery audio-guide route. */
export function TourRail() {
  const { stop } = useReplay();
  const go = useGoToStop();
  return (
    <nav className="tour-rail" aria-label="Presenter tour">
      <span className="t">Tour</span>
      {TOUR.map((s) => (
        <button key={s.n} className="stop" aria-current={stop === s.n ? "step" : undefined} onClick={() => go(s)}>
          <i>{s.n}</i>
          {s.title}
        </button>
      ))}
    </nav>
  );
}
