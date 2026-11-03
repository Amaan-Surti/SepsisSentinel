import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, ApiError, Config } from "../api";

export const SPEEDS = [
  { label: "1×", ms: 1000 },
  { label: "2×", ms: 500 },
  { label: "5×", ms: 200 },
];

interface ReplayState {
  config: Config | null;
  configError: string | null;
  t: number;
  playing: boolean;
  speedMs: number;
  threshold: number; // live slider value
  appliedThreshold: number; // debounced (200 ms) value used for fetching
  showTruth: boolean;
  setT: (t: number) => void;
  step: (d: number) => void;
  togglePlay: () => void;
  reset: () => void;
  setSpeedMs: (ms: number) => void;
  setThreshold: (x: number) => void;
  setShowTruth: (b: boolean) => void;
  retryConfig: () => void;
}

const Ctx = createContext<ReplayState | null>(null);

/** Deep links for the demo: ?t=<hour>&truth=1 (read once on load). */
function initialFromUrl(): { t: number; truth: boolean } {
  const q = new URLSearchParams(window.location.search);
  const t = Number(q.get("t"));
  return { t: Number.isFinite(t) && t > 0 ? Math.round(t) : 0, truth: q.get("truth") === "1" };
}

/** Keys typed into a control keep their native meaning. */
function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  return el.isContentEditable || ["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A"].includes(el.tagName);
}

export function ReplayProvider({ children }: { children: ReactNode }) {
  const [init] = useState(initialFromUrl);
  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [t, setTRaw] = useState(init.t);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(500);
  const [threshold, setThreshold] = useState(0.03);
  const [appliedThreshold, setApplied] = useState(0.03);
  const [showTruth, setShowTruth] = useState(init.truth);

  useEffect(() => {
    const ctl = new AbortController();
    setConfigError(null);
    api
      .config(ctl.signal)
      .then((c) => {
        setConfig(c);
        setThreshold(c.threshold);
        setApplied(c.threshold);
        setTRaw((cur) => Math.min(cur, c.max_hour));
      })
      .catch((e) => {
        if (e.name !== "AbortError") setConfigError(e instanceof ApiError ? e.message : String(e));
      });
    return () => ctl.abort();
  }, [attempt]);

  const maxHour = config?.max_hour ?? 0;
  const setT = useCallback((x: number) => setTRaw(Math.max(0, Math.min(maxHour, Math.round(x)))), [maxHour]);
  const step = useCallback((d: number) => setTRaw((cur) => Math.max(0, Math.min(maxHour, cur + d))), [maxHour]);
  const togglePlay = useCallback(() => setPlaying((p) => (t >= maxHour ? false : !p)), [t, maxHour]);
  const reset = useCallback(() => {
    setPlaying(false);
    setTRaw(0);
  }, []);
  const retryConfig = useCallback(() => setAttempt((a) => a + 1), []);

  useEffect(() => {
    if (!playing) return;
    if (t >= maxHour) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setTRaw((cur) => Math.min(maxHour, cur + 1)), speedMs);
    return () => window.clearTimeout(id);
  }, [playing, t, speedMs, maxHour]);

  useEffect(() => {
    const id = window.setTimeout(() => setApplied(threshold), 200);
    return () => window.clearTimeout(id);
  }, [threshold]);

  // Presenter shortcuts: Space = play/pause, ← / → = step one hour
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!config || e.altKey || e.ctrlKey || e.metaKey || isTypingTarget(e.target)) return;
      if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        step(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [config, togglePlay, step]);

  const value = useMemo<ReplayState>(
    () => ({
      config,
      configError,
      t,
      playing,
      speedMs,
      threshold,
      appliedThreshold,
      showTruth,
      setT,
      step,
      togglePlay,
      reset,
      setSpeedMs,
      setThreshold,
      setShowTruth,
      retryConfig,
    }),
    [config, configError, t, playing, speedMs, threshold, appliedThreshold, showTruth, setT, step, togglePlay, reset, retryConfig],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useReplay(): ReplayState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useReplay must be used inside ReplayProvider");
  return v;
}
