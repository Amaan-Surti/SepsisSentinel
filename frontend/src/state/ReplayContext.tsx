import { createContext, ReactNode, useCallback, useContext, useEffect, useState } from "react";
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

export function ReplayProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [t, setTRaw] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(500);
  const [threshold, setThreshold] = useState(0.03);
  const [appliedThreshold, setApplied] = useState(0.03);
  const [showTruth, setShowTruth] = useState(false);

  useEffect(() => {
    const ctl = new AbortController();
    setConfigError(null);
    api
      .config(ctl.signal)
      .then((c) => {
        setConfig(c);
        setThreshold(c.threshold);
        setApplied(c.threshold);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setConfigError(e instanceof ApiError ? e.message : String(e));
      });
    return () => ctl.abort();
  }, [attempt]);

  const maxHour = config?.max_hour ?? 0;
  const setT = useCallback((x: number) => setTRaw(Math.max(0, Math.min(maxHour, Math.round(x)))), [maxHour]);
  const step = useCallback((d: number) => setTRaw((cur) => Math.max(0, Math.min(maxHour, cur + d))), [maxHour]);

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

  const value: ReplayState = {
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
    togglePlay: () => setPlaying((p) => (t >= maxHour ? false : !p)),
    reset: () => {
      setPlaying(false);
      setTRaw(0);
    },
    setSpeedMs,
    setThreshold,
    setShowTruth,
    retryConfig: () => setAttempt((a) => a + 1),
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useReplay(): ReplayState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useReplay must be used inside ReplayProvider");
  return v;
}
