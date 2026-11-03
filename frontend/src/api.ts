// Typed fetch helpers for the SepsisSentinel API (all routes under /api).

export type Level = "ALERT" | "WATCH" | "OK";
export type ModelKind = "ensemble" | "lgbm";

export interface Config {
  threshold: number;
  k: number;
  sticky: number;
  max_hour: number;
  n_patients: number;
  model_kind: ModelKind;
  model_label: string;
  model_version: string;
  weight_lgbm: number | null;
  reasons_source: string;
  disclaimer: string;
}

export interface Reason {
  text: string;
  feature: string;
  value: number | null;
  shap: number;
}

export interface WardBed {
  patient_id: string;
  display_name: string;
  age: number;
  gender: number;
  unit: number;
  hour: number;
  status: "active" | "discharged";
  risk: number;
  risk_trend: number[];
  alert: boolean;
  alert_since: number | null;
  level: Level;
  vitals: Record<string, number | null>;
}

export interface PatientDetail {
  patient_id: string;
  display_name: string;
  age: number;
  gender: number;
  unit: number;
  stay_hours: number;
  hour: number;
  status: "active" | "discharged";
  level: Level;
  alert_since: number | null;
  threshold: number;
  hours: number[];
  vitals: Record<string, (number | null)[]>;
  labs: Record<string, (number | null)[]>;
  risk: number[];
  alert: boolean[];
  reasons: Reason[];
  onset_hour?: number | null;
  septic?: boolean | null;
  outcome?: string | null;
}

export type Row = Record<string, string | number | null>;

export interface LockboxTest {
  label: string;
  model: string;
  note: string;
  metrics: Row[];
  baselines: Row[];
  subgroups: Row[];
  policy: { threshold: number; k: number; sticky: number; utility_norm: number };
}

export interface DevCV {
  label: string;
  note: string;
  comparison: Row[];
  weights: Row[];
  bootstrap: Row[];
  final: Row[];
  selection: {
    weights: number[];
    chosen_variant: string;
    reason: string;
    pooled_weight: number;
    pooled_policy: { threshold: number; k: number; sticky: number };
    nested_utility: number;
  };
}

export interface Metrics {
  active_model: ModelKind;
  lockbox_test: LockboxTest;
  dev_cv: DevCV | null;
  cross_hospital: Row[];
  ablations: Row[];
  dev_oof: Row[];
  threshold_curve: {
    threshold: number;
    utility_norm: number;
    patient_sensitivity: number;
    false_alerts_per_patient_day: number;
    median_lead_time_h: number;
  }[];
  policy: { threshold: number; k: number; sticky: number; utility_norm?: number };
}

export interface ModelCard {
  title: string;
  model_kind: ModelKind;
  model_version: string;
  markdown: string;
  sections: Record<string, string>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message);
  }
}

export const API_DOWN = "Cannot reach the API at /api — is the backend running?";

async function get<T>(path: string, params: Record<string, string | number | boolean> = {}, signal?: AbortSignal): Promise<T> {
  const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  let res: Response;
  try {
    res = await fetch(`/api${path}${qs ? `?${qs}` : ""}`, { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(API_DOWN);
  }
  if (res.status === 404) throw new ApiError("Not found", 404);
  if (!res.ok) {
    // Vite's dev proxy answers 500/502 when the backend is down
    throw new ApiError(res.status >= 500 ? API_DOWN : `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  config: (signal?: AbortSignal) => get<Config>("/config", {}, signal),
  ward: (t: number, threshold: number, signal?: AbortSignal) => get<WardBed[]>("/ward", { t, threshold }, signal),
  patient: (id: string, t: number, threshold: number, showTruth: boolean, signal?: AbortSignal) =>
    get<PatientDetail>(`/patients/${encodeURIComponent(id)}`, { t, threshold, show_truth: showTruth }, signal),
  metrics: (signal?: AbortSignal) => get<Metrics>("/metrics", {}, signal),
  modelCard: (signal?: AbortSignal) => get<ModelCard>("/model-card", {}, signal),
  figureUrl: (name: string) => `/api/figures/${name}`,
};

export const unitName = (u: number) => (u === 1 ? "MICU" : u === 2 ? "SICU" : "Unknown");
export const sexName = (g: number) => (g === 1 ? "M" : "F");
export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
