// The 34 hourly measurements in the PhysioNet 2019 data, with units as documented by the Challenge
// (FiO2 is stored as a fraction after cleaning). Normal ranges are typical adult reference values for
// orientation only, not ICU treatment targets. Lactate and magnesium are stored on the scale of
// mmol/L and mg/dL respectively, whatever the Challenge's unit label says, so their ranges use that scale. Groups follow the order a clinician reads them.

export interface Variable {
  key: string;
  label: string;
  unit: string;
  digits: number;
  /** Typical adult reference range [low, high], on the same scale as the stored values. */
  normal: [number, number];
}

export interface VariableGroup {
  id: string;
  title: string;
  kind: "vitals" | "labs";
  vars: Variable[];
}

const v = (key: string, label: string, unit: string, digits: number, normal: [number, number]): Variable => ({
  key,
  label,
  unit,
  digits,
  normal,
});

export const GROUPS: VariableGroup[] = [
  {
    id: "vitals",
    title: "Vital signs",
    kind: "vitals",
    vars: [
      v("HR", "Heart rate", "beats/min", 0, [60, 100]),
      v("O2Sat", "Pulse oximetry (SpO₂)", "%", 0, [95, 100]),
      v("Temp", "Temperature", "°C", 1, [36.5, 37.5]),
      v("SBP", "Systolic blood pressure", "mm Hg", 0, [90, 120]),
      v("MAP", "Mean arterial pressure", "mm Hg", 0, [70, 100]),
      v("DBP", "Diastolic blood pressure", "mm Hg", 0, [60, 80]),
      v("Resp", "Respiratory rate", "breaths/min", 0, [12, 20]),
      v("EtCO2", "End-tidal CO₂", "mm Hg", 0, [35, 45]),
    ],
  },
  {
    id: "gas",
    title: "Blood gas and ventilation",
    kind: "labs",
    vars: [
      v("pH", "pH", "", 2, [7.35, 7.45]),
      v("PaCO2", "PaCO₂", "mm Hg", 0, [35, 45]),
      v("SaO2", "Arterial O₂ saturation", "%", 0, [95, 100]),
      v("BaseExcess", "Base excess", "mmol/L", 0, [-2, 2]),
      v("HCO3", "Bicarbonate (HCO₃)", "mmol/L", 0, [22, 26]),
      v("FiO2", "Fraction of inspired O₂", "fraction", 2, [0.21, 0.21]),
    ],
  },
  {
    id: "perfusion",
    title: "Perfusion and cardiac",
    kind: "labs",
    vars: [v("Lactate", "Lactate", "mg/dL", 1, [0.5, 2]), v("TroponinI", "Troponin I", "ng/mL", 2, [0, 0.04])],
  },
  {
    id: "chemistry",
    title: "Chemistry and kidney",
    kind: "labs",
    vars: [
      v("BUN", "Blood urea nitrogen", "mg/dL", 0, [7, 20]),
      v("Creatinine", "Creatinine", "mg/dL", 2, [0.6, 1.2]),
      v("Glucose", "Glucose", "mg/dL", 0, [70, 140]),
      v("Potassium", "Potassium", "mmol/L", 1, [3.5, 5]),
      v("Chloride", "Chloride", "mmol/L", 0, [98, 106]),
      v("Calcium", "Calcium", "mg/dL", 1, [8.5, 10.5]),
      v("Magnesium", "Magnesium", "mmol/dL", 1, [1.7, 2.2]),
      v("Phosphate", "Phosphate", "mg/dL", 1, [2.5, 4.5]),
    ],
  },
  {
    id: "liver",
    title: "Liver",
    kind: "labs",
    vars: [
      v("AST", "Aspartate transaminase (AST)", "IU/L", 0, [10, 40]),
      v("Alkalinephos", "Alkaline phosphatase", "IU/L", 0, [44, 147]),
      v("Bilirubin_total", "Total bilirubin", "mg/dL", 1, [0.1, 1.2]),
      v("Bilirubin_direct", "Direct bilirubin", "mg/dL", 1, [0, 0.3]),
    ],
  },
  {
    id: "blood",
    title: "Blood count and clotting",
    kind: "labs",
    vars: [
      v("WBC", "White cell count", "×10³/µL", 1, [4, 11]),
      v("Hgb", "Haemoglobin", "g/dL", 1, [12, 17.5]),
      v("Hct", "Haematocrit", "%", 0, [36, 50]),
      v("Platelets", "Platelets", "×10³/µL", 0, [150, 450]),
      v("PTT", "Partial thromboplastin time", "s", 1, [25, 35]),
      v("Fibrinogen", "Fibrinogen", "mg/dL", 0, [200, 400]),
    ],
  },
];

export const ALL_VARIABLES: Variable[] = GROUPS.flatMap((g) => g.vars);
export const VARIABLE: Record<string, Variable> = Object.fromEntries(ALL_VARIABLES.map((x) => [x.key, x]));

export const fmtValue = (x: number | null | undefined, digits: number) => (x == null ? "—" : x.toFixed(digits));

/** "60–100 beats/min", or "room air 0.21" for FiO2. */
export const fmtNormal = (v: Variable) =>
  v.normal[0] === v.normal[1]
    ? `room air ${fmtValue(v.normal[0], v.digits)}`
    : `${fmtValue(v.normal[0], v.digits)}–${fmtValue(v.normal[1], v.digits)}${v.unit && v.unit !== "fraction" ? ` ${v.unit}` : ""}`;

/** Where a value sits relative to the normal range. FiO2 above room air counts as "above". */
export const whereVsNormal = (v: Variable, x: number): "low" | "normal" | "high" =>
  x < v.normal[0] - 1e-9 ? "low" : x > v.normal[1] + 1e-9 ? "high" : "normal";
