// The 34 hourly measurements in the PhysioNet 2019 data, with units as documented by the Challenge
// (FiO2 is stored as a fraction after cleaning). Groups follow the order a clinician reads them.

export interface Variable {
  key: string;
  label: string;
  unit: string;
  digits: number;
}

export interface VariableGroup {
  id: string;
  title: string;
  kind: "vitals" | "labs";
  vars: Variable[];
}

const v = (key: string, label: string, unit: string, digits = 1): Variable => ({ key, label, unit, digits });

export const GROUPS: VariableGroup[] = [
  {
    id: "vitals",
    title: "Vital signs",
    kind: "vitals",
    vars: [
      v("HR", "Heart rate", "beats/min", 0),
      v("O2Sat", "Pulse oximetry (SpO₂)", "%", 0),
      v("Temp", "Temperature", "°C", 1),
      v("SBP", "Systolic blood pressure", "mm Hg", 0),
      v("MAP", "Mean arterial pressure", "mm Hg", 0),
      v("DBP", "Diastolic blood pressure", "mm Hg", 0),
      v("Resp", "Respiratory rate", "breaths/min", 0),
      v("EtCO2", "End-tidal CO₂", "mm Hg", 0),
    ],
  },
  {
    id: "gas",
    title: "Blood gas and ventilation",
    kind: "labs",
    vars: [
      v("pH", "pH", "", 2),
      v("PaCO2", "PaCO₂", "mm Hg", 0),
      v("SaO2", "Arterial O₂ saturation", "%", 0),
      v("BaseExcess", "Base excess", "mmol/L", 0),
      v("HCO3", "Bicarbonate (HCO₃)", "mmol/L", 0),
      v("FiO2", "Fraction of inspired O₂", "fraction", 2),
    ],
  },
  {
    id: "perfusion",
    title: "Perfusion and cardiac",
    kind: "labs",
    vars: [v("Lactate", "Lactate", "mg/dL", 1), v("TroponinI", "Troponin I", "ng/mL", 2)],
  },
  {
    id: "chemistry",
    title: "Chemistry and kidney",
    kind: "labs",
    vars: [
      v("BUN", "Blood urea nitrogen", "mg/dL", 0),
      v("Creatinine", "Creatinine", "mg/dL", 2),
      v("Glucose", "Glucose", "mg/dL", 0),
      v("Potassium", "Potassium", "mmol/L", 1),
      v("Chloride", "Chloride", "mmol/L", 0),
      v("Calcium", "Calcium", "mg/dL", 1),
      v("Magnesium", "Magnesium", "mmol/dL", 1),
      v("Phosphate", "Phosphate", "mg/dL", 1),
    ],
  },
  {
    id: "liver",
    title: "Liver",
    kind: "labs",
    vars: [
      v("AST", "Aspartate transaminase (AST)", "IU/L", 0),
      v("Alkalinephos", "Alkaline phosphatase", "IU/L", 0),
      v("Bilirubin_total", "Total bilirubin", "mg/dL", 1),
      v("Bilirubin_direct", "Direct bilirubin", "mg/dL", 1),
    ],
  },
  {
    id: "blood",
    title: "Blood count and clotting",
    kind: "labs",
    vars: [
      v("WBC", "White cell count", "×10³/µL", 1),
      v("Hgb", "Haemoglobin", "g/dL", 1),
      v("Hct", "Haematocrit", "%", 0),
      v("Platelets", "Platelets", "×10³/µL", 0),
      v("PTT", "Partial thromboplastin time", "s", 1),
      v("Fibrinogen", "Fibrinogen", "mg/dL", 0),
    ],
  },
];

export const ALL_VARIABLES: Variable[] = GROUPS.flatMap((g) => g.vars);
export const VARIABLE: Record<string, Variable> = Object.fromEntries(ALL_VARIABLES.map((x) => [x.key, x]));

export const fmtValue = (x: number | null | undefined, digits: number) => (x == null ? "—" : x.toFixed(digits));
