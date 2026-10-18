"""Constants and column definitions for SepsisSentinel."""

RAW_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    "SepsisLabel",
]

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
MEASURED = VITALS + LABS  # 34 time-varying measurements
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
# VITALS_ROLL excludes EtCO2 (almost always missing)
VITALS_ROLL = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS_ROLL = [
    "Lactate", "WBC", "Creatinine", "Platelets", "BUN",
    "Bilirubin_total", "pH", "HCO3",
]
BASELINE_VARS = VITALS_ROLL + ["Lactate", "WBC", "Creatinine", "Platelets"]
SEED = 42
