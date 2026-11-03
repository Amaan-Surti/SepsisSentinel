"""Pydantic v2 request/response models for the SepsisSentinel API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from sepsis.constants import RAW_COLUMNS

Level = Literal["ALERT", "WATCH", "OK"]
ModelKind = Literal["ensemble", "lgbm"]


class Health(BaseModel):
    status: str
    model_kind: ModelKind
    model_version: str


class Config(BaseModel):
    threshold: float
    k: int
    sticky: int
    max_hour: int
    n_patients: int
    model_kind: ModelKind
    model_label: str
    model_version: str
    weight_lgbm: Optional[float]
    reasons_source: str
    disclaimer: str


class Reason(BaseModel):
    text: str
    feature: str
    value: Optional[float]
    shap: float


class WardBed(BaseModel):
    patient_id: str
    display_name: str
    age: int
    gender: int
    unit: int
    hour: int
    status: Literal["active", "discharged"]
    risk: float
    risk_trend: List[float]
    alert: bool
    alert_since: Optional[int]
    level: Level
    vitals: Dict[str, Optional[float]]


class PatientDetail(BaseModel):
    patient_id: str
    display_name: str
    age: int
    gender: int
    unit: int
    stay_hours: int
    hour: int
    status: Literal["active", "discharged"]
    level: Level
    alert_since: Optional[int]
    threshold: float
    hours: List[int]
    vitals: Dict[str, List[Optional[float]]]
    labs: Dict[str, List[Optional[float]]]
    risk: List[float]
    alert: List[bool]
    reasons: List[Reason]
    onset_hour: Optional[int] = None
    septic: Optional[bool] = None
    outcome: Optional[str] = None


RawRow = create_model(
    "RawRow",
    __config__=ConfigDict(extra="forbid"),
    hour=(int, Field(ge=0, le=1000)),
    **{c: (Optional[float], None) for c in RAW_COLUMNS},
)


class PredictRequest(BaseModel):
    rows: List[RawRow] = Field(min_length=1, max_length=2000)  # type: ignore[valid-type]


class PredictResponse(BaseModel):
    model_kind: ModelKind
    hours: List[int]
    risk: List[float]
    alert: List[bool]
    reasons_last_hour: List[Reason]
    reasons_source: str


class LockboxTest(BaseModel):
    """Frozen LightGBM on the 8,068 lockbox patients (evaluated once)."""

    label: str
    model: str
    note: str
    metrics: List[Dict[str, Any]]
    baselines: List[Dict[str, Any]]
    subgroups: List[Dict[str, Any]]
    policy: Dict[str, Any]


class DevCV(BaseModel):
    """LightGBM vs GRU vs ensemble on development data, nested 5-fold CV."""

    label: str
    note: str
    comparison: List[Dict[str, Any]]
    weights: List[Dict[str, Any]]
    bootstrap: List[Dict[str, Any]]
    final: List[Dict[str, Any]]
    selection: Dict[str, Any]


class Metrics(BaseModel):
    active_model: ModelKind
    lockbox_test: LockboxTest
    dev_cv: Optional[DevCV]
    cross_hospital: List[Dict[str, Any]]
    ablations: List[Dict[str, Any]]
    dev_oof: List[Dict[str, Any]]
    threshold_curve: List[Dict[str, Any]]
    policy: Dict[str, Any]


class ModelCard(BaseModel):
    title: str
    model_kind: ModelKind
    model_version: str
    markdown: str
    sections: Dict[str, str]
