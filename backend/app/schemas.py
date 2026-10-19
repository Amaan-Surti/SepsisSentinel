"""Pydantic v2 request/response models for the SepsisSentinel API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from sepsis.constants import RAW_COLUMNS

Level = Literal["ALERT", "WATCH", "OK"]


class Health(BaseModel):
    status: str
    model_version: str


class Config(BaseModel):
    threshold: float
    k: int
    sticky: int
    max_hour: int
    n_patients: int
    model_version: str
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
    hours: List[int]
    risk: List[float]
    alert: List[bool]
    reasons_last_hour: List[Reason]


class Metrics(BaseModel):
    test_metrics: List[Dict[str, Any]]
    baselines: List[Dict[str, Any]]
    cross_hospital: List[Dict[str, Any]]
    ablations: List[Dict[str, Any]]
    subgroups: List[Dict[str, Any]]
    dev_oof: List[Dict[str, Any]]
    threshold_curve: List[Dict[str, Any]]
    policy: Dict[str, Any]


class ModelCard(BaseModel):
    title: str
    model_version: str
    markdown: str
    sections: Dict[str, str]
