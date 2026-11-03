"""SepsisSentinel FastAPI app. Run: uvicorn backend.app.main:app --reload --port 8000

MODEL_KIND=ensemble (default) or MODEL_KIND=lgbm selects the served model.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.schemas import (
    Config,
    Health,
    Metrics,
    ModelCard,
    PatientDetail,
    PredictRequest,
    PredictResponse,
    WardBed,
)
from backend.app.service import PatientNotFound, SepsisService
from backend.app.settings import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

FIGURES = {
    "fig_calibration_test.png",
    "fig_roc_pr_test.png",
    "fig_lead_time_hist_test.png",
    "fig_cross_hospital.png",
    "fig_policy_tradeoff.png",
    "fig_shap_bar_top20.png",
}


def get_service(request: Request) -> SepsisService:
    return request.app.state.service


ThresholdQ = Query(None, gt=0.0, lt=1.0, description="Alert threshold (default: policy)")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the API for one served model (settings default to the environment)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = SepsisService(settings)
        yield

    app = FastAPI(title="SepsisSentinel API", version="1.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=Health)
    def health(svc: SepsisService = Depends(get_service)):
        return svc.health()

    @app.get("/api/config", response_model=Config)
    def config(svc: SepsisService = Depends(get_service)):
        return svc.config()

    @app.get("/api/ward", response_model=List[WardBed])
    def ward(
        t: int = Query(0, ge=0),
        threshold: Optional[float] = ThresholdQ,
        svc: SepsisService = Depends(get_service),
    ):
        return svc.ward_view(t, threshold)

    @app.get("/api/patients/{patient_id}", response_model=PatientDetail)
    def patient(
        patient_id: str,
        t: int = Query(0, ge=0),
        threshold: Optional[float] = ThresholdQ,
        show_truth: bool = False,
        svc: SepsisService = Depends(get_service),
    ):
        try:
            return svc.patient_view(patient_id, t, threshold, show_truth)
        except PatientNotFound:
            raise HTTPException(status_code=404, detail=f"Unknown patient {patient_id}")

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, svc: SepsisService = Depends(get_service)):
        try:
            return svc.predict_rows([r.model_dump() for r in req.rows])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.get("/api/metrics", response_model=Metrics)
    def metrics(svc: SepsisService = Depends(get_service)):
        return svc.metrics()

    @app.get("/api/model-card", response_model=ModelCard)
    def model_card(svc: SepsisService = Depends(get_service)):
        return svc.model_card()

    @app.get("/api/figures/{name}")
    def figure(name: str):
        path = settings.figures_dir / name
        if name not in FIGURES or not path.exists():
            raise HTTPException(status_code=404, detail=f"Unknown figure {name}")
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()
