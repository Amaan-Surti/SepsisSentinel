# SepsisSentinel API. Build from the repo root:  docker build -f backend/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MODEL_KIND=ensemble

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime-only subset of requirements.txt (same pins; no training/notebook packages)
COPY backend/requirements.txt backend/requirements.txt
# Same layer: drop build-only parts of torch (C++ headers, protoc) and bundled test suites
RUN pip install -r backend/requirements.txt \
    && cd /usr/local/lib/python3.11/site-packages \
    && rm -rf torch/include torch/bin/protoc* \
    && find . -depth -type d -name tests -exec rm -rf {} +

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps -e .

COPY configs/ configs/
COPY backend/ backend/
COPY models/final/ models/final/
COPY models/ensemble_final/ models/ensemble_final/
COPY data/demo/ data/demo/
COPY reports/tables/ reports/tables/
COPY reports/figures/ reports/figures/
COPY reports/model_card.md reports/model_card.md

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
