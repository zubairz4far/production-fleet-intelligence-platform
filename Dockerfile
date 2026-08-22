FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ETA_MODEL_PATH=/models/eta_hist_gradient_boosting.joblib

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "fleet_intelligence.serving:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
