# Backend image — FastAPI + TensorFlow inference server
FROM python:3.11-slim

WORKDIR /app

# System deps for scientific Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY backend/ ./backend/

# data/, artifacts/, and the trained .keras model are mounted at runtime
# (see docker-compose.yml) rather than baked into the image — they're large,
# user-specific, and gitignored for the same reason.
ENV TURBOFAN_DATA_PATH=/app/data/
ENV TURBOFAN_MODEL_PATH=/app/turbofan_rul_v4.keras
ENV TURBOFAN_ARTIFACTS_PATH=/app/artifacts/domain_adaptation.joblib
ENV TURBOFAN_CORS_ORIGINS=http://localhost:5173,http://localhost:3000

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
