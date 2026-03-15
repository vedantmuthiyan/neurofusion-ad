# NeuroFusion-AD FHIR API — Multi-stage Dockerfile
#
# Stage 1: builder — installs Python deps into a venv
# Stage 2: runtime — copies venv, runs as non-root user
#
# Build:   docker build -t neurofusion-ad:latest .
# Run:     docker run -p 8000:8000 --env-file .env neurofusion-ad:latest
#
# IEC 62304 traceability: SAD-001 § 5.4

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .

# Create venv and install CPU-only PyTorch + deps
# Note: GPU inference uses CUDA_VISIBLE_DEVICES at runtime (host driver)
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir \
        torch-geometric==2.5.0 && \
    /opt/venv/bin/pip install --no-cache-dir \
        -r requirements.txt \
        httpx \
        asyncpg \
        pytest-asyncio

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/

# Create model and data directories (populated at runtime via volume mounts)
RUN mkdir -p models/final data/processed/adni logs && \
    chown -R appuser:appgroup /app

USER appuser

# Environment defaults (override via env file or -e flags)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/final/best_model.pth \
    SCALER_PATH=/app/data/processed/adni/scaler.pkl \
    TEMPERATURE=0.756 \
    MC_SAMPLES=30 \
    DATABASE_URL=postgresql://neurofusion:neurofusion@db:5432/neurofusion

EXPOSE 8000

# Health check (liveness probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
