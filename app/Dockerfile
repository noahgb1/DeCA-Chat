# syntax=docker/dockerfile:1.6
# Build from app\ with:
#   $env:DOCKER_BUILDKIT="1"
#   docker build -t decachatacr.azurecr.us/decachat:v0.215.37.n1 .

#############################
# Builder: deps in a venv   #
#############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Layer-cached requirements
COPY requirements.txt ./requirements.txt

# Heavy deps from wheels only (fast & reliable)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install --only-binary=:all: \
        numpy \
        pandas==2.2.3 \
        pyarrow==16.* \
        duckdb==1.0.*

# Then the rest of your requirements (handles libs without wheels)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

#############################
# Runtime: minimal, non-root #
#############################
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime libs + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser

WORKDIR /app

# Copy venv and app code
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

# Healthcheck using curl (NO python here-doc)
# If you don't have /healthz yet, comment the next two lines
#HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  #CMD curl -fsS http://127.0.0.1:5000/healthz || exit 1

EXPOSE 5000
ENTRYPOINT ["python", "/app/app.py"]
