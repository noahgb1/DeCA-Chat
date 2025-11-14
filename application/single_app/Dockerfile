# ----------------------------------------
# Builder stage: install dependencies
# ----------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (add more if certain wheels need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY application/single_app/requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt \
    && find /install -type f -name '*.pyc' -delete

# ----------------------------------------
# Runtime stage
# ----------------------------------------
FROM python:3.11-slim
WORKDIR /app

# Environment defaults (override at deploy)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    GUNICORN_WORKERS=3 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=180

# Create non-root user
RUN useradd -u 1000 -m nonroot

# Copy installed packages
COPY --from=builder /install /install

# Copy application code (your app.py ends up at /app/app.py)
COPY application/single_app/ ./ 

# Expose port
EXPOSE 5000

USER nonroot

# Optional healthcheck (uncomment after adding /health route)
# HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# Gunicorn command:
# - app:app because /app/app.py defines "app"
# - Adjust workers/threads/timeouts via env vars
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:5000 app:app --workers=${GUNICORN_WORKERS} --threads=${GUNICORN_THREADS} --timeout=${GUNICORN_TIMEOUT} --access-logfile=- --error-logfile=- --log-level=info"]
