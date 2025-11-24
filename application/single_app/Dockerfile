# ----------------------------------------
# Builder stage (context is application/single_app)
# ----------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from current directory
COPY requirements.txt ./requirements.txt

# Install gunicorn and all deps into /install
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir gunicorn \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt \
    && find /install -type f -name '*.pyc' -delete

# ----------------------------------------
# Runtime stage
# ----------------------------------------
FROM python:3.11-slim
WORKDIR /app

# Ensure Python can import packages from /install and pick up /install/bin scripts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.11/site-packages" \
    GUNICORN_WORKERS=3 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=180

# Non-root user
RUN useradd -u 1000 -m nonroot

# Bring installed packages from builder
COPY --from=builder /install /install

# Copy app source (this puts app.py at /app/app.py)
COPY . ./

EXPOSE 5000
USER nonroot

# Optional healthcheck (uncomment after adding a /health route in app.py)
# HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# Run with Gunicorn; app.py defines "app"
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:5000 app:app --workers=${GUNICORN_WORKERS} --threads=${GUNICORN_THREADS} --timeout=${GUNICORN_TIMEOUT} --access-logfile=- --error-logfile=- --log-level=info"]