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
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=8 \
    GUNICORN_TIMEOUT=180 \
    # default session dir to a writable location (can be overridden by App Settings)
    SESSION_FILE_DIR=/tmp/flask_session

# Install small runtime libs commonly required by numeric libs & certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -u 1000 -m nonroot

# Create directories the app expects and make them writable by the nonroot user
RUN mkdir -p /app/flask_session /tmp/flask_session /app/static/images \
 && chown -R nonroot:nonroot /app /tmp/flask_session

# Bring installed packages from builder
COPY --from=builder --chown=nonroot:nonroot /install /install

# Copy app source with nonroot ownership
COPY --chown=nonroot:nonroot . ./

EXPOSE 5000
USER nonroot

# Run with Gunicorn; app.py defines "app"
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:5000 app:app --workers=${GUNICORN_WORKERS} --threads=${GUNICORN_THREADS} --timeout=${GUNICORN_TIMEOUT} --access-logfile=- --error-logfile=- --log-level=info"]