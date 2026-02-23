# ─────────────────────────────────────────────────────
# Stage 1: Base image
# ─────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="Desmond Onam"
LABEL description="A/B Testing MLOps Dashboard — Streamlit + MLflow"

# System deps
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────
# Stage 2: Python deps
# ─────────────────────────────────────────────────────
FROM base AS dependencies

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────
# Stage 3: Application
# ─────────────────────────────────────────────────────
FROM dependencies AS app

WORKDIR /app

# Copy project files
COPY . .

# Create directories
RUN mkdir -p models mlruns data outputs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]