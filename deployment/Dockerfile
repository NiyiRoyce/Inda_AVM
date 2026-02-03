# ==================================================
# Base image: Python 3.11 slim
# ==================================================
FROM python:3.11-slim

# ==================================================
# System setup
# ==================================================
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ==================================================
# Python dependencies
# ==================================================
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================================================
# Application code
# ==================================================
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY artifacts/ ./artifacts/
COPY config/ ./config/

# ==================================================
# Environment variables
# ==================================================
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV AIP_STORAGE_URI=/app/artifacts

# ==================================================
# Expose port
# ==================================================
EXPOSE 8080

# ==================================================
# Production CMD for Vertex AI
# ==================================================
CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--proxy-headers", \
     "--timeout-keep-alive", "75"]
