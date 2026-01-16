FROM python:3.10-slim

# ==================================================
# System setup
# ==================================================
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==================================================
# Python dependencies
# ==================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ==================================================
# Application code
# ==================================================
COPY . .

# ==================================================
# Environment variables
# ==================================================
ENV PYTHONUNBUFFERED=1
ENV AIP_STORAGE_URI=/app/artifacts
ENV PORT=8080

# IMPORTANT:
# Do NOT set GOOGLE_APPLICATION_CREDENTIALS
# Vertex AI injects ADC automatically

# ==================================================
# Runtime
# ==================================================
EXPOSE 8080

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--proxy-headers", \
     "--timeout-keep-alive", "75"]
