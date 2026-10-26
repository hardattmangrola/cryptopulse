# Data Ingestion Microservice Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

# Install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries 10 --no-cache-dir -r requirements.txt

# Copy shared utilities and ingestion service code
COPY shared/ /app/shared/
COPY services/ /app/services/

EXPOSE 8001

CMD ["uvicorn", "services.ingestion.app:app", "--host", "0.0.0.0", "--port", "8001"]
