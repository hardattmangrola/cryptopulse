# Monitoring & Continuous Maintenance Microservice Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities, models, and monitoring service code
COPY shared/ /app/shared/
COPY services/ /app/services/
COPY models/ /app/models/

EXPOSE 8004

CMD ["uvicorn", "services.monitoring.app:app", "--host", "0.0.0.0", "--port", "8004"]
