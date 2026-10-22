# Training Microservice Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities, models directory, and training service code
COPY shared/ /app/shared/
COPY models/ /app/models/
COPY services/ /app/services/

EXPOSE 8003

CMD ["uvicorn", "services.training.app:app", "--host", "0.0.0.0", "--port", "8003"]
