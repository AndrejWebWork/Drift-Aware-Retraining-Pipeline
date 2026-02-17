FROM python:3.9-slim

# Metadata
LABEL maintainer="drift-pipeline"
LABEL description="Drift-Aware Retraining Pipeline - Production-Ready MLOps"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY drift_pipeline/ ./drift_pipeline/

# Create directories for artifacts
RUN mkdir -p /app/model_registry /app/logs /app/metrics

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIPELINE_REGISTRY_PATH=/app/model_registry
ENV PIPELINE_LOG_PATH=/app/logs

# Expose ports for monitoring
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import drift_detection; print('healthy')" || exit 1

# Run pipeline
CMD ["python", "-m", "drift_pipeline.example_usage"]
