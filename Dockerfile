# Single image used for the FastAPI backend, the Celery worker, and the
# Streamlit frontend, exactly as shown in the architecture diagram
# ("Local Registry localhost:5001 -> rag-app:fixed" fanning out to all three).
# Which process runs is decided by the `command:` in each Kubernetes Deployment.
FROM python:3.11-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY app /code/app

EXPOSE 8000 8501

# Default command (overridden per-Deployment in k8s/*.yaml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
