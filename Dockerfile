FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow / TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app/ app/
COPY models/ models/

# logs/ is created at runtime by app.py, but declare a volume mount point
RUN mkdir -p logs

EXPOSE 8501

ENV MODEL_VERSION=v1

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
