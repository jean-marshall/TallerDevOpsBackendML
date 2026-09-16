# Imagen del servicio de inferencia BarkVision.
# Build (desde la carpeta ml/):
#   docker build -f serving/Dockerfile -t barkvision-serving .
# Run local (montando el registro de modelos como volumen):
#   docker run -p 8080:8080 -v "%cd%/serving/models:/models" barkvision-serving   (PowerShell: ${PWD})
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/models \
    PORT=8080

WORKDIR /app

# libgomp1: requerido por TensorFlow (OpenMP) en la imagen slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Deps primero, para aprovechar la caché de capas de Docker.
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código del servicio.
COPY serving/app.py serving/model_loader.py ./

EXPOSE 8080

# Cloud Run inyecta el puerto en $PORT; localmente cae a 8080.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
