# Imagen del servicio de inferencia BarkVision.
# Build (desde la raíz del repo, que es lo que hace Jenkins):
#   docker build -t mi-app-web .
# Run local:
#   docker run -p 80:8080 mi-app-web
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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código del servicio.
COPY app.py model_loader.py ./

# Modelos horneados en la imagen (el deploy por SSH no monta volúmenes).
# model_loader.py lee desde MODELS_DIR=/models.
COPY models/ /models/

EXPOSE 8080

# El contenedor escucha en $PORT (8080). Publicá con -p 80:8080.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
