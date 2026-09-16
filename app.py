"""API de inferencia de BarkVision (FastAPI).

Endpoints:
    GET  /health              -> liveness + versión activa
    GET  /versions            -> versiones registradas y cuál está activa
    GET  /model?version=v1    -> info de una versión (input, clases, metadata)
    POST /predict             -> sube una imagen, devuelve % por clase + overlay PNG

Correr local:   uvicorn app:app --reload   (desde serving/)
En contenedor:  ver serving/Dockerfile
"""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

import model_loader

# Paleta por id de clase. Debe coincidir con scripts/predict.py y las máscaras:
# 0 fondo, 1 corteza_adherida, 2 corteza_suelta, 3 madera_expuesta.
PALETTE = np.array([[20, 20, 20], [220, 50, 50], [50, 200, 80], [60, 120, 230]],
                   dtype=np.uint8)

# Imágenes por forward pass en /predict_batch. Acota el pico de memoria (clave en
# CPU / Cloud Run). Subilo si tenés RAM de sobra. Override con la env MAX_BATCH.
MAX_BATCH = int(os.environ.get("MAX_BATCH", "4"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precargamos la versión activa al arrancar, para que el primer /predict no
    # pague la latencia de cargar el modelo.
    try:
        entry = model_loader.get_model()
        print(f"[startup] modelo activo cargado: {entry['version']} "
              f"(input {entry['input_size']})")
    except Exception as e:  # noqa: BLE001
        print(f"[startup] aviso: no se pudo precargar un modelo activo: {e}")
    yield


app = FastAPI(title="BarkVision Serving", version="1.0.0", lifespan=lifespan)


def _preprocess(image: Image.Image, size: tuple[int, int]):
    """Resize + /255 -> batch (1,H,W,3). MISMO preprocesado que el entrenamiento
    (shared/preprocessing.normalize_image). Devuelve también la imagen redimensionada."""
    h, w = size
    resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...], np.asarray(resized)


def _overlay_png_b64(base_rgb: np.ndarray, mask: np.ndarray) -> str:
    """Superpone la máscara coloreada sobre la imagen y la devuelve como PNG base64."""
    color = PALETTE[mask]
    blended = base_rgb.astype(np.float32).copy()
    fg = mask > 0
    blended[fg] = 0.5 * base_rgb[fg] + 0.5 * color[fg]
    buf = io.BytesIO()
    Image.fromarray(blended.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _class_stats(mask: np.ndarray, names: list[str]) -> list[dict]:
    """Cuenta píxeles y porcentaje por clase a partir de una máscara (H,W)."""
    counts = np.bincount(mask.reshape(-1), minlength=len(names))
    total = int(counts.sum())
    return [{"id": i, "name": names[i], "pixels": int(counts[i]),
             "percent": round(100.0 * counts[i] / total, 2)}
            for i in range(len(names))]


@app.get("/health")
def health():
    try:
        reg = model_loader.load_registry()
        return {"status": "ok", "active_version": reg.get("active")}
    except Exception as e:  # noqa: BLE001
        return {"status": "no_model", "detail": str(e)}


@app.get("/versions")
def versions():
    reg = model_loader.load_registry()
    return {"active": reg.get("active"), "versions": reg.get("versions", {})}


@app.get("/model")
def model_info(version: str | None = Query(default=None)):
    try:
        entry = model_loader.get_model(version)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"version": entry["version"], "input_size": entry["input_size"],
            "class_names": entry["class_names"], "metadata": entry["metadata"]}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    version: str | None = Query(default=None),
    include_overlay: bool = Query(default=True),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Subí una imagen JPEG o PNG.")
    try:
        entry = model_loader.get_model(version)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")

    batch, resized = _preprocess(image, entry["input_size"])
    probs = np.asarray(entry["model"].predict(batch, verbose=0))[0]  # (H,W,num_classes)
    mask = np.argmax(probs, axis=-1).astype(np.uint8)

    names = entry["class_names"] or [str(i) for i in range(int(mask.max()) + 1)]
    classes = _class_stats(mask, names)

    resp = {"version": entry["version"], "filename": file.filename,
            "input_size": list(entry["input_size"]), "classes": classes}
    if include_overlay:
        resp["overlay_png_base64"] = _overlay_png_b64(resized, mask)
    return resp


@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    version: str | None = Query(default=None),
    include_overlay: bool = Query(default=False),
):
    """Procesa varias imágenes en UNA request (ej. un 'análisis' = ~20 fotos).

    Devuelve el resultado por imagen y un AGREGADO sobre todas (la composición
    total de clases del análisis). Los overlays vienen solo si include_overlay=true
    (por defecto false: 20 PNGs en base64 harían la respuesta enorme).
    """
    if not files:
        raise HTTPException(status_code=422, detail="Subí al menos una imagen.")
    try:
        entry = model_loader.get_model(version)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    size = entry["input_size"]

    imgs, resized_list, results, valid_idx = [], [], [], []
    for f in files:
        item = {"filename": f.filename}
        if f.content_type not in ("image/jpeg", "image/png"):
            item["error"] = f"tipo no soportado: {f.content_type}"
            results.append(item)
            continue
        raw = await f.read()
        try:
            image = Image.open(io.BytesIO(raw))
            batch, resized = _preprocess(image, size)
        except Exception as e:  # noqa: BLE001
            item["error"] = f"imagen inválida: {e}"
            results.append(item)
            continue
        valid_idx.append(len(results))
        imgs.append(batch[0])
        resized_list.append(resized)
        results.append(item)

    if not imgs:
        raise HTTPException(status_code=400, detail="Ninguna imagen válida en la request.")

    # Inferencia en lotes (batch_size acota el pico de memoria).
    arr = np.stack(imgs, axis=0)  # (N, H, W, 3)
    probs = np.asarray(entry["model"].predict(arr, batch_size=MAX_BATCH, verbose=0))
    masks = np.argmax(probs, axis=-1).astype(np.uint8)  # (N, H, W)

    names = entry["class_names"] or [str(i) for i in range(int(masks.max()) + 1)]
    agg = np.zeros(len(names), dtype=np.int64)
    for k, ridx in enumerate(valid_idx):
        m = masks[k]
        results[ridx]["classes"] = _class_stats(m, names)
        agg += np.bincount(m.reshape(-1), minlength=len(names))
        if include_overlay:
            results[ridx]["overlay_png_base64"] = _overlay_png_b64(resized_list[k], m)

    total = int(agg.sum())
    aggregate = {
        "total_pixels": total,
        "classes": [{"id": i, "name": names[i], "pixels": int(agg[i]),
                     "percent": round(100.0 * agg[i] / total, 2)}
                    for i in range(len(names))],
    }
    return {"version": entry["version"], "n_images": len(files),
            "n_processed": len(imgs), "aggregate": aggregate, "per_image": results}


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
