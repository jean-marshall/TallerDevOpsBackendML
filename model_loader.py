"""Carga y cachea modelos versionados para el servicio de inferencia.

El "registro" de modelos vive en MODELS_DIR (por defecto serving/models/), con un
registry.json que dice cuál es la versión activa y qué versiones hay. Cada versión
es una carpeta con `model.keras` (o `model/` SavedModel) + `metadata.json`.

En Cloud Run: montá/horneá los modelos y seteá MODELS_DIR (ej. /models).
"""
from __future__ import annotations

import functools
import json
import os

import keras

MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(
            f"No hay registro de modelos en {REGISTRY_PATH}. "
            "Registrá uno con: python scripts/register_model.py --keras <...> --version v1"
        )
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_version(version: str | None = None) -> str:
    """Devuelve la versión pedida, o la activa si version es None. Valida que exista."""
    reg = load_registry()
    v = version or reg.get("active")
    if v not in reg.get("versions", {}):
        raise KeyError(
            f"Versión '{v}' no existe. Disponibles: {list(reg.get('versions', {}))}"
        )
    return v


@functools.lru_cache(maxsize=8)
def _load(version: str) -> dict:
    """Carga (y cachea) el modelo de una versión ya resuelta."""
    meta = load_registry()["versions"][version]
    vdir = os.path.join(MODELS_DIR, version)
    keras_path = os.path.join(vdir, "model.keras")
    sm_path = os.path.join(vdir, "model")

    if os.path.exists(keras_path):
        # compile=False: solo forward pass; no necesitamos la loss de entrenamiento.
        model = keras.models.load_model(keras_path, compile=False)
    elif os.path.isdir(sm_path):
        # SavedModel: se envuelve en un Model para poder usar .predict().
        h, w = meta["image_size"]
        inp = keras.Input((h, w, 3))
        out = keras.layers.TFSMLayer(sm_path, call_endpoint="serve")(inp)
        if isinstance(out, dict):
            out = list(out.values())[0]
        model = keras.Model(inp, out)
    else:
        raise FileNotFoundError(f"No hay model.keras ni model/ en {vdir}")

    ishape = model.input_shape  # (None, H, W, 3)
    input_size = (int(ishape[1]), int(ishape[2])) if ishape[1] else tuple(meta["image_size"])
    return {"model": model, "metadata": meta, "input_size": input_size,
            "class_names": meta.get("class_names")}


def get_model(version: str | None = None) -> dict:
    """Devuelve dict con el modelo, su metadata, input_size y class_names.

    Cachea por versión, así múltiples requests reusan el modelo ya cargado.
    """
    version = resolve_version(version)
    return {"version": version, **_load(version)}
