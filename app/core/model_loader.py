# app/core/model_loader.py

import os
import shutil
from threading import Lock
from app.detectors.shelf_detector import ShelfDetector
from app.detectors.product_detector import ProductDetector
from app.utils.hf_utils import download_model
from app.config import SHELF_CONF_THRESHOLD


_shelf_detector = None
_product_detector = None
_lock = Lock()


def _ensure_model_in_local_folder(
    cached_path: str,
    local_path: str
) -> str:
    """
    Asegura que el archivo del modelo exista en la ruta local destino.
    Si no existe, lo copia desde la caché de Hugging Face.

    Parámetros
    ----------
    cached_path : str
        Ruta devuelta por la descarga en caché de Hugging Face.
    local_path : str
        Ruta local donde se quiere guardar el modelo.

    Retorna
    -------
    str
        Ruta local final donde quedó guardado el modelo.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        shutil.copyfile(cached_path, local_path)

    return local_path


def load_models() -> tuple[ShelfDetector, ProductDetector]:
    """
    Carga los modelos de detección de repisas y productos (patrón singleton).
    Los modelos se descargan si hace falta y se inicializan una sola vez.

    Retorna
    -------
    tuple[ShelfDetector, ProductDetector]
        Instancias inicializadas de los detectores de repisas y productos.
    """
    global _shelf_detector, _product_detector

    with _lock:

        if _shelf_detector is None:
            cached_shelf_path = download_model(
                repo_id="MagaliSoto/RetailMapper",
                filename="shelf-model.pt",
                local_path="models/shelf-model.pt"
            )

            final_shelf_path = _ensure_model_in_local_folder(
                cached_shelf_path,
                "models/shelf-model.pt"
            )

            _shelf_detector = ShelfDetector(final_shelf_path, SHELF_CONF_THRESHOLD)

        if _product_detector is None:
            cached_product_path = download_model(
                repo_id="MagaliSoto/RetailMapper",
                filename="product-model.pt",
                local_path="models/product-model.pt"
            )

            final_product_path = _ensure_model_in_local_folder(
                cached_product_path,
                "models/product-model.pt"
            )

            _product_detector = ProductDetector(final_product_path)

    return _shelf_detector, _product_detector


def get_models() -> tuple[ShelfDetector, ProductDetector]:
    """
    Devuelve las instancias de los modelos ya cargados.

    Retorna
    -------
    tuple[ShelfDetector, ProductDetector]
        Instancias de los detectores de repisas y productos.

    Lanza
    -----
    RuntimeError
        Si los modelos no se cargaron antes de llamar a esta función.
    """
    if _shelf_detector is None or _product_detector is None:
        raise RuntimeError("Models not loaded. Did startup run?")

    return _shelf_detector, _product_detector
