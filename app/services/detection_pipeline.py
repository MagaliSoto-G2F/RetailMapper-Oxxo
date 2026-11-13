import time
import logging
from typing import Dict, List, Tuple

import numpy as np

from app.config import PRODUCT_CONF_THRESHOLD, EMPTY_SPACE_MIN_WIDTH_RATIO
from app.core.model_loader import get_models
from app.localization.assign_row import assign_rows
from app.localization.assign_column import assign_columns
from app.localization.find_empty_spaces import find_empty_spaces

logger = logging.getLogger(__name__)


def detect_shelf_image(img: np.ndarray) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Detecta repisas y productos en una imagen de góndola y ubica cada producto
    por fila (nivel de repisa, abajo = 1) y columna (izquierda = 1).
    Además detecta los espacios vacíos de cada fila.

    Retorna
    -------
    Tuple[List[Dict], List[Dict], List[Dict]]
        - shelves : lista de {'bbox', 'conf'}
        - products : lista de {'id', 'bbox', 'conf', 'row', 'col'}
        - empty_spaces : lista de {'id', 'bbox', 'row', 'after_col', 'width_px', 'estimated_slots'}
    """
    start = time.perf_counter()

    shelf_detector, product_detector = get_models()

    shelves = shelf_detector.detect(img)
    products = product_detector.detect(img, conf_threshold=PRODUCT_CONF_THRESHOLD)

    products = assign_rows(products, shelves)
    products = assign_columns(products)

    for i, p in enumerate(products, start=1):
        p["id"] = i

    empty_spaces = find_empty_spaces(
        products, shelves, min_width_ratio=EMPTY_SPACE_MIN_WIDTH_RATIO
    )

    logger.info(
        f"Detection finished in {time.perf_counter() - start:.3f}s | "
        f"Shelves: {len(shelves)} | Products: {len(products)} | "
        f"Empty spaces: {len(empty_spaces)}"
    )

    return shelves, products, empty_spaces
