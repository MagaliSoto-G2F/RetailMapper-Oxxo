# app/localization/verify_empty_spaces.py

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Un píxel es "estructura" (parante metálico negro) si su gris es menor a esto
_DARK_GRAY = 70
# Una columna es parante si al menos esta fracción de sus píxeles es oscura
_UPRIGHT_DARK_FRACTION = 0.75
# Hay frente de estante en una columna si su borde horizontal en la base de
# la fila es al menos esta fracción del que hay bajo el producto vecino.
# Calibrado: estante 0.26–1.6, piso / pasillo fuera de la góndola 0.03–0.05.
_SHELF_FRONT_MIN_RATIO = 0.12


def _dark_column_profile(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Fracción de píxeles oscuros por columna dentro de bbox (suavizada)."""
    x1, y1, x2, y2 = map(int, bbox)
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    frac = (gray < _DARK_GRAY).mean(axis=0)
    return np.convolve(frac, np.ones(5) / 5, mode="same")


def trim_to_uprights(img: np.ndarray, space: Dict) -> Optional[Tuple[int, int, int, int]]:
    """
    Recorta un hueco que toca el borde de la fila hasta el primer parante de
    la góndola, buscando desde el producto hacia afuera.

    La bbox de la repisa suele cubrir los parantes, la pared o la góndola
    vecina; sin este recorte esas zonas aparecen como huecos falsos.

    Retorna la bbox recortada, o None si el hueco queda sin ancho.
    """
    x1, y1, x2, y2 = map(int, space["bbox"])
    open_left, open_right = space.get("_open_left"), space.get("_open_right")

    # Huecos entre dos productos o filas vacías: no se recortan
    if open_left == open_right:
        return (x1, y1, x2, y2)

    is_upright = _dark_column_profile(img, (x1, y1, x2, y2)) >= _UPRIGHT_DARK_FRACTION

    if open_left:
        # Producto a la derecha → buscar hacia la izquierda
        hits = np.flatnonzero(is_upright)
        if hits.size:
            x1 = x1 + int(hits.max()) + 1
    else:
        # Producto a la izquierda → buscar hacia la derecha
        hits = np.flatnonzero(is_upright)
        if hits.size:
            x2 = x1 + int(hits.min())

    return (x1, y1, x2, y2) if x2 > x1 else None


def _edges(img: np.ndarray, bbox: Tuple[int, int, int, int], margin_y: float = 0.1) -> np.ndarray:
    """Mapa de bordes Canny del recorte (sin un margen arriba y abajo)."""
    x1, y1, x2, y2 = map(int, bbox)
    my = int((y2 - y1) * margin_y)
    crop = img[y1 + my:y2 - my, x1:x2]
    if crop.size == 0:
        return np.zeros((1, max(0, x2 - x1)), dtype=np.uint8)
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


def _shelf_front_profile(img: np.ndarray, x1: int, x2: int, base_y: int, height: int) -> np.ndarray:
    """
    Intensidad del borde horizontal más fuerte por columna, en una franja
    alrededor de la base de la fila (donde está el frente del estante).
    """
    h = img.shape[0]
    y1 = max(0, int(base_y - 0.1 * height))
    y2 = min(h, int(base_y + 0.25 * height))
    gray = cv2.GaussianBlur(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return np.convolve(sobel_y.max(axis=0), np.ones(5) / 5, mode="same")


def trim_to_shelf_front(img: np.ndarray, space: Dict, bbox: Tuple[int, int, int, int]
                        ) -> Optional[Tuple[int, int, int, int]]:
    """
    Recorta un hueco que toca el borde de la fila hasta donde termina el
    frente del estante, buscando desde el producto hacia afuera.

    Donde hay estante, en la base de la fila aparece un borde horizontal
    fuerte (canto de la madera o la chapa). Fuera de la góndola (piso,
    pasillo, pared) no. A diferencia del recorte por parantes, no depende
    del color de la góndola. La referencia es el borde bajo el producto
    vecino, así se adapta a cada imagen.
    """
    x1, y1, x2, y2 = bbox
    open_left, open_right = space.get("_open_left"), space.get("_open_right")
    if open_left == open_right:
        return bbox

    w = img.shape[1]
    nw = int(space.get("_neighbor_width") or 0)
    if nw <= 0:
        return bbox
    # Columnas del producto vecino (referencia) + las del hueco
    if open_left:
        ref = (x2, min(w, x2 + nw))
    else:
        ref = (max(0, x1 - nw), x1)
    lo, hi = min(x1, ref[0]), max(x2, ref[1])
    profile = _shelf_front_profile(img, lo, hi, y2, y2 - y1)
    ref_strength = float(np.median(profile[ref[0] - lo:ref[1] - lo]))
    if ref_strength <= 0:
        return bbox

    has_front = profile[x1 - lo:x2 - lo] >= _SHELF_FRONT_MIN_RATIO * ref_strength
    no_front = np.flatnonzero(~has_front)
    if no_front.size == 0:
        return bbox
    if open_left:
        x1 = x1 + int(no_front.max()) + 1   # desde el producto (derecha) hacia la izquierda
    else:
        x2 = x1 + int(no_front.min())       # desde el producto (izquierda) hacia la derecha
    return (x1, y1, x2, y2) if x2 > x1 else None


def texture_stats(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
    """
    Mide qué tan "liso" es el recorte.

    El fondo de una góndola vacía (pared, panel trasero, tabla) tiene pocos
    bordes. Un producto tiene logos y texto, que dan muchos bordes.

    Retorna
    -------
    Dict
        - 'edge_density': fracción de píxeles que son borde según Canny (0–1)
        - 'color_std': desvío estándar medio de los canales BGR (informativo)
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {"edge_density": 0.0, "color_std": 0.0}
    edges = _edges(img, bbox)
    return {
        "edge_density": float(np.count_nonzero(edges)) / max(1, edges.size),
        "color_std": float(crop.reshape(-1, 3).std(axis=0).mean()),
    }


def _runs(free: np.ndarray) -> List[Tuple[int, int]]:
    """Tramos contiguos [inicio, fin) donde free es True."""
    padded = np.concatenate([[False], free, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def verify_empty_spaces(
    img: np.ndarray,
    candidates: List[Dict],
    low_conf_products: List[Dict],
    max_edge_density: float,
    debug: bool = False,
    rejected: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Confirma los huecos candidatos de `find_empty_spaces` usando la imagen.

    1. Los huecos en los extremos de la fila se recortan hasta el parante de
       la góndola y hasta donde termina el frente del estante. Si lo que
       queda no alcanza para otro producto como el vecino, se descartan.
    2. Cada columna del hueco se marca como ocupada si:
       - es un parante (columna casi negra). Separa huecos "interiores" que en
         realidad cruzan a la góndola vecina, o
       - la densidad de bordes en una ventana de medio producto centrada en
         ella supera `max_edge_density` (logos/texto: producto no detectado), o
       - la cubre una detección de producto de baja confianza.
    3. Los tramos libres donde entra otro producto como el vecino se reportan como espacios vacíos. Un hueco puede partirse en varios
       (p. ej. un producto no detectado en medio de un hueco real) o
       desaparecer.

    A cada hueco confirmado se le agregan 'edge_density', 'color_std' y
    'max_window_edges' (la densidad de bordes de su ventana más texturada)
    para poder calibrar los umbrales, y se eliminan los campos internos ('_*').

    Si se pasa `rejected`, se le agregan los candidatos descartados con
    'reason' ('uprights' | 'texture' | 'low_conf') y 'max_window_edges'
    (None si se descartó antes de medir la textura).

    Retorna
    -------
    List[Dict]
        Huecos confirmados, en el mismo orden que los candidatos.
    """
    confirmed: List[Dict] = []

    for space in candidates:
        original = space["bbox"]
        min_gap = space.get("_min_gap", 0)
        ref_width = space.get("_ref_width")
        public = {k: v for k, v in space.items() if not k.startswith("_")}

        # El hueco tiene que poder alojar otro producto como el de al lado
        min_gap = max(min_gap, space.get("_neighbor_width") or 0)

        bbox = trim_to_uprights(img, space)
        if bbox is not None:
            bbox = trim_to_shelf_front(img, space, bbox)
        if bbox is None or bbox[2] - bbox[0] < min_gap:
            if debug:
                print(f"[verify_empty_spaces] Row {space['row']} {original} → {bbox} "
                      f"REJECT: no room for another product inside the gondola")
            if rejected is not None:
                rejected.append({**public, "reason": "uprights", "max_window_edges": None})
            continue

        x1, y1, x2, y2 = bbox
        height = y2 - y1

        # Textura por columna con ventana deslizante de medio producto
        col_density = (_edges(img, bbox) > 0).mean(axis=0)
        win = max(5, int((ref_width or (x2 - x1)) * 0.5))
        window_density = np.convolve(col_density, np.ones(win) / win, mode="same")
        textured = window_density > max_edge_density
        uprights = _dark_column_profile(img, bbox) >= _UPRIGHT_DARK_FRACTION
        occupied = textured | uprights

        # Detecciones de baja confianza que están a la altura de la fila
        for p in low_conf_products:
            px1, py1, px2, py2 = p["bbox"]
            v_overlap = max(0, min(py2, y2) - max(py1, y1))
            if v_overlap > 0.5 * min(py2 - py1, height):
                a, b = max(px1, x1) - x1, min(px2, x2) - x1
                if b > a:
                    occupied[a:b] = True

        runs = [(x1 + a, x1 + b) for a, b in _runs(~occupied) if b - a >= min_gap]

        if debug:
            print(f"[verify_empty_spaces] Row {space['row']} {original} → {bbox} "
                  f"max_window_edges={window_density.max():.3f} → free runs {runs or 'REJECT'}")

        if not runs and rejected is not None:
            # Motivo: el primer filtro que, por sí solo, deja el hueco sin tramos libres
            def _has_runs(mask):
                return any(b - a >= min_gap for a, b in _runs(~mask))
            if not _has_runs(textured):
                reason = "texture"
            elif not _has_runs(textured | uprights):
                reason = "uprights"
            else:
                reason = "low_conf"
            rejected.append({
                **public,
                "bbox": bbox,
                "reason": reason,
                "max_window_edges": round(float(window_density.max()), 4),
            })

        for rx1, rx2 in runs:
            run_bbox = (int(rx1), y1, int(rx2), y2)
            stats = texture_stats(img, run_bbox)
            width = run_bbox[2] - run_bbox[0]
            confirmed.append({
                **public,
                "bbox": run_bbox,
                "width_px": width,
                "estimated_slots": max(1, int(width // ref_width)) if ref_width else None,
                "edge_density": round(stats["edge_density"], 4),
                "color_std": round(stats["color_std"], 2),
                "max_window_edges": round(float(window_density[rx1 - x1:rx2 - x1].max()), 4),
            })

    return confirmed
