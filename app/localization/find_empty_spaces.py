# app/localization/find_empty_spaces.py

from statistics import median
from typing import Dict, List, Optional, Tuple


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Une los intervalos horizontales (x1, x2) que se superponen.
    Así los productos apilados o encimados cuentan como un solo bloque ocupado.
    """
    merged: List[Tuple[int, int]] = []
    for x1, x2 in sorted(intervals):
        if merged and x1 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x2))
        else:
            merged.append((x1, x2))
    return merged


def find_empty_spaces(
    products: List[Dict],
    shelves: List[Dict],
    min_width_ratio: float = 1.0,
    debug: bool = False
) -> List[Dict]:
    """
    Propone espacios vacíos candidatos en cada fila de la góndola de forma
    geométrica. Los candidatos se deben confirmar después con
    `verify_empty_spaces`, porque un producto no detectado también deja un hueco.

    Para cada repisa (fila) se toman los intervalos horizontales ocupados por
    sus productos y se buscan los huecos:
    - entre el borde izquierdo de la góndola y el primer producto,
    - entre productos consecutivos,
    - entre el último producto y el borde derecho de la góndola.

    Los bordes de la góndola se estiman como el x mínimo / máximo de los
    productos de todas las filas: la bbox de la repisa suele extenderse sobre
    los parantes, la pared o la góndola vecina y generaba huecos falsos.

    Un hueco se considera candidato si su ancho es al menos `min_width_ratio`
    veces el ancho de referencia (mediana del ancho de los productos de la fila
    o, si la fila está vacía, de toda la imagen).
    Una repisa sin productos se reporta completa como candidato.
    Los niveles cortados por el borde de la imagen (altura < 40% de la mediana)
    se ignoran.

    Parámetros
    ----------
    products : List[Dict]
        Productos con 'bbox', 'row' y 'col' (salida de assign_rows / assign_columns).
    shelves : List[Dict]
        Repisas detectadas con 'bbox'.
    min_width_ratio : float
        Ancho mínimo del hueco, relativo al ancho de referencia del producto.
    debug : bool
        Si es True, imprime información detallada de cada fila.

    Retorna
    -------
    List[Dict]
        Espacios vacíos candidatos con:
        - 'bbox' (x1, y1, x2, y2): en vertical cubre la franja donde se apoyan
          los productos de la fila (no la repisa entera)
        - 'row' (int)
        - 'after_col' (int | None): columna del producto a la izquierda del
          hueco (None si el hueco está en el borde izquierdo o la fila está vacía)
        - 'width_px' (int)
        - 'estimated_slots' (int | None): cuántos productos entrarían en el hueco
    """
    if not shelves:
        return []

    # Mismo orden que assign_rows: fila 1 = repisa más baja
    shelves_sorted = sorted(shelves, key=lambda s: s["bbox"][3], reverse=True)

    all_widths = [p["bbox"][2] - p["bbox"][0] for p in products]
    global_ref_width: Optional[float] = median(all_widths) if all_widths else None

    # Límites horizontales de la góndola según los productos de todas las filas
    gondola_x1 = min((p["bbox"][0] for p in products), default=None)
    gondola_x2 = max((p["bbox"][2] for p in products), default=None)

    # Niveles cortados por el borde de la imagen (mucho más bajos que el
    # resto): no se ve lo suficiente para juzgar si falta algo
    median_height = median(s["bbox"][3] - s["bbox"][1] for s in shelves_sorted)

    empty_spaces: List[Dict] = []

    for row_idx, shelf in enumerate(shelves_sorted, start=1):
        sx1, sy1, sx2, sy2 = map(int, shelf["bbox"])
        if sy2 - sy1 < 0.4 * median_height:
            if debug:
                print(f"\n[find_empty_spaces] Row {row_idx}: partially visible, skipped")
            continue
        row_products = [p for p in products if p.get("row") == row_idx]

        # Recortar la fila a los límites de la góndola
        rx1 = max(sx1, gondola_x1) if gondola_x1 is not None else sx1
        rx2 = min(sx2, gondola_x2) if gondola_x2 is not None else sx2

        # Repisa sin productos → toda la repisa es candidato
        if not row_products:
            width = rx2 - rx1
            if width <= 0:
                continue
            slots = int(width // global_ref_width) if global_ref_width else None
            empty_spaces.append({
                "bbox": (rx1, sy1, rx2, sy2),
                "row": row_idx,
                "after_col": None,
                "width_px": width,
                "estimated_slots": slots,
                # Datos internos para verify_empty_spaces
                "_open_left": True,
                "_open_right": True,
                "_ref_width": global_ref_width,
                "_min_gap": (global_ref_width or 0) * min_width_ratio,
                "_neighbor_width": global_ref_width or 0,
            })
            if debug:
                print(f"\n[find_empty_spaces] Row {row_idx}: empty shelf")
            continue

        ref_width = median(p["bbox"][2] - p["bbox"][0] for p in row_products)
        min_gap = ref_width * min_width_ratio

        # Franja vertical donde están los productos de esta fila
        band_y1 = max(sy1, int(min(p["bbox"][1] for p in row_products)))
        band_y2 = min(sy2, int(median(p["bbox"][3] for p in row_products)))
        if band_y2 <= band_y1:
            band_y1, band_y2 = sy1, sy2

        occupied = _merge_intervals([(p["bbox"][0], p["bbox"][2]) for p in row_products])

        if debug:
            print(
                f"\n[find_empty_spaces] Row {row_idx}: ref_width={ref_width:.1f} "
                f"min_gap={min_gap:.1f} x=[{rx1}, {rx2}] occupied={occupied}"
            )

        # Bordes de los huecos candidatos: [borde góndola, bloques..., borde góndola]
        boundaries = [(rx1, rx1)] + occupied + [(rx2, rx2)]

        for (_, gap_x1), (gap_x2, _) in zip(boundaries, boundaries[1:]):
            width = gap_x2 - gap_x1
            if width < min_gap:
                continue

            # Columna del producto más a la derecha que termina antes del hueco
            left_cols = [p["col"] for p in row_products if p["bbox"][2] <= gap_x1 and "col" in p]
            after_col = max(left_cols) if left_cols else None

            # Ancho de los productos pegados al hueco: lo más probable es que
            # falte uno de esos, así que el hueco tiene que poder alojarlo
            neighbor_widths = [
                p["bbox"][2] - p["bbox"][0] for p in row_products
                if p["bbox"][2] == gap_x1 or p["bbox"][0] == gap_x2
            ]
            neighbor_width = min(neighbor_widths) if neighbor_widths else ref_width

            empty_spaces.append({
                "bbox": (gap_x1, band_y1, gap_x2, band_y2),
                "row": row_idx,
                "after_col": after_col,
                "width_px": width,
                "estimated_slots": max(1, int(width // ref_width)),
                # Datos internos para verify_empty_spaces: si el hueco toca
                # el borde de la fila (no hay producto de ese lado)
                "_open_left": gap_x1 == rx1,
                "_open_right": gap_x2 == rx2,
                "_ref_width": ref_width,
                "_min_gap": min_gap,
                "_neighbor_width": neighbor_width,
            })

            if debug:
                print(f" - Gap x={gap_x1}-{gap_x2} width={width} after_col={after_col}")

    return empty_spaces
