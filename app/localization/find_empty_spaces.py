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
    Detecta espacios vacíos en cada fila de la góndola de forma geométrica.

    Para cada repisa (fila) se toman los intervalos horizontales ocupados por
    sus productos y se buscan los huecos:
    - entre el borde izquierdo de la repisa y el primer producto,
    - entre productos consecutivos,
    - entre el último producto y el borde derecho de la repisa.
    Un hueco se considera espacio vacío si su ancho es al menos
    `min_width_ratio` veces el ancho de referencia (mediana del ancho de los
    productos de la fila o, si la fila está vacía, de toda la imagen).
    Una repisa sin productos se reporta completa como espacio vacío.

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
        Espacios vacíos con:
        - 'id' (int)
        - 'bbox' (x1, y1, x2, y2)
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

    empty_spaces: List[Dict] = []

    for row_idx, shelf in enumerate(shelves_sorted, start=1):
        sx1, sy1, sx2, sy2 = map(int, shelf["bbox"])
        row_products = [p for p in products if p.get("row") == row_idx]

        # Repisa sin productos → toda la repisa es espacio vacío
        if not row_products:
            width = sx2 - sx1
            slots = int(width // global_ref_width) if global_ref_width else None
            empty_spaces.append({
                "bbox": (sx1, sy1, sx2, sy2),
                "row": row_idx,
                "after_col": None,
                "width_px": width,
                "estimated_slots": slots,
            })
            if debug:
                print(f"\n[find_empty_spaces] Row {row_idx}: empty shelf")
            continue

        ref_width = median(p["bbox"][2] - p["bbox"][0] for p in row_products)
        min_gap = ref_width * min_width_ratio

        occupied = _merge_intervals([(p["bbox"][0], p["bbox"][2]) for p in row_products])

        if debug:
            print(
                f"\n[find_empty_spaces] Row {row_idx}: ref_width={ref_width:.1f} "
                f"min_gap={min_gap:.1f} occupied={occupied}"
            )

        # Bordes de los huecos candidatos: [borde repisa, bloques..., borde repisa]
        boundaries = [(sx1, sx1)] + occupied + [(sx2, sx2)]

        for (_, gap_x1), (gap_x2, _) in zip(boundaries, boundaries[1:]):
            width = gap_x2 - gap_x1
            if width < min_gap:
                continue

            # Columna del producto más a la derecha que termina antes del hueco
            left_cols = [p["col"] for p in row_products if p["bbox"][2] <= gap_x1 and "col" in p]
            after_col = max(left_cols) if left_cols else None

            empty_spaces.append({
                "bbox": (gap_x1, sy1, gap_x2, sy2),
                "row": row_idx,
                "after_col": after_col,
                "width_px": width,
                "estimated_slots": max(1, int(width // ref_width)),
            })

            if debug:
                print(f" - Gap x={gap_x1}-{gap_x2} width={width} after_col={after_col}")

    for i, space in enumerate(empty_spaces, start=1):
        space["id"] = i

    return empty_spaces
