# app/localization/assign_column.py

from typing import List, Dict


def assign_columns(
    products: List[Dict],
    x_overlap_threshold: float = 0.3,
    debug: bool = False
) -> List[Dict]:
    """
    Asigna un número de columna a los productos dentro de cada fila de la repisa.
    Los productos que se superponen horizontalmente se agrupan en la misma columna.

    Parámetros
    ----------
    products : List[Dict]
        Lista de detecciones de productos que contienen:
        - 'bbox' (tuple[int, int, int, int])
        - 'row' (int)
    x_overlap_threshold : float
        Proporción mínima de superposición horizontal (0–1) para considerar
        que dos productos están en la misma columna.
    debug : bool
        Si es True, imprime información detallada de la agrupación.

    Retorna
    -------
    List[Dict]
        Lista de productos actualizada con:
        - 'col' (int) asignada por fila.
    """
    if not products:
        return products

    # Agrupar productos por fila
    rows: Dict[int, List[Dict]] = {}
    for p in products:
        row = p.get("row", 0)
        rows.setdefault(row, []).append(p)

    for row, items in rows.items():

        # Ordenar productos de izquierda a derecha según el centro en X
        items_sorted = sorted(
            items,
            key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2
        )

        col_idx = 1
        prev_box = None

        if debug:
            print(f"\n[assign_columns] Processing row {row} with {len(items_sorted)} products")

        for prod in items_sorted:
            x1, y1, x2, y2 = prod["bbox"]
            width = x2 - x1

            if prev_box is not None:
                x1_prev, y1_prev, x2_prev, y2_prev = prev_box

                # Calcular la proporción de superposición horizontal
                overlap_x = max(0, min(x2, x2_prev) - max(x1, x1_prev))
                min_width = min(width, x2_prev - x1_prev)
                x_overlap_ratio = overlap_x / min_width if min_width > 0 else 0

                # Si la superposición está por debajo del umbral → nueva columna
                if x_overlap_ratio < x_overlap_threshold:
                    col_idx += 1

                if debug:
                    print(
                        f" - Compared with previous: "
                        f"overlap_ratio={x_overlap_ratio:.2f}, col={col_idx}"
                    )

            prod["col"] = col_idx
            prev_box = prod["bbox"]

    return products
