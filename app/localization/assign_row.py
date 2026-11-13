# app/localization/assign_row.py

def assign_rows(
        products:list[dict], 
        shelves:list[dict],
         debug=False
    )-> list[dict]:
    """
    Asigna cada producto a la fila de la repisa que tiene directamente debajo.
    Las filas se numeran de abajo (1) hacia arriba (N).

    Parámetros
    ----------
    products : list[dict]
        Lista de detecciones de productos con 'bbox'.
    shelves : list[dict]
        Lista de detecciones de repisas con 'bbox'.
    debug : bool
        Si es True, imprime información detallada del proceso de asignación.

    Retorna
    -------
    list[dict]
        Lista de productos actualizada con 'row' asignada.
    """
    if not shelves or not products:
        return products

    # Ordenar repisas por la coordenada Y inferior (descendente = de abajo hacia arriba)
    shelves_sorted = sorted(shelves, key=lambda s: s["bbox"][3], reverse=True)

    if debug:
        print(f"\n[assign_rows] Found {len(shelves_sorted)} shelves (bottom→top)")

    for p in products:
        x1, y1, x2, y2 = p["bbox"]
        center_y = (y1 + y2) / 2
        row_assigned = None

        # Buscar la repisa que está directamente debajo del producto
        for i, shelf in enumerate(shelves_sorted):
            shelf_top = shelf["bbox"][1]  # Y superior de la repisa
            shelf_bottom = shelf["bbox"][3]  # Y inferior de la repisa
            if shelf_bottom >= center_y >= shelf_top:
                # El producto está dentro de la repisa → se asigna esa fila
                row_assigned = i + 1
                break
            elif center_y < shelf_top:
                # El producto está por encima de esta repisa, seguir buscando más arriba
                continue
            elif center_y > shelf_bottom:
                # El producto está por debajo de esta repisa → se asigna esta fila
                row_assigned = i + 1
                break

        # Si el producto está por encima de la repisa más alta
        if row_assigned is None:
            row_assigned = len(shelves_sorted)

        p["row"] = row_assigned

        if debug:
            print(f" - Product bbox={p['bbox']} center_y={center_y:.1f} → Row {row_assigned}")

    return products
