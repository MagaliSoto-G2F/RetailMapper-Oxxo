import cv2

from app.config import COLOR_SHELF, COLOR_PRODUCT, COLOR_EMPTY_SPACE


def draw_shelves(frame, shelves, color=COLOR_SHELF):
    """Dibuja la línea inferior de cada repisa detectada."""
    for s in shelves:
        x1, y1, x2, y2 = map(int, s["bbox"])
        cv2.line(frame, (x1, y2), (x2, y2), color, 2)
        cv2.putText(frame, f"shelf {s.get('conf', 0):.2f}",
                    (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_products(frame, products, color=COLOR_PRODUCT):
    """Dibuja un recuadro por producto con su ID, fila y columna."""
    for p in products:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"ID {p.get('id')} R{p.get('row')} C{p.get('col')}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def draw_empty_spaces(frame, empty_spaces, color=COLOR_EMPTY_SPACE):
    """Dibuja un recuadro por espacio vacío con su ID y fila."""
    for e in empty_spaces:
        x1, y1, x2, y2 = map(int, e["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"EMPTY {e.get('id')} R{e.get('row')}", (x1 + 5, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
