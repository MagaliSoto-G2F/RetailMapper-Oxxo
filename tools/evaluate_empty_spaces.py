"""
Evalúa la detección de espacios vacíos sobre una carpeta de imágenes y
grafica el resultado.

Uso (desde la raíz del repo):
    python -m tools.evaluate_empty_spaces [--images input_images_test] [--out output/eval]

Genera en --out:
    - <imagen>_eval.jpg : productos (verde), huecos confirmados (rojo) y
      candidatos descartados (naranja, con el motivo)
    - performance.png   : gráfico de separación por textura, resultado de los
      candidatos y tiempo de post-proceso por imagen
    - summary.json      : los números detrás del gráfico
"""

import argparse
import json
import os
import time
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.config import (  # noqa: E402
    PRODUCT_CONF_THRESHOLD,
    EMPTY_SPACE_MIN_WIDTH_RATIO,
    EMPTY_SPACE_LOW_CONF_THRESHOLD,
    EMPTY_SPACE_MAX_EDGE_DENSITY,
)
from app.core.model_loader import load_models  # noqa: E402
from app.localization.assign_row import assign_rows  # noqa: E402
from app.localization.assign_column import assign_columns  # noqa: E402
from app.localization.find_empty_spaces import find_empty_spaces  # noqa: E402
from app.localization.verify_empty_spaces import texture_stats, verify_empty_spaces  # noqa: E402
from app.utils.draw import draw_shelves, draw_products, draw_empty_spaces  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

REASON_LABELS = {
    "uprights": "parante / fuera de góndola",
    "texture": "textura (producto no detectado)",
    "low_conf": "detección de baja confianza",
}

# Paleta (colores categóricos en orden fijo + tinta y superficie)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
C_PRODUCT = "#2a78d6"    # azul
C_CONFIRMED = "#eb6834"  # naranja
C_REJ_TEXTURE = "#1baf7a"  # aqua (textura o baja confianza)
C_REJ_UPRIGHT = "#eda100"  # amarillo
C_TIME = "#52514e"


def evaluate_image(img, shelf_det, prod_det) -> Dict:
    t0 = time.perf_counter()
    shelves = shelf_det.detect(img)
    all_products = prod_det.detect(img, conf_threshold=EMPTY_SPACE_LOW_CONF_THRESHOLD)
    t1 = time.perf_counter()

    products = [p for p in all_products if p["conf"] >= PRODUCT_CONF_THRESHOLD]
    low_conf = [p for p in all_products if p["conf"] < PRODUCT_CONF_THRESHOLD]
    products = assign_columns(assign_rows(products, shelves))
    for i, p in enumerate(products, start=1):
        p["id"] = i

    candidates = find_empty_spaces(products, shelves, min_width_ratio=EMPTY_SPACE_MIN_WIDTH_RATIO)
    n_candidates = len(candidates)
    rejected: List[Dict] = []
    confirmed = verify_empty_spaces(
        img, candidates, low_conf,
        max_edge_density=EMPTY_SPACE_MAX_EDGE_DENSITY,
        rejected=rejected,
    )
    for i, s in enumerate(confirmed, start=1):
        s["id"] = i
    t2 = time.perf_counter()

    return {
        "shelves": shelves,
        "products": products,
        "low_conf": low_conf,
        "n_candidates": n_candidates,
        "confirmed": confirmed,
        "rejected": rejected,
        "product_edge_density": [texture_stats(img, p["bbox"])["edge_density"] for p in products],
        "yolo_ms": (t1 - t0) * 1000,
        "postproc_ms": (t2 - t1) * 1000,
    }


def draw_eval(img, r: Dict):
    out = draw_shelves(img.copy(), r["shelves"])
    out = draw_products(out, r["products"])
    for c in r["rejected"]:
        x1, y1, x2, y2 = map(int, c["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(out, f"X {c['reason']}", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    return draw_empty_spaces(out, r["confirmed"])


def _style_axis(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(results: Dict[str, Dict], out_path: str):
    names = list(results)
    fig = plt.figure(figsize=(12, 9), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax_sep = fig.add_subplot(gs[0, :])
    ax_out = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])

    # ---- 1. Separación por textura ----
    rng = np.random.default_rng(0)
    groups = [
        ("Productos detectados",
         [d for r in results.values() for d in r["product_edge_density"]], C_PRODUCT),
        ("Huecos confirmados",
         [c["max_window_edges"] for r in results.values() for c in r["confirmed"]], C_CONFIRMED),
        ("Descartados por textura\no baja confianza",
         [c["max_window_edges"] for r in results.values() for c in r["rejected"]
          if c["reason"] != "uprights" and c["max_window_edges"] is not None], C_REJ_TEXTURE),
    ]
    floor, ceil = 3e-3, 0.6
    for y, (label, values, color) in enumerate(reversed(groups)):
        if not values:
            ax_sep.text(floor * 1.2, y, "ninguno", va="center", color=TEXT_2, fontsize=9, style="italic")
            continue
        xs = np.clip(values, floor, ceil)
        ys = y + rng.uniform(-0.18, 0.18, len(xs))
        ax_sep.scatter(xs, ys, s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax_sep.text(ceil * 0.95, y, f"n={len(xs)}", va="center", ha="right", color=TEXT_2, fontsize=9)

    ax_sep.axvline(EMPTY_SPACE_MAX_EDGE_DENSITY, color=TEXT, linestyle=(0, (4, 3)), linewidth=1.2)
    ax_sep.text(EMPTY_SPACE_MAX_EDGE_DENSITY * 1.05, len(groups) - 0.45,
                f"umbral = {EMPTY_SPACE_MAX_EDGE_DENSITY}", color=TEXT, fontsize=9)
    ax_sep.set_xscale("log")
    ax_sep.set_xlim(floor, ceil)
    ax_sep.set_yticks(range(len(groups)))
    ax_sep.set_yticklabels([g[0] for g in reversed(groups)], color=TEXT, fontsize=10)
    ax_sep.set_ylim(-0.6, len(groups) - 0.3)
    ax_sep.set_xlabel(
        "Densidad de bordes (Canny, escala log) · productos: bbox completa · "
        "huecos: su ventana más texturada", color=TEXT_2, fontsize=9)
    _style_axis(ax_sep, "Separación por textura: a la izquierda del umbral = vacío")

    # ---- 2. Resultado de los candidatos por imagen ----
    series = [
        ("Confirmados", lambda r: len(r["confirmed"]), C_CONFIRMED),
        ("Descartados: textura / baja conf.",
         lambda r: sum(c["reason"] != "uprights" for c in r["rejected"]), C_REJ_TEXTURE),
        ("Descartados: parante / fuera de góndola",
         lambda r: sum(c["reason"] == "uprights" for c in r["rejected"]), C_REJ_UPRIGHT),
    ]
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for label, fn, color in series:
        vals = np.array([fn(results[n]) for n in names], dtype=float)
        ax_out.barh(y, vals, left=left, color=color, height=0.55, label=label,
                    edgecolor=SURFACE, linewidth=2)
        left += vals
    for yi, n, total in zip(y, names, left):
        n_cand = results[n]["n_candidates"]
        label = f"{n_cand} candidatos"
        if total > n_cand:
            # Un candidato puede partirse en varios huecos al verificarlo
            label += f" ({int(total - n_cand)} partido{'s' if total - n_cand > 1 else ''})"
        ax_out.text(total + 0.1, yi, label, va="center", color=TEXT_2, fontsize=9)
    ax_out.set_yticks(y)
    ax_out.set_yticklabels(names, color=TEXT, fontsize=10)
    ax_out.invert_yaxis()
    ax_out.set_xlim(0, max(left.max(), 1) * 1.45)
    ax_out.xaxis.get_major_locator().set_params(integer=True)
    ax_out.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=1, frameon=False,
                  fontsize=8.5, labelcolor=TEXT)
    _style_axis(ax_out, "Resultado de los candidatos")

    # ---- 3. Tiempo de post-proceso ----
    post = [results[n]["postproc_ms"] for n in names]
    ax_time.barh(y, post, color=C_TIME, height=0.55)
    for yi, n, v in zip(y, names, post):
        ax_time.text(v, yi, f"  {v:.0f} ms  (inferencia YOLO {results[n]['yolo_ms']:.0f} ms)",
                     va="center", color=TEXT_2, fontsize=9)
    ax_time.set_yticks(y)
    ax_time.set_yticklabels(names, color=TEXT, fontsize=10)
    ax_time.invert_yaxis()
    ax_time.set_xlim(0, max(max(post), 1) * 2.6)
    ax_time.set_xlabel("ms por imagen (filas, columnas, huecos y verificación)", color=TEXT_2, fontsize=9)
    _style_axis(ax_time, "Tiempo de post-proceso")

    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="input_images_test")
    parser.add_argument("--out", default=os.path.join("output", "eval"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shelf_det, prod_det = load_models()

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMAGE_EXTS))
    # Warm-up: la primera inferencia incluye la inicialización de PyTorch
    warm = np.zeros((640, 640, 3), dtype=np.uint8)
    shelf_det.detect(warm)
    prod_det.detect(warm)

    results: Dict[str, Dict] = {}
    for f in files:
        img = cv2.imread(os.path.join(args.images, f))
        if img is None:
            print(f"[skip] {f}: no se pudo leer")
            continue
        name = os.path.splitext(f)[0]
        r = evaluate_image(img, shelf_det, prod_det)
        results[name] = r
        cv2.imwrite(os.path.join(args.out, f"{name}_eval.jpg"), draw_eval(img, r))

    print(f"\n{'imagen':<16}{'filas':>6}{'prod':>6}{'cand':>6}{'conf':>6}"
          f"{'textura':>9}{'baja':>6}{'parante':>9}{'post ms':>9}{'yolo ms':>9}")
    summary = {}
    for name, r in results.items():
        reasons = {k: sum(c["reason"] == k for c in r["rejected"]) for k in REASON_LABELS}
        summary[name] = {
            "shelves": len(r["shelves"]),
            "products": len(r["products"]),
            "low_conf_products": len(r["low_conf"]),
            "candidates": r["n_candidates"],
            "confirmed": len(r["confirmed"]),
            "rejected_by_reason": reasons,
            "postproc_ms": round(r["postproc_ms"], 1),
            "yolo_ms": round(r["yolo_ms"], 1),
            "empty_spaces": [{k: c[k] for k in ("row", "bbox", "width_px", "max_window_edges")}
                             for c in r["confirmed"]],
        }
        print(f"{name:<16}{len(r['shelves']):>6}{len(r['products']):>6}{r['n_candidates']:>6}"
              f"{len(r['confirmed']):>6}{reasons['texture']:>9}{reasons['low_conf']:>6}"
              f"{reasons['uprights']:>9}{r['postproc_ms']:>9.0f}{r['yolo_ms']:>9.0f}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    plot(results, os.path.join(args.out, "performance.png"))
    print(f"\nGráfico: {os.path.join(args.out, 'performance.png')}")


if __name__ == "__main__":
    main()
