"""
Prueba aparte del enfoque de Roboflow para detectar faltantes
(https://roboflow.com/blog/retail-store-object-detection) y lo compara con
el pipeline propio.

El enfoque de Roboflow detecta los espacios vacíos directamente con un
modelo entrenado (RF-DETR sobre el dataset "empty-shelf-detection" de
Roboflow Universe, clase "Empty-space") y clasifica la góndola según cuánta
área cubren las detecciones:
    0 detecciones           -> STOCKED
    cobertura < 20 %        -> PARTIALLY STOCKED
    cobertura >= 20 %       -> EMPTY

Backends:
    roboflow : API hospedada de Roboflow. Necesita ROBOFLOW_API_KEY y el
               id del modelo ("proyecto/versión", p. ej. el fork entrenado
               de policenta/empty-shelf-detection).
    rfdetr   : pesos RF-DETR locales (pip install rfdetr), p. ej. los que
               se descargan de Roboflow después de entrenar.
    yolo     : pesos YOLO locales (ultralytics), p. ej.
               akul-29/Retail-Shelf-Gap-Detection_Model de Hugging Face.
               Se cargan sin ejecutar el pickle (ver _safe_load_yolo).

Uso (desde la raíz del repo):
    python -m tools.roboflow_empty_shelf --backend yolo --weights best.pt --confidence 0.25
    set ROBOFLOW_API_KEY=...
    python -m tools.roboflow_empty_shelf --model-id empty-shelf-detection/1
    python -m tools.roboflow_empty_shelf --backend rfdetr --weights checkpoint_best_total.pth

Genera en --out (por defecto output/roboflow_compare):
    - <imagen>_compare.jpg : izquierda pipeline propio, derecha Roboflow
    - predictions/<imagen>.json : respuesta cruda del modelo (caché: al
      volver a correr no se llama otra vez a la API; --no-cache para forzar)
    - comparison.json : estado, cobertura y coincidencias por imagen
"""

import argparse
import base64
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
ROBOFLOW_URL = "https://detect.roboflow.com"

# Umbrales del artículo (puntos de partida, a ajustar)
DEFAULT_CONFIDENCE = 0.4
PARTIAL_COVERAGE = 0.20
# Dos huecos son "el mismo" si la intersección cubre este tanto del menor
MATCH_MIN_OVERLAP = 0.3

COLOR_OURS = (0, 0, 255)        # rojo (igual que el pipeline)
COLOR_ROBOFLOW = (214, 120, 42)  # azul
COLOR_PRODUCT = (0, 200, 0)

BBox = Tuple[int, int, int, int]


# ---------------------------------------------------------------- backends

def predict_roboflow(img: np.ndarray, model_id: str, api_key: str, confidence: float) -> Dict:
    """Llama a la API hospedada de detección de Roboflow."""
    import requests

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("could not encode image")
    resp = requests.post(
        f"{ROBOFLOW_URL}/{model_id}",
        params={"api_key": api_key, "confidence": int(confidence * 100), "format": "json"},
        data=base64.b64encode(buf.tobytes()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Roboflow API {resp.status_code}: {resp.text[:300]}")
    return resp.json()


class RFDETRPredictor:
    """Pesos RF-DETR locales. Devuelve el mismo formato que la API hospedada."""

    def __init__(self, weights: str, size: str = "small"):
        try:
            import rfdetr
        except ImportError as e:
            raise SystemExit("Falta el paquete rfdetr: pip install rfdetr") from e
        cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
               "base": "RFDETRBase", "large": "RFDETRLarge"}[size]
        self.model = getattr(rfdetr, cls)(pretrain_weights=weights)

    def __call__(self, img: np.ndarray, confidence: float) -> Dict:
        from PIL import Image

        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dets = self.model.predict(rgb, threshold=confidence)
        preds = []
        for (x1, y1, x2, y2), conf in zip(dets.xyxy, dets.confidence):
            preds.append({
                "x": float(x1 + x2) / 2, "y": float(y1 + y2) / 2,
                "width": float(x2 - x1), "height": float(y2 - y1),
                "confidence": float(conf), "class": "Empty-space",
            })
        h, w = img.shape[:2]
        return {"predictions": preds, "image": {"width": w, "height": h}}


def _safe_load_yolo(weights: str):
    """
    Carga pesos YOLO de terceros sin ejecutar código arbitrario: el .pt es un
    pickle, así que se lee con torch.load(weights_only=True) permitiendo solo
    las clases de torch / ultralytics de una red de detección. Después se
    arma un modelo nuevo desde su yaml y se le copian los pesos.
    """
    import tempfile

    import torch
    import yaml
    from torch import nn
    from ultralytics import YOLO
    from ultralytics.nn import tasks
    from ultralytics.nn.modules import block, conv, head

    torch.serialization.add_safe_globals([
        set, torch.Size, torch.FloatStorage, torch.HalfStorage, torch.LongStorage,
        torch._utils._rebuild_parameter, torch._utils._rebuild_tensor_v2,
        nn.SiLU, nn.BatchNorm2d, nn.ModuleList, nn.Sequential, nn.Conv2d, nn.MaxPool2d, nn.Upsample,
        block.Bottleneck, block.C2f, block.DFL, block.SPPF, conv.Concat, conv.Conv, head.Detect,
        tasks.DetectionModel,
    ])
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    src = ckpt.get("ema") or ckpt["model"]

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(dict(src.yaml), f)
        cfg = f.name
    try:
        model = YOLO(cfg, task="detect")
    finally:
        os.remove(cfg)
    model.model.load_state_dict(src.float().state_dict())
    model.model.names = src.names
    model.model.eval()
    return model


class YOLOPredictor:
    """
    Pesos YOLO locales (ultralytics) entrenados para detectar huecos, p. ej.
    akul-29/Retail-Shelf-Gap-Detection_Model de Hugging Face.
    `classes`: nombres de clase que cuentan como hueco (None = todas).
    """

    def __init__(self, weights: str, classes: Optional[List[str]] = None):
        self.model = _safe_load_yolo(weights)
        self.classes = {c.lower() for c in classes} if classes else None

    def __call__(self, img: np.ndarray, confidence: float) -> Dict:
        r = self.model(img, conf=confidence, verbose=False)[0]
        preds = []
        for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                               r.boxes.cls.tolist()):
            name = r.names[int(cls)]
            if self.classes and name.lower() not in self.classes:
                continue
            preds.append({
                "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "width": x2 - x1, "height": y2 - y1,
                "confidence": conf, "class": name,
            })
        h, w = img.shape[:2]
        return {"predictions": preds, "image": {"width": w, "height": h}}


# ---------------------------------------------------------------- análisis

def to_boxes(response: Dict) -> List[Dict]:
    """Respuesta de Roboflow (centro + tamaño) -> [{'bbox': (x1, y1, x2, y2), 'conf'}]."""
    boxes = []
    for p in response.get("predictions", []):
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        boxes.append({
            "bbox": (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)),
            "conf": round(float(p.get("confidence", 0)), 3),
            "class": p.get("class"),
        })
    return boxes


def shelf_status(boxes: List[Dict], img_w: int, img_h: int,
                 partial_coverage: float = PARTIAL_COVERAGE) -> Tuple[str, float]:
    """Regla del artículo: suma de áreas de las cajas / área de la imagen."""
    if not boxes:
        return "STOCKED", 0.0
    area = sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) for b in boxes)
    coverage = area / float(img_w * img_h)
    status = "PARTIALLY STOCKED" if coverage < partial_coverage else "EMPTY"
    return status, round(coverage, 4)


def overlap_min(a: BBox, b: BBox) -> float:
    """Área de intersección relativa a la caja más chica."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def compare(ours: List[Dict], theirs: List[Dict], min_overlap: float = MATCH_MIN_OVERLAP) -> Dict:
    """Empareja huecos propios con detecciones de Roboflow (greedy por superposición)."""
    pairs = sorted(
        ((overlap_min(o["bbox"], t["bbox"]), i, j)
         for i, o in enumerate(ours) for j, t in enumerate(theirs)),
        reverse=True,
    )
    used_o, used_t, matches = set(), set(), []
    for score, i, j in pairs:
        if score < min_overlap:
            break
        if i in used_o or j in used_t:
            continue
        used_o.add(i)
        used_t.add(j)
        matches.append({"ours": i, "roboflow": j, "overlap": round(score, 3)})
    return {
        "matches": matches,
        "only_ours": [i for i in range(len(ours)) if i not in used_o],
        "only_roboflow": [j for j in range(len(theirs)) if j not in used_t],
    }


# ---------------------------------------------------------------- dibujo

def _label(img, text, x, y, color):
    cv2.putText(img, text, (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_compare(img, products, ours, theirs, status, coverage, source: str = "Roboflow") -> np.ndarray:
    left = img.copy()
    for p in products:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(left, (x1, y1), (x2, y2), COLOR_PRODUCT, 1)
    for i, s in enumerate(ours):
        x1, y1, x2, y2 = map(int, s["bbox"])
        cv2.rectangle(left, (x1, y1), (x2, y2), COLOR_OURS, 3)
        _label(left, f"#{i} R{s.get('row')}", x1 + 4, y1 + 18, COLOR_OURS)

    right = img.copy()
    for j, t in enumerate(theirs):
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(right, (x1, y1), (x2, y2), COLOR_ROBOFLOW, 3)
        _label(right, f"#{j} {t['conf']:.2f}", x1 + 4, y1 + 18, COLOR_ROBOFLOW)

    for panel, title in ((left, f"Pipeline propio: {len(ours)} huecos"),
                         (right, f"{source}: {len(theirs)} | {status} ({coverage:.1%})")):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return np.hstack([left, right])


# ---------------------------------------------------------------- main

def _cached_predict(name: str, img, predict, cache_dir: Optional[str]) -> Tuple[Dict, float, bool]:
    path = os.path.join(cache_dir, f"{os.path.splitext(name)[0]}.json") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), 0.0, True
    t0 = time.perf_counter()
    response = predict(img)
    elapsed = time.perf_counter() - t0
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)
    return response, elapsed, False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="input_images_test")
    parser.add_argument("--out", default="output/roboflow_compare")
    parser.add_argument("--backend", choices=["roboflow", "rfdetr", "yolo"], default="roboflow")
    parser.add_argument("--model-id", help="roboflow: 'proyecto/versión' del modelo")
    parser.add_argument("--weights", help="rfdetr: checkpoint .pth | yolo: pesos .pt")
    parser.add_argument("--classes", nargs="*", help="yolo: clases que cuentan como hueco (por defecto todas)")
    parser.add_argument("--rfdetr-size", default="small",
                        choices=["nano", "small", "medium", "base", "large"])
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--partial-coverage", type=float, default=PARTIAL_COVERAGE)
    parser.add_argument("--no-cache", action="store_true", help="volver a llamar al modelo aunque haya caché")
    parser.add_argument("--skip-pipeline", action="store_true", help="no correr el pipeline propio")
    args = parser.parse_args()

    if args.backend == "roboflow":
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not args.model_id or not api_key:
            parser.error("backend roboflow necesita --model-id y la variable ROBOFLOW_API_KEY")
        predict = lambda img: predict_roboflow(img, args.model_id, api_key, args.confidence)  # noqa: E731
    else:
        if not args.weights:
            parser.error(f"backend {args.backend} necesita --weights")
        predictor = (RFDETRPredictor(args.weights, args.rfdetr_size) if args.backend == "rfdetr"
                     else YOLOPredictor(args.weights, args.classes))
        predict = lambda img: predictor(img, args.confidence)  # noqa: E731

    cache_dir = None if args.no_cache else os.path.join(args.out, "predictions")
    os.makedirs(cache_dir or args.out, exist_ok=True)

    if not args.skip_pipeline:
        from app.core.model_loader import load_models
        from app.services.detection_pipeline import detect_shelf_image
        load_models()

    names = sorted(n for n in os.listdir(args.images) if n.lower().endswith(IMAGE_EXTS))
    summary = {}
    for name in names:
        img = cv2.imread(os.path.join(args.images, name))
        if img is None:
            print(f"[skip] {name}: no se pudo leer")
            continue
        h, w = img.shape[:2]

        response, rf_time, cached = _cached_predict(name, img, predict, cache_dir)
        theirs = to_boxes(response)
        status, coverage = shelf_status(theirs, w, h, args.partial_coverage)

        products, ours, our_time = [], [], None
        if not args.skip_pipeline:
            t0 = time.perf_counter()
            _, products, ours = detect_shelf_image(img)
            our_time = round(time.perf_counter() - t0, 3)

        cmp = compare(ours, theirs)
        cv2.imwrite(os.path.join(args.out, f"{os.path.splitext(name)[0]}_compare.jpg"),
                    draw_compare(img, products, ours, theirs, status, coverage,
                                 source="Roboflow" if args.backend == "roboflow" else f"Modelo {args.backend}"))

        summary[name] = {
            "roboflow": {
                "status": status, "coverage": coverage, "detections": theirs,
                "time_s": None if cached else round(rf_time, 3), "cached": cached,
            },
            "pipeline": {
                "empty_spaces": [{"bbox": list(map(int, s["bbox"])), "row": s.get("row")} for s in ours],
                "time_s": our_time,
            },
            "comparison": cmp,
        }
        print(f"{name:20s} modelo={len(theirs):2d} ({status}, {coverage:.1%})  "
              f"propio={len(ours):2d}  coinciden={len(cmp['matches'])}  "
              f"solo_roboflow={len(cmp['only_roboflow'])}  solo_propio={len(cmp['only_ours'])}")

    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResultados en {args.out}")


if __name__ == "__main__":
    main()
