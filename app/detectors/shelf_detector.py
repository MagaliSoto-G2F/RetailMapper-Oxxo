# app/detectors/shelf_detector.py

from ultralytics import YOLO


def _vertical_nms(detections: list[dict], max_overlap: float) -> list[dict]:
    """
    Las repisas son niveles apilados: dos detecciones que se solapan mucho en
    vertical son el mismo nivel. Se queda la de mayor confianza.
    """
    kept: list[dict] = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        _, y1, _, y2 = det["bbox"]
        duplicate = False
        for k in kept:
            _, ky1, _, ky2 = k["bbox"]
            overlap = max(0, min(y2, ky2) - max(y1, ky1))
            if overlap / max(1, min(y2 - y1, ky2 - ky1)) > max_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


class ShelfDetector:
    """
    Detector de repisas basado en YOLO.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.4,
                 max_vertical_overlap: float = 0.5) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.max_vertical_overlap = max_vertical_overlap

    def detect(self, frame) -> list[dict]:
        """
        Ejecuta la detección de repisas sobre un frame (array numpy BGR).

        Retorna
        -------
        list[dict]
            Repisas detectadas con 'bbox' (x1, y1, x2, y2) y 'conf'.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        if not results or not hasattr(results[0], "boxes"):
            return detections

        for det in results[0].boxes:
            conf = float(det.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append({"bbox": (x1, y1, x2, y2), "conf": conf})

        return _vertical_nms(detections, self.max_vertical_overlap)
