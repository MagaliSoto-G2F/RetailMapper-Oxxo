# app/detectors/shelf_detector.py

from ultralytics import YOLO


class ShelfDetector:
    """
    Detector de repisas basado en YOLO.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.4) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[dict]:
        """
        Ejecuta la detección de repisas sobre un frame (array numpy BGR).

        Retorna
        -------
        list[dict]
            Repisas detectadas con 'bbox' (x1, y1, x2, y2) y 'conf'.
        """
        results = self.model(frame)
        detections = []

        if not results or not hasattr(results[0], "boxes"):
            return detections

        for det in results[0].boxes:
            conf = float(det.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append({"bbox": (x1, y1, x2, y2), "conf": conf})

        return detections
