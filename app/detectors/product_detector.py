# app/detectors/product_detector.py

from ultralytics import YOLO


class ProductDetector:
    """
    Detector de productos basado en YOLO.
    """

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold: float = 0.3) -> list[dict]:
        """
        Ejecuta la detección de productos sobre un frame (array numpy BGR).

        Retorna
        -------
        list[dict]
            Productos detectados con 'bbox' (x1, y1, x2, y2) y 'conf'.
        """
        results = self.model(frame)
        detections = []

        if not results or not hasattr(results[0], "boxes"):
            return detections

        for det in results[0].boxes:
            conf = float(det.conf[0])
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append({"bbox": (x1, y1, x2, y2), "conf": conf})

        return detections
