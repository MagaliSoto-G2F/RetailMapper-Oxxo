# app/stream/monitor.py
"""
Monitorea una góndola desde un stream de video y avisa cuando se retira un
producto.

Uso (desde la raíz del repo):
    python -m app.stream.monitor --source 0                       # webcam
    python -m app.stream.monitor --source rtsp://user:pass@ip/... # cámara IP
    python -m app.stream.monitor --source video.mp4 --show        # archivo
"""

import argparse
import logging
import os
import time
from typing import List, Optional

import cv2

from app.config import (
    STREAM_PROCESS_INTERVAL_SEC,
    STREAM_CONSENSUS_WINDOW,
    STREAM_CONSENSUS_MIN_HITS,
    STREAM_MATCH_MIN_OVERLAP,
    STREAM_EVENTS_FOLDER,
)
from app.core.model_loader import load_models
from app.services.detection_pipeline import detect_shelf_image
from app.stream.alerts import AlertSink, LogAlertSink
from app.stream.sources import open_source
from app.stream.tracker import EmptySpaceTracker, ShelfEvent
from app.utils.draw import draw_products

logger = logging.getLogger(__name__)

COLOR_CONFIRMED = (0, 0, 255)  # rojo
COLOR_PENDING = (0, 165, 255)  # naranja
COLOR_EVENT = (255, 0, 255)    # magenta


def draw_tracks(frame, tracks: List[dict], highlight: Optional[ShelfEvent] = None):
    """Huecos confirmados en rojo, en observación (sin consenso aún) en naranja."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color = COLOR_CONFIRMED if t["confirmed"] else COLOR_PENDING
        label = f"EMPTY T{t['id']} R{t['row']}" if t["confirmed"] else f"? T{t['id']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 4, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if highlight is not None:
        x1, y1, x2, y2 = map(int, highlight.bbox)
        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), COLOR_EVENT, 3)
        cv2.putText(frame, highlight.type.upper(), (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EVENT, 2)
    return frame


class StreamMonitor:
    def __init__(
        self,
        source: str,
        alert_sink: Optional[AlertSink] = None,
        interval: float = STREAM_PROCESS_INTERVAL_SEC,
        events_folder: str = STREAM_EVENTS_FOLDER,
        show: bool = False,
    ):
        self.source_name = source
        self.alert_sink = alert_sink or LogAlertSink()
        self.interval = interval
        self.events_folder = events_folder
        self.show = show
        self.tracker = EmptySpaceTracker(
            window=STREAM_CONSENSUS_WINDOW,
            min_hits=STREAM_CONSENSUS_MIN_HITS,
            min_overlap=STREAM_MATCH_MIN_OVERLAP,
        )
        self.events: List[ShelfEvent] = []

    def _save_snapshot(self, frame, products, event: ShelfEvent) -> str:
        os.makedirs(self.events_folder, exist_ok=True)
        img = draw_products(frame.copy(), products)
        img = draw_tracks(img, self.tracker.tracks(), highlight=event)
        stamp = event.timestamp.replace(":", "-")
        path = os.path.join(self.events_folder, f"{stamp}_{event.type}_T{event.track_id}.jpg")
        cv2.imwrite(path, img)
        return path

    def run(self, max_seconds: Optional[float] = None):
        load_models()
        source = open_source(self.source_name)
        clock = time.monotonic if source.is_live else (lambda: 0.0)
        start = clock()
        next_t = start if source.is_live else 0.0
        logger.info(f"Monitoring {self.source_name!r} every {self.interval}s")

        try:
            while True:
                item = source.next_frame(next_t)
                if item is None:
                    if source.is_live:
                        continue  # el hilo lector está reconectando
                    break  # fin del archivo
                frame, t = item
                next_t = t + self.interval
                if max_seconds is not None and t - start > max_seconds:
                    break

                _, products, empty_spaces = detect_shelf_image(frame)
                for event in self.tracker.update(empty_spaces, stream_time=t - start):
                    event.snapshot = self._save_snapshot(frame, products, event)
                    self.events.append(event)
                    self.alert_sink.send(event)

                if self.show:
                    view = draw_products(frame.copy(), products)
                    view = draw_tracks(view, self.tracker.tracks())
                    status = "calibrando linea base" if self.tracker.warming_up else "monitoreando"
                    cv2.putText(view, f"t={t - start:.0f}s {status}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("RetailMapper stream", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            source.close()
            if self.show:
                cv2.destroyAllWindows()

        return self.events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="índice de cámara, URL rtsp/http o archivo de video")
    parser.add_argument("--interval", type=float, default=STREAM_PROCESS_INTERVAL_SEC,
                        help="segundos entre análisis")
    parser.add_argument("--show", action="store_true", help="mostrar ventana con el estado en vivo")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    StreamMonitor(args.source, interval=args.interval, show=args.show).run(max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
