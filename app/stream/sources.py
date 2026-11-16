# app/stream/sources.py

import logging
import os
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Frame = Tuple[np.ndarray, float]


class LiveSource:
    """
    Cámara en vivo (índice local, RTSP, HTTP).

    Un hilo lee frames continuamente y guarda solo el último: así el análisis
    (más lento que la cámara) siempre trabaja sobre el presente y no sobre un
    buffer atrasado. Si la conexión se cae, reintenta con espera creciente.
    """

    is_live = True

    def __init__(self, source: Union[int, str], reconnect_max_wait: float = 30.0):
        self.source = source
        self.reconnect_max_wait = reconnect_max_wait
        self._frame: Optional[np.ndarray] = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        # Buffer mínimo: no queremos frames viejos encolados
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        wait = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                logger.error(f"Cannot open video source {self.source!r}, retrying in {wait:.0f}s")
                self._stop.wait(wait)
                wait = min(wait * 2, self.reconnect_max_wait)
                continue

            logger.info(f"Video source {self.source!r} opened")
            wait = 1.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    logger.warning(f"Lost video source {self.source!r}, reconnecting")
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                self._new_frame.set()
            cap.release()

    def next_frame(self, not_before: float) -> Optional[Frame]:
        """Espera hasta `not_before` (reloj monotónico) y devuelve el último frame."""
        delay = not_before - time.monotonic()
        if delay > 0:
            self._stop.wait(delay)
        self._new_frame.wait(timeout=5.0)
        with self._lock:
            if self._frame is None:
                return None
            self._new_frame.clear()
            return self._frame.copy(), self._frame_time

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


class FileSource:
    """
    Archivo de video. Se recorre en orden usando el tiempo del propio video,
    así el resultado es el mismo que si fuera en vivo, pero sin esperar.
    """

    is_live = False

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file {path!r}")

    def next_frame(self, not_before: float) -> Optional[Frame]:
        """Avanza hasta el primer frame con tiempo >= `not_before` (segundos de video)."""
        while True:
            if not self.cap.grab():
                return None
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t >= not_before:
                ok, frame = self.cap.retrieve()
                return (frame, t) if ok else None

    def close(self):
        self.cap.release()


def open_source(source: str):
    """
    '0', '1'...          → cámara local por índice
    rtsp:// http(s):// … → cámara IP en vivo
    ruta existente       → archivo de video
    """
    if source.isdigit():
        return LiveSource(int(source))
    if "://" in source:
        return LiveSource(source)
    if os.path.exists(source):
        return FileSource(source)
    raise ValueError(f"Unknown video source {source!r}")
