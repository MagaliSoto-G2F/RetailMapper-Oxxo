# app/stream/alerts.py

import json
import logging
from dataclasses import asdict
from typing import Protocol

from app.stream.tracker import ShelfEvent

logger = logging.getLogger(__name__)


class AlertSink(Protocol):
    """Destino de los eventos de la góndola (webhook, mail, cola, etc.)."""

    def send(self, event: ShelfEvent) -> None:
        ...


class LogAlertSink:
    """
    Sink por defecto: solo deja el evento en el log.

    TODO: implementar el envío real de alertas (pendiente de definir el canal:
    webhook, WhatsApp/Telegram, mail, cola de mensajes...). Alcanza con una
    clase con `send(event)` y pasarla a `StreamMonitor(alert_sink=...)`.
    El evento trae `snapshot` con la ruta de la captura anotada.
    """

    def send(self, event: ShelfEvent) -> None:
        logger.warning("SHELF EVENT %s", json.dumps(asdict(event), ensure_ascii=False))
