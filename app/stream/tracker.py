# app/stream/tracker.py

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
from statistics import median
from typing import Deque, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]


@dataclass
class ShelfEvent:
    """
    Cambio estable en la góndola.

    type:
      - 'product_removed': apareció un hueco nuevo, o uno existente creció.
      - 'restocked': un hueco desapareció o se achicó.
    slots: cuántos productos cambiaron (delta, siempre > 0).
    total_slots: tamaño del hueco después del cambio (0 si desapareció).
    """
    type: str
    track_id: int
    row: Optional[int]
    bbox: BBox
    slots: int
    total_slots: int
    stream_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    snapshot: Optional[str] = None


@dataclass
class _Track:
    id: int
    bbox: BBox
    row: Optional[int]
    window: int
    hits: Deque[bool] = field(init=False)
    slots: Deque[int] = field(init=False)
    confirmed: bool = False
    reported_slots: int = 0
    baseline: bool = False  # nació durante el calentamiento: ya estaba vacío

    def __post_init__(self):
        self.hits = deque(maxlen=self.window)
        self.slots = deque(maxlen=self.window)

    @property
    def n_hits(self) -> int:
        return sum(self.hits)

    @property
    def n_misses(self) -> int:
        return len(self.hits) - self.n_hits

    @property
    def stable_slots(self) -> int:
        return int(median(self.slots)) if self.slots else 0

    def consensus_slots(self, min_hits: int) -> Optional[int]:
        """Tamaño que se repite en al menos `min_hits` análisis (None si no hay acuerdo)."""
        if not self.slots:
            return None
        value, count = Counter(self.slots).most_common(1)[0]
        return value if count >= min_hits else None


def _overlap(a: BBox, b: BBox) -> float:
    """Superposición horizontal × vertical, cada una relativa al lado menor."""
    ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    min_w = min(a[2] - a[0], b[2] - b[0])
    min_h = min(a[3] - a[1], b[3] - b[1])
    if min_w <= 0 or min_h <= 0:
        return 0.0
    return (ox / min_w) * (oy / min_h)


class EmptySpaceTracker:
    """
    Sigue los espacios vacíos a lo largo de los análisis de un stream y emite
    eventos solo cuando un cambio es estable (consenso temporal).

    - Un hueco se confirma cuando aparece en al menos `min_hits` de los
      últimos `window` análisis. Una persona que pasa o un frame movido
      generan huecos falsos durante 1–2 análisis, que nunca se confirman.
    - Los huecos confirmados durante los primeros `window` análisis son la
      línea base (ya estaban vacíos al arrancar) y no generan alerta.
    - Un hueco confirmado que crece (más `estimated_slots`) genera otro
      'product_removed' con la diferencia.
    - Un hueco confirmado que falta en `min_hits` de los últimos `window`
      análisis se da por repuesto, salvo que haya quedado dentro de otro hueco
      confirmado.
    - Si un hueco crece y cubre a otros huecos confirmados (se retiró el
      producto que los separaba), se fusionan sumando lo ya reportado, así la
      alerta informa solo lo que se retiró de verdad.

    El emparejamiento entre análisis es por geometría, no por número de fila,
    porque la numeración puede cambiar si el detector pierde una repisa.
    """

    def __init__(self, window: int, min_hits: int, min_overlap: float):
        if not 0 < min_hits <= window:
            raise ValueError("min_hits must be in (0, window]")
        self.window = window
        self.min_hits = min_hits
        self.min_overlap = min_overlap
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._updates = 0

    @property
    def warming_up(self) -> bool:
        return self._updates < self.window

    def tracks(self) -> List[Dict]:
        """Estado actual, para dibujar: [{'id', 'bbox', 'row', 'confirmed', 'slots'}]."""
        return [
            {"id": t.id, "bbox": t.bbox, "row": t.row,
             "confirmed": t.confirmed, "slots": t.reported_slots or t.stable_slots}
            for t in self._tracks
        ]

    def _match(self, spaces: List[Dict]) -> Dict[int, int]:
        """Empareja track → hueco detectado, de mayor a menor superposición."""
        pairs = sorted(
            ((_overlap(t.bbox, s["bbox"]), ti, si)
             for ti, t in enumerate(self._tracks)
             for si, s in enumerate(spaces)),
            reverse=True,
        )
        matches: Dict[int, int] = {}
        used = set()
        for score, ti, si in pairs:
            if score < self.min_overlap:
                break
            if ti in matches or si in used:
                continue
            matches[ti] = si
            used.add(si)
        return matches

    def update(self, empty_spaces: List[Dict], stream_time: float) -> List[ShelfEvent]:
        """
        Incorpora los huecos confirmados de un análisis (salida de
        detect_shelf_image) y devuelve los eventos estables que se produjeron.
        """
        warming_up = self.warming_up
        self._updates += 1
        events: List[ShelfEvent] = []

        matches = self._match(empty_spaces)
        matched_spaces = set(matches.values())

        for ti, track in enumerate(self._tracks):
            if ti in matches:
                s = empty_spaces[matches[ti]]
                track.hits.append(True)
                track.slots.append(int(s.get("estimated_slots") or 1))
                track.bbox = tuple(int(v) for v in s["bbox"])
                track.row = s.get("row")
            else:
                track.hits.append(False)

        for si, s in enumerate(empty_spaces):
            if si in matched_spaces:
                continue
            track = _Track(self._next_id, tuple(int(v) for v in s["bbox"]), s.get("row"), self.window)
            track.baseline = warming_up
            track.hits.append(True)
            track.slots.append(int(s.get("estimated_slots") or 1))
            self._tracks.append(track)
            self._next_id += 1

        survivors: List[_Track] = []
        for track in self._tracks:
            event = self._step(track, stream_time, warming_up)
            if event is not None:
                events.append(event)
            if self._alive(track):
                survivors.append(track)
                continue
            keeper = self._covering_track(track)
            if keeper is not None:
                keeper.reported_slots += track.reported_slots
                track.reported_slots = 0
            elif track.confirmed and track.reported_slots > 0:
                events.append(ShelfEvent(
                    "restocked", track.id, track.row, track.bbox,
                    slots=track.reported_slots, total_slots=0, stream_time=stream_time,
                ))
        # Los absorbidos por otro hueco en este update quedan sin historial
        self._tracks = [t for t in survivors if t.hits]
        return events

    def _step(self, track: _Track, t: float, warming_up: bool) -> Optional[ShelfEvent]:
        if not track.confirmed:
            if track.n_hits >= self.min_hits:
                track.confirmed = True
                track.reported_slots = track.consensus_slots(self.min_hits) or track.stable_slots
                if not (warming_up or track.baseline):
                    return ShelfEvent(
                        "product_removed", track.id, track.row, track.bbox,
                        slots=track.reported_slots, total_slots=track.reported_slots, stream_time=t,
                    )
            return None

        # Hueco ya confirmado: ¿cambió su tamaño de forma estable?
        if track.n_hits < self.min_hits:
            return None
        stable = track.consensus_slots(self.min_hits)
        if stable is None or stable == track.reported_slots:
            return None
        if stable > track.reported_slots:
            self._absorb_covered(track)
            if stable <= track.reported_slots:
                return None
        kind = "product_removed" if stable > track.reported_slots else "restocked"
        delta = abs(stable - track.reported_slots)
        track.reported_slots = stable
        return ShelfEvent(kind, track.id, track.row, track.bbox,
                          slots=delta, total_slots=stable, stream_time=t)

    def _absorb_covered(self, keeper: _Track) -> None:
        """
        Pasa a `keeper` lo ya reportado de los huecos confirmados que quedaron
        dentro de él (se retiró el producto que los separaba) y los elimina.
        Se llama solo ante cambios estables, nunca por un único análisis.
        """
        for other in self._tracks:
            if other is keeper or not other.confirmed or other.reported_slots == 0:
                continue
            if _overlap(other.bbox, keeper.bbox) >= 0.5:
                keeper.reported_slots += other.reported_slots
                other.reported_slots = 0
                other.hits.clear()  # muere en este update sin emitir evento

    def _alive(self, track: _Track) -> bool:
        if not track.hits:
            return False
        if track.confirmed:
            return track.n_misses < self.min_hits
        # Un candidato muere cuando ya no puede llegar a min_hits en la ventana
        recent = list(track.hits)[-(self.window - self.min_hits + 1):]
        return any(recent)

    def _covering_track(self, track: _Track) -> Optional[_Track]:
        """Hueco confirmado y vivo que contiene a `track` (se unieron)."""
        if not track.confirmed:
            return None
        for other in self._tracks:
            if (other is not track and other.confirmed and self._alive(other)
                    and _overlap(track.bbox, other.bbox) >= 0.5):
                return other
        return None
