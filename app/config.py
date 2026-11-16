# =====================
# Rutas y carpetas
# =====================

OUTPUT_FOLDER = "output"

SHELF_MODEL_PATH = "models/shelf-model.pt"
PRODUCT_MODEL_PATH = "models/product-model.pt"

# =====================
# Parámetros de detección
# =====================

# Bajo a propósito: en la cámara RTSP el nivel de arriba sale con conf
# 0.30–0.55 y con 0.4 aparecía y desaparecía entre frames. Los duplicados que
# entran por el umbral bajo se eliminan con supresión vertical (ShelfDetector).
SHELF_CONF_THRESHOLD = 0.25
PRODUCT_CONF_THRESHOLD = 0.3

# Un hueco en la fila es candidato a espacio vacío si su ancho es al menos
# este múltiplo del ancho medio (mediana) de los productos de esa fila.
# Es permisivo a propósito: la verificación de abajo filtra los falsos.
EMPTY_SPACE_MIN_WIDTH_RATIO = 0.7

# Verificación de candidatos
# - Detecciones de producto con conf entre este valor y PRODUCT_CONF_THRESHOLD
#   no se reportan como productos, pero anulan un hueco si caen dentro.
EMPTY_SPACE_LOW_CONF_THRESHOLD = 0.15
# - Un hueco se descarta si su densidad de bordes (Canny) supera este valor:
#   tiene logos/texto, parece un producto no detectado y no fondo vacío
# Calibrado con imgGondola2: huecos reales ≈ 0.007, productos 0.08–0.25
EMPTY_SPACE_MAX_EDGE_DENSITY = 0.04

# Guardar una copia de cada imagen procesada con las detecciones dibujadas
SAVE_ANNOTATED_IMAGE = True

# =====================
# Visualización (BGR)
# =====================

COLOR_SHELF = (255, 0, 0)     # Azul
COLOR_PRODUCT = (0, 255, 0)   # Verde
COLOR_EMPTY_SPACE = (0, 0, 255)  # Rojo

# =====================
# Stream de video (app/stream)
# =====================

# Cada cuántos segundos se analiza un frame (la inferencia en CPU tarda ~0.5 s)
STREAM_PROCESS_INTERVAL_SEC = 1.0
# Consenso temporal: un hueco se confirma si aparece en al menos
# STREAM_CONSENSUS_MIN_HITS de los últimos STREAM_CONSENSUS_WINDOW análisis,
# y se da por repuesto si falta en esa misma cantidad. Filtra personas que
# pasan, reflejos y frames desenfocados. Latencia de alerta ≈ MIN_HITS × INTERVAL.
STREAM_CONSENSUS_WINDOW = 8
STREAM_CONSENSUS_MIN_HITS = 6
# Dos huecos de análisis distintos son el mismo si se superponen al menos esto
# (producto de la superposición horizontal y vertical, relativas al menor)
STREAM_MATCH_MIN_OVERLAP = 0.25
# Capturas de cada evento (producto retirado / repuesto)
STREAM_EVENTS_FOLDER = "output/stream_events"
