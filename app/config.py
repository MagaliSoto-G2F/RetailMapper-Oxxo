# =====================
# Rutas y carpetas
# =====================

OUTPUT_FOLDER = "output"

SHELF_MODEL_PATH = "models/shelf-model.pt"
PRODUCT_MODEL_PATH = "models/product-model.pt"

# =====================
# Parámetros de detección
# =====================

SHELF_CONF_THRESHOLD = 0.4
PRODUCT_CONF_THRESHOLD = 0.3

# Un hueco en la fila se considera espacio vacío si su ancho es al menos
# este múltiplo del ancho medio (mediana) de los productos de esa fila
EMPTY_SPACE_MIN_WIDTH_RATIO = 1.0

# Guardar una copia de cada imagen procesada con las detecciones dibujadas
SAVE_ANNOTATED_IMAGE = True

# =====================
# Visualización (BGR)
# =====================

COLOR_SHELF = (255, 0, 0)     # Azul
COLOR_PRODUCT = (0, 255, 0)   # Verde
COLOR_EMPTY_SPACE = (0, 0, 255)  # Rojo
