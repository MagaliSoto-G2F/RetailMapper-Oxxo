# RetailMapper v.2

API REST (FastAPI + Docker) que detecta repisas y productos en una imagen
de góndola, ubica cada producto por fila y columna y detecta los
**espacios vacíos en la góndola**.

## Pipeline

    Imagen de góndola
       ↓
    Detección de repisas (YOLO)
       ↓
    Detección de productos (YOLO)
       ↓
    Asignación de fila / columna
       ↓
    Detección de espacios vacíos (geométrica)
       ↓
    Respuesta JSON (+ imagen anotada en output/)

Los dos modelos YOLO se descargan desde Hugging Face
(`MagaliSoto/RetailMapper`) al iniciar la API.

## Endpoint

**POST `/detect`** — `multipart/form-data` con un campo `image`.

``` bash
curl -X POST http://localhost:8000/detect -F "image=@input_images_test/imgGondola2.jpeg"
```

Respuesta:

``` json
{
  "status": "ok",
  "shelves_count": 3,
  "products_count": 22,
  "empty_spaces_count": 2,
  "shelves": [{ "bbox": [0, 1, 898, 428], "conf": 0.84 }],
  "products": [{ "bbox": [57, 209, 181, 417], "conf": 0.89, "row": 3, "col": 1, "id": 1 }],
  "empty_spaces": [{ "bbox": [120, 564, 295, 760], "row": 2, "after_col": null, "width_px": 175, "estimated_slots": 2, "edge_density": 0.011, "color_std": 51.85, "id": 1 }],
  "annotated_image": "output/<uuid>.jpg"
}
```

- `bbox` es `[x1, y1, x2, y2]` en píxeles.
- Cada `shelf` es un nivel completo de la góndola (de la tabla de arriba a la
  de abajo), no solo la tabla.
- `row` cuenta los niveles de repisa desde abajo (1 = el más bajo).
- `col` cuenta los productos de izquierda a derecha dentro de cada fila.
- `empty_spaces`: huecos de cada fila. `after_col` es la columna del producto
  a la izquierda del hueco (`null` si está en el borde izquierdo o la fila
  está vacía) y `estimated_slots` es cuántos productos entrarían.
  `edge_density` y `color_std` miden la textura del hueco (sirven para
  calibrar los umbrales).

## Detección de espacios vacíos

Se hace en dos pasos, sin modelos extra.

**1. Candidatos geométricos** (`app/localization/find_empty_spaces.py`)

1. Para cada repisa se toman los intervalos horizontales `[x1, x2]` de sus
   productos y se unen los que se superponen (productos apilados = un bloque).
2. Se miden los huecos entre el borde de la góndola, cada bloque y el otro
   borde. El borde de la góndola se estima con el x mínimo / máximo de los
   productos de todas las filas.
3. Un hueco es candidato si su ancho es ≥ `EMPTY_SPACE_MIN_WIDTH_RATIO`
   × la mediana del ancho de los productos de esa fila. En vertical, el hueco
   cubre la franja donde están los productos de la fila.
4. Una repisa sin productos se reporta completa como candidato.

**2. Verificación con la imagen** (`app/localization/verify_empty_spaces.py`)

1. Los huecos en los extremos de la fila se recortan, buscando desde el
   producto hacia afuera, hasta:
   - el primer parante de la góndola (columnas casi negras), y
   - donde termina el frente del estante (el borde horizontal del canto de
     la tabla o la chapa en la base de la fila; funciona con góndolas de
     cualquier color).

   Así no se reportan como vacíos los parantes, la pared, el pasillo o la
   góndola vecina.
2. Cada columna del hueco se marca como ocupada si la densidad de bordes
   (Canny) en una ventana de medio producto supera
   `EMPTY_SPACE_MAX_EDGE_DENSITY` (logos y texto: producto no detectado), o si
   la cubre una detección de producto con confianza entre
   `EMPTY_SPACE_LOW_CONF_THRESHOLD` y `PRODUCT_CONF_THRESHOLD`.
3. Los tramos libres donde entra otro producto como el de al lado (ancho
   ≥ el producto vecino) son los espacios vacíos. Un hueco puede partirse
   en varios o desaparecer. Esto también descarta la separación normal
   entre productos.

En la imagen anotada los espacios vacíos se dibujan en rojo.

### Evaluar sobre un set de imágenes

``` bash
python -m tools.evaluate_empty_spaces --images input_images_test --out output/eval
```

Genera `performance.png` (separación por textura frente al umbral, resultado
de los candidatos y tiempo de post-proceso por imagen), una `<imagen>_eval.jpg`
por imagen con los huecos confirmados en rojo y los descartados en naranja con
su motivo, y `summary.json` con los números.

## Alternativa: detectar el hueco con un modelo

Otra forma de encontrar faltantes, basada en el
[artículo de Roboflow](https://roboflow.com/blog/retail-store-object-detection):
en lugar de detectar productos y deducir los huecos por geometría, se usa un
modelo entrenado para detectar **directamente el espacio vacío**.

| | Pipeline actual | Modelo de huecos |
|---|---|---|
| Modelos | 2 (repisas + productos) | 1 (huecos) |
| Post-proceso | filas, columnas, candidatos, verificación por textura y parantes | ninguno |
| Tiempo por imagen (CPU) | 0,52–0,85 s | 0,15–0,25 s (≈ 3× más rápido) |
| Qué devuelve | hueco con fila, columna y cuántos productos entran | solo la caja del hueco |
| Ajuste a una góndola nueva | umbrales | reentrenar con imágenes etiquetadas |

Además de los huecos, el artículo clasifica la góndola según cuánta área de
la imagen cubren: 0 detecciones = STOCKED, < 20 % = PARTIALLY STOCKED,
≥ 20 % = EMPTY.

### Probarlo

La herramienta `tools/roboflow_empty_shelf.py` corre el modelo, lo compara
con el pipeline actual y no modifica el pipeline. Acepta tres orígenes de
modelo:

``` bash
# 1. Modelo YOLO local. Probado con el modelo público de Hugging Face
#    akul-29/Retail-Shelf-Gap-Detection_Model (YOLOv8, 2095 imágenes de góndolas)
python -m tools.roboflow_empty_shelf --backend yolo --weights best.pt --confidence 0.25

# 2. Modelo hospedado en Roboflow (p. ej. un fork entrenado de
#    policenta/empty-shelf-detection, como en el artículo)
set ROBOFLOW_API_KEY=...
python -m tools.roboflow_empty_shelf --model-id <proyecto>/<versión>

# 3. Pesos RF-DETR locales, descargados después de entrenar en Roboflow
#    (pip install rfdetr)
python -m tools.roboflow_empty_shelf --backend rfdetr --weights checkpoint_best_total.pth
```

Los pesos `.pt` de terceros son un pickle, y cargarlos con `YOLO(...)`
ejecutaría cualquier código que traigan. Por eso el backend `yolo` los lee
con `torch.load(weights_only=True)`, que solo acepta las clases de torch y
ultralytics de una red de detección, y copia los pesos a un modelo armado
desde cero.

En `--out` (por defecto `output/roboflow_compare`) quedan:
- una `<imagen>_compare.jpg`, con el pipeline actual a la izquierda y el
  modelo a la derecha;
- `comparison.json`, con el estado, la cobertura y qué huecos detectaron
  los dos o solo uno;
- las respuestas crudas en `predictions/`. Funcionan como caché: al volver a
  correr no se llama otra vez al modelo; `--no-cache` lo fuerza.

### Resultados con las imágenes de prueba

Con el modelo de Hugging Face (`--confidence 0.25`):

| Imagen | Pipeline actual | Modelo de huecos |
|---|---|---|
| imgGondola1 | 1 hueco real ✔ | 0 (no lo ve) |
| imgGondola2 | 2 huecos reales ✔ | 3: coincide en 1; no ve el hueco grande y marca 2 rendijas entre productos |
| imgGondola3 | 0 ✔ | 2 rendijas entre frascos |
| imgGondola4 | 0 ✔ | 1 rendija |
| img2 (súper) | 1 falso (poste) | 1 hueco oscuro real ✔ |
| img3 | 0 ✔ | 0 ✔ |

El modelo es más rápido y más simple, pero hoy acierta menos en nuestras
góndolas. Se entrenó con góndolas de supermercado llenas: detecta bien los
huecos angostos y oscuros entre productos (img2), pero no los espacios
grandes con la pared de fondo. Por la misma razón, la regla de cobertura da
siempre < 2 % con estas fotos.

Para que esta forma reemplace al pipeline habría que reentrenar el modelo
con fotos propias de góndolas etiquetadas, ya sea en Roboflow o con
ultralytics. Mientras tanto sirve como segunda opinión en góndolas densas,
donde el pipeline actual tiene más falsos positivos.

## Monitoreo desde un stream de video

Además de la API por imagen, se puede monitorear una góndola desde una cámara
y recibir un evento cada vez que se retira (o se repone) un producto
(`app/stream/`).

``` bash
python -m app.stream.monitor --source 0                        # webcam local
python -m app.stream.monitor --source rtsp://user:pass@ip/...  # cámara IP
python -m app.stream.monitor --source video.mp4 --show         # archivo, con ventana en vivo
```

Cómo funciona:

1. Cada `STREAM_PROCESS_INTERVAL_SEC` se analiza el último frame con el mismo
   pipeline que la API (repisas, productos y huecos verificados). En cámaras
   en vivo un hilo lector guarda solo el frame más reciente y reconecta si
   se corta la señal.
2. `EmptySpaceTracker` sigue cada hueco entre análisis (por geometría, no por
   número de fila) y solo lo confirma si aparece en
   `STREAM_CONSENSUS_MIN_HITS` de los últimos `STREAM_CONSENSUS_WINDOW`
   análisis. Una persona que pasa o un frame movido no generan alertas.
3. Los huecos que ya estaban al arrancar (primeros `WINDOW` análisis) son la
   línea base y no alertan.
4. Eventos:
   - `product_removed`: aparece un hueco nuevo o crece uno existente (`slots`
     = cuántos productos se retiraron). Si al retirar un producto se unen dos
     huecos, se informa solo lo nuevo.
   - `restocked`: un hueco desaparece o se achica.

   Con la configuración por defecto un evento tarda unos 6 s en dispararse.
5. Cada evento guarda una captura anotada en `STREAM_EVENTS_FOLDER` y se
   envía al `AlertSink`. Por ahora `LogAlertSink` solo lo deja en el log;
   el envío real de alertas está **pendiente** (`app/stream/alerts.py`).

## Docker

``` bash
docker build -t retail-mapper-api .
docker run -p 8000:8000 -v ${PWD}/output:/app/output retail-mapper-api
```

Documentación interactiva: http://localhost:8000/docs

## Configuración

`app/config.py`: umbrales de confianza de cada detector, sensibilidad de la
detección de espacios vacíos (`EMPTY_SPACE_MIN_WIDTH_RATIO`) y si se guarda
la imagen anotada.

## Estructura del proyecto

    app/
    ├── main.py                     # App FastAPI, endpoint /detect
    ├── config.py
    ├── core/model_loader.py        # Descarga y carga los modelos YOLO una sola vez
    ├── detectors/                  # Detectores YOLO de repisas y productos
    ├── localization/               # Asignación de filas/columnas y espacios vacíos
    ├── services/detection_pipeline.py
    └── utils/                      # Descarga desde Hugging Face, dibujo

## Contacto

**Magali Soto** — sotomagali265@gmail.com
