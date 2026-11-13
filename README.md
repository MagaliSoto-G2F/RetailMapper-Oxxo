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
  "shelves": [{ "bbox": [0, 410, 899, 470], "conf": 0.91 }],
  "products": [{ "bbox": [57, 209, 181, 417], "conf": 0.89, "row": 3, "col": 1, "id": 1 }],
  "empty_spaces": [{ "bbox": [0, 800, 225, 1085], "row": 1, "after_col": null, "width_px": 225, "estimated_slots": 2, "id": 1 }],
  "annotated_image": "output/<uuid>.jpg"
}
```

- `bbox` es `[x1, y1, x2, y2]` en píxeles.
- `row` cuenta los niveles de repisa desde abajo (1 = el más bajo).
- `col` cuenta los productos de izquierda a derecha dentro de cada fila.
- `empty_spaces`: huecos de cada fila. `after_col` es la columna del producto
  a la izquierda del hueco (`null` si está en el borde izquierdo o la fila
  está vacía) y `estimated_slots` es cuántos productos entrarían.

## Detección de espacios vacíos

Es geométrica, no usa un modelo extra (`app/localization/find_empty_spaces.py`):

1. Para cada repisa se toman los intervalos horizontales `[x1, x2]` de sus
   productos y se unen los que se superponen (productos apilados = un bloque).
2. Se miden los huecos entre el borde izquierdo de la repisa, cada bloque y
   el borde derecho.
3. Un hueco es espacio vacío si su ancho es ≥ `EMPTY_SPACE_MIN_WIDTH_RATIO`
   × la mediana del ancho de los productos de esa fila.
4. Una repisa sin productos se reporta completa como espacio vacío.

En la imagen anotada los espacios vacíos se dibujan en rojo.

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
