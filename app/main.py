import logging
import os
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import OUTPUT_FOLDER, SAVE_ANNOTATED_IMAGE
from app.core.model_loader import load_models
from app.services.detection_pipeline import detect_shelf_image
from app.utils.draw import draw_shelves, draw_products, draw_empty_spaces


def configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"

    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


configure_logging()

app = FastAPI(title="Retail Mapper API")


@app.on_event("startup")
def startup_event():
    logging.getLogger(__name__).info("Loading models at startup")
    load_models()


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        data = np.frombuffer(await image.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "Invalid image file"}
            )

        shelves, products, empty_spaces = detect_shelf_image(img)

        response = {
            "status": "ok",
            "shelves_count": len(shelves),
            "products_count": len(products),
            "empty_spaces_count": len(empty_spaces),
            "shelves": shelves,
            "products": products,
            "empty_spaces": empty_spaces,
        }

        if SAVE_ANNOTATED_IMAGE:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            annotated = draw_shelves(img.copy(), shelves)
            annotated = draw_products(annotated, products)
            annotated = draw_empty_spaces(annotated, empty_spaces)
            out_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(out_path, annotated)
            response["annotated_image"] = out_path

        return response

    except Exception as e:
        logging.getLogger(__name__).exception("Error in /detect endpoint")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )
