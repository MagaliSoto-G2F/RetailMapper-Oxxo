import os
import logging
from typing import Optional

from huggingface_hub import hf_hub_download, login


logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    filename: str,
    local_path: str
) -> str:
    """
    Descarga un archivo de modelo desde Hugging Face Hub si no está disponible localmente.

    Si el repositorio es privado, se intenta autenticar usando la
    variable de entorno HUGGINGFACE_HUB_TOKEN.

    Parámetros
    ----------
    repo_id : str
        ID del repositorio en Hugging Face (ej.: "org/model-name").
    filename : str
        Nombre del archivo dentro del repositorio.
    local_path : str
        Ruta local esperada para el archivo descargado.

    Retorna
    -------
    str
        Ruta al archivo del modelo descargado (o ya existente).

    Lanza
    -----
    ValueError
        Si los parámetros requeridos no son válidos.
    RuntimeError
        Si la descarga falla.
    """

    # ---- Validación de parámetros ----
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string.")

    if not filename or not isinstance(filename, str):
        raise ValueError("filename must be a non-empty string.")

    if not local_path or not isinstance(local_path, str):
        raise ValueError("local_path must be a non-empty string.")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # ---- Omitir la descarga si el archivo ya existe ----
    if os.path.exists(local_path):
        logger.info(
            "Model already available locally",
            extra={"local_path": local_path}
        )
        return local_path

    hf_token: Optional[str] = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # ---- Autenticar si hay token disponible ----
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Authenticated with Hugging Face Hub")
        except Exception:
            logger.exception("Failed to authenticate with Hugging Face Hub")

    logger.info(
        "Downloading model from Hugging Face",
        extra={
            "repo_id": repo_id,
            "model_filename": filename
        }
    )

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token
        )

        logger.info(
            "Model downloaded successfully",
            extra={"downloaded_path": model_path}
        )

        return model_path

    except Exception as e:
        logger.exception(
            "Model download failed",
            extra={
                "repo_id": repo_id,
                "model_filename": filename
            }
        )
        raise RuntimeError(
            f"Failed to download model '{filename}' from '{repo_id}'"
        ) from e
