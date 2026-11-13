FROM python:3.10-slim

WORKDIR /app

# ---- Dependencias del sistema (OpenCV) ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencias de Python ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Aplicación ----
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
