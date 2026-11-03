FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model is downloaded + quantized at BUILD time and baked into the image.
# This layer is cached, so it only re-runs if requirements.txt or download_model.py change.
COPY download_model.py .
RUN python download_model.py

COPY utils/ ./utils/
COPY app.py .
COPY index.html .

EXPOSE 7860

# Railway injects $PORT (falls back to 7860 elsewhere). Shell form so variables expand at runtime.
CMD gunicorn app:app \
    --workers ${WEB_CONCURRENCY:-2} \
    --bind 0.0.0.0:${PORT:-7860} \
    --timeout 120 \
    --worker-class uvicorn.workers.UvicornWorker
