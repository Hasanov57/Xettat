# Xəttat API — CPU image.
#   docker build -t xettat .
#   docker run -p 8000:8000 -v /path/to/checkpoints:/models \
#     -e MODEL_FAST=/models/bilstm_v1/best.pt -e MODEL_ACCURATE=/models/byt5_v1/best.pt xettat
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY serve/requirements.txt serve/requirements.txt
RUN pip install -r serve/requirements.txt transformers==4.52.4
COPY pyproject.toml ./
COPY src ./src
COPY serve ./serve
RUN pip install --no-deps -e .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
