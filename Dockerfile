FROM python:3.13-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# CPU-only torch keeps the image small on linux/amd64
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY src ./src
RUN pip install .
COPY web ./web
COPY observability ./observability
COPY eval ./eval
EXPOSE 8000
CMD ["uvicorn", "finsearch.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
