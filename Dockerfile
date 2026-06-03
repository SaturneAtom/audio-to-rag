# Dockerfile

ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "main.py"]

