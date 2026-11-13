FROM python:3.12-slim-bookworm

# ffmpeg: WAV conversion, Whisper decoding, and the shared libraries torchcodec needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Fully pinned, so pip does not have to resolve anything.
# The torch wheel from PyPI bundles its own CUDA runtime: no CUDA base image is needed,
# only an NVIDIA driver on the GPU host (see README, "Deploy on ECS").
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "main.py"]
