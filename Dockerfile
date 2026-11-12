ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

# ffmpeg is required by the WAV conversion step and by Whisper;
# git is required to install Whisper from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/openai/whisper.git

COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "main.py"]
