"""Smoke test for the Docker image: real dependencies, no AWS call, no GPU required.

Checks that the pinned stack installs and works together:
- every heavy dependency imports, and the application modules import on top of them;
- ffmpeg converts audio and torchcodec (pyannote's decoder) can decode it;
- Whisper "tiny" transcribes a short clip end to end;
- FAISS builds and queries an index.

Run inside the image:
    docker run --rm audio-to-rag:local python docker/smoke_test.py
"""

import os
import subprocess
import sys
import tempfile

# Dummy configuration so that config.py can be imported without a real deployment.
# boto3 clients are created at import time but make no network call until used.
os.environ.setdefault("S3_BUCKET_NAME", "smoke-test-bucket")
os.environ.setdefault("DYNAMODB_TABLE_NAME", "smoke-test-table")
os.environ.setdefault("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/000000000000/smoke-test")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "smoke-test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "smoke-test")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step(name):
    print(f"\n== {name}", flush=True)


step("Versions")
import faiss  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pyannote.audio  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import torchcodec  # noqa: E402
import whisper  # noqa: E402

for module in (torch, torchaudio, torchcodec, pyannote.audio, whisper, librosa, faiss, np):
    print(f"{module.__name__:16} {getattr(module, '__version__', '?')}")
print(f"{'CUDA build':16} {torch.version.cuda}")
print(f"{'CUDA available':16} {torch.cuda.is_available()}")

step("Application modules")
from pyannote.audio import Pipeline  # noqa: E402,F401
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput  # noqa: E402,F401

import audio  # noqa: E402,F401
import chunking  # noqa: E402,F401
import embeddings  # noqa: E402,F401
import nova  # noqa: E402,F401
import storage  # noqa: E402,F401
import worker  # noqa: E402,F401

print("audio, chunking, embeddings, nova, storage, worker: imported")

with tempfile.TemporaryDirectory() as tmp:
    step("ffmpeg")
    source = os.path.join(tmp, "tone.mp3")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", source],
        check=True,
    )
    wav = os.path.join(tmp, "tone.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", source, "-ar", "16000", "-ac", "1", wav], check=True)
    print("mp3 -> 16 kHz mono wav: ok")

    step("torchcodec")
    from torchcodec.decoders import AudioDecoder  # noqa: E402

    samples = AudioDecoder(wav).get_all_samples()
    print(f"decoded {samples.data.shape[-1]} samples at {samples.sample_rate} Hz")
    assert samples.sample_rate == 16000

    step("Whisper tiny")
    model = whisper.load_model("tiny", device="cpu")
    result = model.transcribe(wav, language="fr", fp16=False)
    print(f"transcribed {len(result['segments'])} segment(s)")

step("FAISS")
vectors = np.random.default_rng(0).random((8, 512), dtype=np.float32)
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(512)
index.add(vectors)
_, ids = index.search(vectors[:1], 1)
assert ids[0][0] == 0
print("index built and queried: ok")

print("\nSMOKE TEST PASSED")
