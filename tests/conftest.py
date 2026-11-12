"""Test setup: dummy environment and stubs for the heavy runtime dependencies.

The tests only exercise pure logic, so torch, whisper, pyannote, librosa and
faiss are replaced by lightweight stand-ins, and every AWS client is replaced
by a mock so that no test can reach the network.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.update({
    "S3_BUCKET_NAME": "test-bucket",
    "DYNAMODB_TABLE_NAME": "test-table",
    "SQS_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/000000000000/test-queue",
    "UNIQUE_MEETINGS_TABLE_NAME": "test-unique-meetings",
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "ONE_SHOT_MODE": "true",
    "NOVA_DIRECT_CHUNKING": "false",
})


class FakeIndexFlatIP:
    """Minimal stand-in for faiss.IndexFlatIP that records the added rows."""

    def __init__(self, dimension):
        self.d = dimension
        self.rows = np.zeros((0, dimension), dtype="float32")

    def add(self, matrix):
        self.rows = np.vstack([self.rows, matrix])

    @property
    def ntotal(self):
        return self.rows.shape[0]


def _install_stubs():
    torch = MagicMock(name="torch")
    torch.cuda.is_available.return_value = False

    faiss = types.ModuleType("faiss")
    faiss.IndexFlatIP = FakeIndexFlatIP
    faiss.normalize_L2 = lambda matrix: None
    faiss.write_index = MagicMock(name="faiss.write_index")

    pyannote = types.ModuleType("pyannote")
    pyannote_audio = types.ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = MagicMock(name="Pipeline")
    pyannote.audio = pyannote_audio

    sys.modules.update({
        "torch": torch,
        "faiss": faiss,
        "whisper": MagicMock(name="whisper"),
        "librosa": MagicMock(name="librosa"),
        "pyannote": pyannote,
        "pyannote.audio": pyannote_audio,
    })


_install_stubs()

import aws_utils  # noqa: E402
import chunking  # noqa: E402
import embeddings  # noqa: E402
import nova  # noqa: E402
import worker  # noqa: E402

FAKE_EMBEDDING = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def no_aws(monkeypatch):
    """Replace every AWS client with a mock and make retries instantaneous."""
    monkeypatch.setattr(aws_utils, "s3_client", MagicMock(name="s3_client"))
    monkeypatch.setattr(aws_utils, "sqs_client", MagicMock(name="sqs_client"))
    monkeypatch.setattr(aws_utils, "dynamodb", MagicMock(name="dynamodb"))
    monkeypatch.setattr(worker, "sqs_client", MagicMock(name="sqs_client"))
    monkeypatch.setattr(worker, "dynamodb", MagicMock(name="dynamodb"))
    monkeypatch.setattr(nova, "bedrock_client", MagicMock(name="bedrock_client"))
    monkeypatch.setattr(embeddings, "bedrock_client", MagicMock(name="bedrock_client"))
    monkeypatch.setattr(aws_utils.time, "sleep", lambda seconds: None)


@pytest.fixture
def fake_embedding(monkeypatch):
    """Make chunk creation return a fixed embedding instead of calling Titan."""
    monkeypatch.setattr(chunking, "generate_titan_embedding", lambda text: FAKE_EMBEDDING.copy())
    return FAKE_EMBEDDING


def make_segment(start, end, text, speaker="Speaker_A1B2"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


@pytest.fixture
def segment():
    return make_segment
