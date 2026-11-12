import numpy as np
import pytest
from botocore.exceptions import ClientError

import aws_utils
import embeddings
from models import Chunk


def _chunk(chunk_id, embedding):
    return Chunk(id=chunk_id, start_time=0.0, end_time=1.0, text="texte", embedding=embedding)


def _client_error():
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Operation")


def test_faiss_index_has_one_row_per_chunk_in_order():
    chunks = [_chunk(i + 1, np.array([float(i), 1.0], dtype=np.float32)) for i in range(3)]

    index = embeddings.create_faiss_index(chunks)

    assert index.ntotal == len(chunks)
    assert [row[0] for row in index.rows] == [0.0, 1.0, 2.0]


def test_faiss_index_refuses_chunks_without_embedding():
    chunks = [_chunk(1, np.ones(2, dtype=np.float32)), _chunk(2, None)]

    with pytest.raises(RuntimeError, match="no embedding"):
        embeddings.create_faiss_index(chunks)


def test_with_retries_retries_then_returns_result():
    outcomes = [_client_error(), _client_error(), "ok"]

    def operation():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert aws_utils.with_retries(operation, "test operation", retries=3) == "ok"
    assert outcomes == []


def test_with_retries_reraises_after_last_attempt():
    calls = []

    def operation():
        calls.append(1)
        raise _client_error()

    with pytest.raises(ClientError):
        aws_utils.with_retries(operation, "test operation", retries=3)
    assert len(calls) == 3


def test_s3_upload_encodes_text_and_reports_success():
    assert aws_utils.safe_s3_upload("bucket", "key", "données") is True

    aws_utils.s3_client.put_object.assert_called_once_with(
        Bucket="bucket", Key="key", Body="données".encode("utf-8")
    )
