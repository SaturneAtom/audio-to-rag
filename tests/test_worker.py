import json
from unittest.mock import MagicMock

import pytest

import worker
from config import STATUS_FAILED, STATUS_PROCESSED

VALID_BODY = {"userId": "user-1", "reunionId": "meeting-1", "s3Audio": "user-1/meeting-1/audio.mp3"}


def _message(body):
    return {"ReceiptHandle": "rh", "Body": json.dumps(body)}


def test_valid_message_is_accepted():
    assert worker._validate_message_fields(VALID_BODY) == (
        "user-1", "meeting-1", "user-1/meeting-1/audio.mp3"
    )


def test_missing_field_raises_key_error():
    with pytest.raises(KeyError):
        worker._validate_message_fields({"userId": "user-1", "reunionId": "meeting-1"})


@pytest.mark.parametrize("override", [
    {"userId": 42},
    {"reunionId": "../../etc/passwd"},
    {"userId": "a" * 129},
    {"s3Audio": "audio file;rm -rf.mp3"},
])
def test_invalid_fields_raise_value_error(override):
    with pytest.raises(ValueError):
        worker._validate_message_fields({**VALID_BODY, **override})


def test_success_status_value_is_unchanged():
    assert STATUS_PROCESSED == "TRAITÉ_INTELLIGENT"


def test_malformed_message_is_discarded_without_status_write(monkeypatch):
    update = MagicMock()
    monkeypatch.setattr(worker, "safe_dynamodb_update", update)

    assert worker.process_message({"ReceiptHandle": "rh", "Body": "not json"}) is True
    update.assert_not_called()


def test_pipeline_failure_returns_false_and_writes_failed_status(monkeypatch):
    update = MagicMock()
    monkeypatch.setattr(worker, "safe_dynamodb_update", update)
    monkeypatch.setattr(worker, "safe_s3_download", MagicMock())
    monkeypatch.setattr(worker, "process_audio_intelligent", MagicMock(side_effect=RuntimeError("boom")))

    assert worker.process_message(_message(VALID_BODY)) is False

    table, key, expression, values = update.call_args.args
    assert key == {"userId": "user-1", "reunionId": "meeting-1"}
    assert expression == "SET etat = :s"
    assert values == {":s": STATUS_FAILED}


def test_failed_status_write_error_does_not_mask_the_failure(monkeypatch):
    monkeypatch.setattr(worker, "safe_dynamodb_update", MagicMock(side_effect=RuntimeError("dynamo down")))
    monkeypatch.setattr(worker, "safe_s3_download", MagicMock())
    monkeypatch.setattr(worker, "process_audio_intelligent", MagicMock(side_effect=RuntimeError("boom")))

    assert worker.process_message(_message(VALID_BODY)) is False


def test_success_writes_processed_status_and_output_keys(monkeypatch):
    update = MagicMock()
    monkeypatch.setattr(worker, "safe_dynamodb_update", update)
    monkeypatch.setattr(worker, "safe_s3_download", MagicMock())
    monkeypatch.setattr(worker, "process_audio_intelligent",
                        MagicMock(return_value=([object()], "chunks.json", "index.faiss")))

    assert worker.process_message(_message(VALID_BODY)) is True

    table, key, expression, values = update.call_args.args
    assert expression == "SET etat = :s, s3Chunks = :c, processedAt = :p, s3Index = :f"
    assert values[":s"] == STATUS_PROCESSED
    assert values[":c"] == "chunks.json"
    assert values[":f"] == "index.faiss"


@pytest.mark.parametrize("success", [True, False])
def test_one_shot_poll_reports_message_outcome(monkeypatch, success):
    worker.sqs_client.receive_message.return_value = {"Messages": [_message(VALID_BODY)]}
    monkeypatch.setattr(worker, "process_message", lambda message: success)

    assert worker.poll_sqs() is success
    assert worker.sqs_client.delete_message.called is success
