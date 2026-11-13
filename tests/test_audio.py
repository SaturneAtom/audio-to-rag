import re
import subprocess
import uuid

import pytest

import audio


def test_convert_to_wav_rejects_paths_outside_tmp():
    with pytest.raises(ValueError):
        audio.convert_to_wav("/etc/passwd")


def test_convert_to_wav_rejects_traversal_out_of_tmp():
    with pytest.raises(ValueError):
        audio.convert_to_wav("/tmp/../etc/passwd")


def test_convert_to_wav_raises_when_ffmpeg_fails(monkeypatch):
    def failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Invalid data found")

    monkeypatch.setattr(audio.subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        audio.convert_to_wav(f"/tmp/{uuid.uuid4().hex}.mp3")


def test_merge_assigns_speaker_with_longest_overlap(segment):
    transcript = [segment(0.0, 10.0, "bonjour", speaker="placeholder")]
    diarization = [
        {"start": 0.0, "end": 3.0, "speaker": "Speaker_A"},
        {"start": 3.0, "end": 10.0, "speaker": "Speaker_B"},
    ]

    merged = audio.merge_transcription_diarization(transcript, diarization)

    assert merged[0]["speaker"] == "Speaker_B"


def test_merge_keeps_speaker_without_overlap_or_diarization(segment):
    transcript = [segment(0.0, 2.0, "bonjour", speaker="placeholder")]
    far_away = [{"start": 50.0, "end": 60.0, "speaker": "Speaker_A"}]

    assert audio.merge_transcription_diarization(transcript, far_away)[0]["speaker"] == "placeholder"
    assert audio.merge_transcription_diarization(transcript, None) == transcript


def test_speaker_mapping_is_anonymous_and_unique():
    originals = [f"SPEAKER_{i:02d}" for i in range(20)]

    mapping = audio.create_speaker_mapping(originals)

    assert set(mapping) == set(originals)
    assert len(set(mapping.values())) == len(originals)
    assert all(re.fullmatch(r"Speaker_[A-Z]\d[A-Z]\d", new_id) for new_id in mapping.values())


class _FakeAnnotation:
    def itertracks(self, yield_label=False):
        return iter(())


class _FakeDiarizeOutput:
    """Shape of pyannote.audio 4.x pipeline output."""

    def __init__(self, annotation):
        self.speaker_diarization = annotation


def test_speaker_annotation_unwraps_pyannote_4_output():
    annotation = _FakeAnnotation()
    assert audio.speaker_annotation(_FakeDiarizeOutput(annotation)) is annotation


def test_speaker_annotation_keeps_legacy_annotation():
    annotation = _FakeAnnotation()
    assert audio.speaker_annotation(annotation) is annotation
