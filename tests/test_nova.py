import json

import pytest

import nova


def _nova_reply(text):
    return {"output": {"message": {"content": [{"text": text}]}}}


def test_truncated_json_is_detected():
    assert nova.is_json_incomplete('{"chunks": [{"theme": "budget"},') is True
    assert nova.is_json_incomplete('{"chunks": [{"theme":') is True
    assert nova.is_json_incomplete('{"chunks": []}') is False
    assert nova.is_json_incomplete('```json\n{"chunks": []}\n```') is False


def test_parse_strips_markdown_fences():
    assert nova._parse_nova_json('```json\n{"theme_principal": "budget"}\n```') == {"theme_principal": "budget"}
    assert nova._parse_nova_json('{"theme_principal": "budget"}') == {"theme_principal": "budget"}


def test_parse_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        nova._parse_nova_json("Voici le JSON demandé")


def test_pagination_concatenates_continuations(monkeypatch):
    monkeypatch.setattr(nova.time, "sleep", lambda seconds: None)
    nova.bedrock_client.converse.side_effect = [
        _nova_reply('{"themes_principaux": [{"theme": "budget"},'),
        _nova_reply('{"theme": "planning"}]}'),
    ]

    response = nova.get_nova_response_with_pagination("prompt")

    assert nova._parse_nova_json(response) == {
        "themes_principaux": [{"theme": "budget"}, {"theme": "planning"}]
    }
    assert nova.bedrock_client.converse.call_count == 2


def test_pagination_stops_after_max_continuations(monkeypatch):
    monkeypatch.setattr(nova.time, "sleep", lambda seconds: None)
    nova.bedrock_client.converse.return_value = _nova_reply('{"a": [')

    nova.get_nova_response_with_pagination("prompt")

    assert nova.bedrock_client.converse.call_count == 1 + nova.MAX_CONTINUATIONS


def test_cleaned_text_is_redistributed_without_losing_words(segment):
    segments = [segment(0, 1, "euh bonjour à tous"), segment(1, 2, "on commence euh la réunion")]

    cleaned = nova.apply_cleaned_text_to_segments(segments, "Bonjour à tous, on commence la réunion.")

    assert " ".join(s["text"] for s in cleaned) == "Bonjour à tous, on commence la réunion."
    assert cleaned[0]["original_text"] == "euh bonjour à tous"
    assert [(s["start"], s["end"]) for s in cleaned] == [(0, 1), (1, 2)]


def test_cleaned_text_is_ignored_when_it_diverges_too_much(segment):
    segments = [segment(0, 1, "bonjour à tous et bienvenue dans cette réunion")]

    assert nova.apply_cleaned_text_to_segments(segments, "Ok.") is segments


def test_segment_groups_must_partition_segments_in_order():
    assert nova._validate_segment_groups([[2, 3], [1, 0]], 4) == [[0, 1], [2, 3]]
    for bad_groups in ([[0, 1]], [[0, 1], [1, 2]], [[0, 2], [1, 3]], [[0, 1, 2, 9]]):
        with pytest.raises(RuntimeError):
            nova._validate_segment_groups(bad_groups, 4)


def test_direct_chunking_is_off_by_default(monkeypatch, fake_embedding, segment):
    monkeypatch.setattr(nova, "create_semantic_chunks_with_nova", lambda segments: pytest.fail("must not be called"))

    chunks = nova._create_base_chunks([segment(0, 1, "bonjour à tous.")])

    assert len(chunks) == 1


def test_direct_chunking_falls_back_to_rules_on_failure(monkeypatch, fake_embedding, segment):
    monkeypatch.setattr(nova, "NOVA_DIRECT_CHUNKING", True)
    nova.bedrock_client.converse.return_value = _nova_reply('{"chunks": [{"segment_indices": [0], "theme": "x"}]}')
    segments = [segment(0, 1, "bonjour à tous."), segment(1, 2, "on commence.")]

    chunks = nova._create_base_chunks(segments)

    assert len(chunks) == 1
    assert chunks[0].reasoning.startswith("Chunk final")


def test_direct_chunking_uses_nova_groups_when_valid(monkeypatch, fake_embedding, segment):
    monkeypatch.setattr(nova, "NOVA_DIRECT_CHUNKING", True)
    nova.bedrock_client.converse.return_value = _nova_reply(json.dumps({"chunks": [
        {"segment_indices": [2], "theme": "planning"},
        {"segment_indices": [0, 1], "theme": "budget"},
    ]}))
    segments = [segment(0, 1, "un."), segment(1, 2, "deux."), segment(2, 3, "trois.")]

    chunks = nova._create_base_chunks(segments)

    assert [(c.theme, c.start_time, c.end_time) for c in chunks] == [("budget", 0, 2), ("planning", 2, 3)]
