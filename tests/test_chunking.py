import numpy as np

import chunking
from config import MAX_WORDS_PER_CHUNK, MIN_WORDS_PER_CHUNK
from models import Chunk


def words(count, ending=""):
    return " ".join(["mot"] * count) + ending


def test_chunk_closes_when_max_words_is_reached(fake_embedding, segment):
    half = MAX_WORDS_PER_CHUNK // 2
    segments = [segment(i, i + 1, words(half)) for i in range(4)]

    chunks = chunking.create_rule_based_chunks(segments)

    assert [chunking.count_words(c.text.split("]: ")[1]) for c in chunks] == [2 * half, 2 * half]
    assert [c.id for c in chunks] == [1, 2]


def test_chunk_closes_on_sentence_end_followed_by_speaker_change(fake_embedding, segment):
    segments = [
        segment(0.0, 5.0, words(MIN_WORDS_PER_CHUNK, "."), speaker="Speaker_A"),
        segment(5.1, 9.0, words(10, "."), speaker="Speaker_B"),
    ]

    chunks = chunking.create_rule_based_chunks(segments)

    assert len(chunks) == 2
    assert chunks[0].end_time == 5.0
    assert chunks[0].nova_chunking["speakers_impliques"] == ["Speaker_A"]


def test_chunk_stays_open_without_pause_or_sentence_end(fake_embedding, segment):
    segments = [
        segment(0.0, 5.0, words(MIN_WORDS_PER_CHUNK, ".")),
        segment(5.1, 9.0, words(MIN_WORDS_PER_CHUNK)),
        segment(9.1, 12.0, words(10)),
    ]

    chunks = chunking.create_rule_based_chunks(segments)

    assert len(chunks) == 1
    assert (chunks[0].start_time, chunks[0].end_time) == (0.0, 12.0)


def test_chunk_closes_on_sentence_end_followed_by_pause(fake_embedding, segment):
    pause = chunking.PAUSE_BOUNDARY_SECONDS + 0.5
    segments = [
        segment(0.0, 5.0, words(MIN_WORDS_PER_CHUNK, ".")),
        segment(5.0 + pause, 9.0, words(10)),
    ]

    assert len(chunking.create_rule_based_chunks(segments)) == 2


def test_oversized_chunk_is_split_on_segment_boundaries(fake_embedding, segment):
    segments = [segment(i, i + 1, words(60)) for i in range(4)]
    parent = chunking.create_chunk_from_segments(segments, 7)
    parent.theme = "budget"

    sub_chunks = chunking.divide_chunk_intelligently(parent, segments, max_words=120)

    assert [(c.start_time, c.end_time) for c in sub_chunks] == [(0, 2), (2, 4)]
    assert all(c.theme == "budget" for c in sub_chunks)
    assert [c.nova_chunking["sub_chunk_index"] for c in sub_chunks] == [1, 2]
    assert all(c.nova_chunking["parent_chunk_id"] == 7 for c in sub_chunks)


def test_word_split_prefers_sentence_endings(fake_embedding):
    text = words(6, ".") + " " + words(6)
    parent = Chunk(id=1, start_time=0.0, end_time=10.0, text=text)

    sub_chunks = chunking.divide_chunk_by_words(parent, max_words=10)

    assert [chunking.count_words(c.text) for c in sub_chunks] == [6, 6]
    assert sub_chunks[0].text.endswith(".")


def _short_chunk(chunk_id, embedding):
    return Chunk(
        id=chunk_id, start_time=chunk_id, end_time=chunk_id + 1, text=words(5),
        embedding=np.array(embedding, dtype=np.float32),
        speakers_info=[{"name": "Speaker_A", "speaking_time": 1.0, "dominance_score": 1.0, "turns": 1}],
        nova_chunking={"size_adjustment": "too_short"},
    )


def test_short_chunk_merges_into_similar_neighbour(fake_embedding):
    previous = _short_chunk(1, [1.0, 0.0, 0.0])
    previous.nova_chunking = {"size_adjustment": "optimal"}
    short = _short_chunk(2, [1.0, 0.0, 0.0])

    merged = chunking.merge_short_chunks_intelligently([previous, short])

    assert len(merged) == 1
    assert (merged[0].start_time, merged[0].end_time) == (1, 3)
    assert merged[0].nova_chunking["merged_with"] == 2
    assert merged[0].speakers_info[0]["turns"] == 2


def test_short_chunk_is_kept_when_neighbour_is_dissimilar(fake_embedding):
    previous = _short_chunk(1, [1.0, 0.0, 0.0])
    previous.nova_chunking = {"size_adjustment": "optimal"}
    short = _short_chunk(2, [0.0, 1.0, 0.0])

    assert len(chunking.merge_short_chunks_intelligently([previous, short])) == 2
