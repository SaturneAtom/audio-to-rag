import re
import logging
import numpy as np
from collections import Counter
from typing import List, Dict, Optional
from config import MIN_WORDS_PER_CHUNK, MAX_WORDS_PER_CHUNK
from models import Chunk
from embeddings import generate_titan_embedding

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    'alors', 'donc', 'mais', 'pour', 'avec', 'dans', 'cette', 'celui', 'celle',
    'leurs', 'nous', 'vous', 'ils', 'elles', 'tout', 'tous', 'toutes', 'très',
    'bien', 'plus', 'moins', 'autre', 'comme', 'même', 'encore', 'aussi', 'peut',
    'être', 'avoir', 'faire', 'dire', 'voir', 'savoir', 'aller', 'venir', 'pouvoir',
    'vouloir', 'devoir', 'prendre', 'donner', 'mettre', 'partir', 'passer', 'temps',
    'année', 'jour', 'moment', 'fois', 'chose', 'façon', 'manière', 'genre', 'type',
    'exemple', 'question', 'réponse', 'problème', 'solution'
}


def count_words(text: str) -> int:
    return len(text.split())


def create_chunk_from_segments(segments: List[Dict], chunk_id: int) -> Chunk:
    """Crée un Chunk à partir d'une liste de segments avec speakers"""
    if not segments:
        raise ValueError("Segments vides")

    start_time = segments[0]["start"]
    end_time = segments[-1]["end"]

    formatted_text_parts = []
    current_speaker = None

    for segment in segments:
        speaker = segment["speaker"]
        text = segment["text"].strip()

        if speaker != current_speaker:
            formatted_text_parts.append(f"\n[{speaker}]: {text}")
            current_speaker = speaker
        else:
            formatted_text_parts.append(f" {text}")

    formatted_text = "".join(formatted_text_parts).strip()
    simple_text = " ".join([seg["text"].strip() for seg in segments])

    speaker_stats = {}
    for segment in segments:
        speaker = segment["speaker"]
        duration = segment["end"] - segment["start"]
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"speaking_time": 0, "turns": 0}
        speaker_stats[speaker]["speaking_time"] += duration
        speaker_stats[speaker]["turns"] += 1

    total_time = sum([stats["speaking_time"] for stats in speaker_stats.values()])
    speakers_info = []
    for speaker, stats in speaker_stats.items():
        dominance_score = stats["speaking_time"] / total_time if total_time > 0 else 0
        speakers_info.append({
            "name": speaker,
            "speaking_time": round(stats["speaking_time"], 2),
            "dominance_score": round(dominance_score, 2),
            "turns": stats["turns"]
        })

    embedding = generate_titan_embedding(simple_text)

    return Chunk(
        id=chunk_id,
        start_time=round(start_time, 2),
        end_time=round(end_time, 2),
        text=formatted_text,
        embedding=embedding,
        speakers_info=speakers_info
    )


def find_chunk_segments(chunk: Chunk, segments: List[Dict]) -> List[Dict]:
    """Retrouve les segments originaux qui correspondent à un chunk via timestamps"""
    return [
        seg for seg in segments
        if seg["start"] < chunk.end_time and seg["end"] > chunk.start_time
    ]


def divide_chunk_intelligently(chunk: Chunk, chunk_segments: List[Dict], max_words: int) -> List[Chunk]:
    """Divise un chunk trop long en respectant les frontières de segments"""
    if not chunk_segments:
        return divide_chunk_by_words(chunk, max_words)

    sub_chunks = []
    current_segments = []
    current_words = 0

    for segment in chunk_segments:
        segment_words = count_words(segment["text"])

        if current_words + segment_words > max_words and current_segments:
            sub_chunk = create_chunk_from_segments(current_segments, len(sub_chunks) + 1)
            sub_chunk.theme = chunk.theme
            sub_chunk.nova_chunking = chunk.nova_chunking.copy()
            sub_chunk.nova_chunking["parent_chunk_id"] = chunk.id
            sub_chunk.nova_chunking["division_reason"] = "too_long"
            sub_chunk.nova_chunking["sub_chunk_index"] = len(sub_chunks) + 1
            sub_chunks.append(sub_chunk)
            current_segments = [segment]
            current_words = segment_words
        else:
            current_segments.append(segment)
            current_words += segment_words

    if current_segments:
        sub_chunk = create_chunk_from_segments(current_segments, len(sub_chunks) + 1)
        sub_chunk.theme = chunk.theme
        sub_chunk.nova_chunking = chunk.nova_chunking.copy()
        sub_chunk.nova_chunking["parent_chunk_id"] = chunk.id
        sub_chunk.nova_chunking["division_reason"] = "too_long"
        sub_chunk.nova_chunking["sub_chunk_index"] = len(sub_chunks) + 1
        sub_chunks.append(sub_chunk)

    return sub_chunks


def divide_chunk_by_words(chunk: Chunk, max_words: int) -> List[Chunk]:
    """Fallback : division brutale d'un chunk par nombre de mots"""
    words = chunk.text.split()
    total_words = len(words)

    if total_words <= max_words:
        return [chunk]

    sub_chunks = []
    current_start = 0

    while current_start < total_words:
        current_end = min(current_start + max_words, total_words)

        if current_end < total_words:
            for i in range(current_end - 1, current_start, -1):
                if words[i].endswith(('.', '!', '?')):
                    current_end = i + 1
                    break

        sub_text = " ".join(words[current_start:current_end])
        sub_chunk = Chunk(
            id=len(sub_chunks) + 1,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=sub_text,
            embedding=generate_titan_embedding(sub_text),
            speakers_info=chunk.speakers_info.copy(),
            theme=chunk.theme,
            nova_chunking=chunk.nova_chunking.copy()
        )
        sub_chunk.nova_chunking["parent_chunk_id"] = chunk.id
        sub_chunk.nova_chunking["division_reason"] = "too_long_fallback"
        sub_chunk.nova_chunking["sub_chunk_index"] = len(sub_chunks) + 1
        sub_chunks.append(sub_chunk)
        current_start = current_end

    return sub_chunks


def calculate_chunk_similarity(chunk1: Chunk, chunk2: Chunk) -> float:
    """Calcule la similarité cosinus entre deux chunks"""
    if chunk1.embedding is None or chunk2.embedding is None:
        return 0.0

    dot_product = np.dot(chunk1.embedding, chunk2.embedding)
    norm1 = np.linalg.norm(chunk1.embedding)
    norm2 = np.linalg.norm(chunk2.embedding)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def merge_chunks(chunk1: Chunk, chunk2: Chunk) -> Chunk:
    """Fusionne deux chunks en conservant les métadonnées et en recalculant l'embedding"""
    merged_text = chunk1.text + " " + chunk2.text
    merged_start = min(chunk1.start_time, chunk2.start_time)
    merged_end = max(chunk1.end_time, chunk2.end_time)

    merged_speakers = chunk1.speakers_info.copy()
    for speaker in chunk2.speakers_info:
        existing = next((s for s in merged_speakers if s["name"] == speaker["name"]), None)
        if existing:
            existing["speaking_time"] += speaker["speaking_time"]
            existing["turns"] += speaker["turns"]
        else:
            merged_speakers.append(speaker)

    total_time = sum([s["speaking_time"] for s in merged_speakers])
    for speaker in merged_speakers:
        speaker["dominance_score"] = speaker["speaking_time"] / total_time if total_time > 0 else 0

    merged_chunk = Chunk(
        id=chunk1.id,
        start_time=merged_start,
        end_time=merged_end,
        text=merged_text,
        embedding=generate_titan_embedding(merged_text),
        speakers_info=merged_speakers,
        theme=chunk1.theme,
        nova_chunking=chunk1.nova_chunking.copy()
    )
    merged_chunk.nova_chunking["merged_with"] = chunk2.id
    merged_chunk.nova_chunking["merge_reason"] = "size_optimization"
    merged_chunk.nova_chunking["original_themes"] = [chunk1.theme, chunk2.theme]
    merged_chunk.nova_chunking["word_count"] = count_words(merged_text)

    return merged_chunk


def merge_short_chunks_intelligently(chunks: List[Chunk]) -> List[Chunk]:
    """Fusionne les chunks trop courts avec leur voisin le plus similaire"""
    if len(chunks) <= 1:
        return chunks

    final_chunks = []
    i = 0

    while i < len(chunks):
        current_chunk = chunks[i]

        if (current_chunk.nova_chunking.get("size_adjustment") == "too_short" and
                count_words(current_chunk.text) < MIN_WORDS_PER_CHUNK):

            best_neighbor = None
            best_similarity = -1
            best_neighbor_idx = -1

            if i > 0:
                prev_chunk = chunks[i - 1]
                if prev_chunk.embedding is not None and current_chunk.embedding is not None:
                    similarity = calculate_chunk_similarity(current_chunk, prev_chunk)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_neighbor = prev_chunk
                        best_neighbor_idx = i - 1

            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                if next_chunk.embedding is not None and current_chunk.embedding is not None:
                    similarity = calculate_chunk_similarity(current_chunk, next_chunk)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_neighbor = next_chunk
                        best_neighbor_idx = i + 1

            if best_similarity > 0.5:
                logger.debug(f"Fusion chunk {current_chunk.id} avec voisin (similarité: {best_similarity:.3f})")
                if best_neighbor_idx < i:
                    final_chunks[-1] = merge_chunks(final_chunks[-1], current_chunk)
                else:
                    final_chunks.append(merge_chunks(current_chunk, best_neighbor))
                    i += 1
            else:
                logger.debug(f"Chunk {current_chunk.id} reste court (pas de voisin similaire)")
                final_chunks.append(current_chunk)
        else:
            final_chunks.append(current_chunk)

        i += 1

    return final_chunks


def _extract_theme_from_text(text: str, chunk_index: int) -> str:
    """Extrait un thème à partir du contenu textuel pour le fallback"""
    words = re.findall(r'\b\w{4,}\b', text.lower())
    meaningful_words = [w for w in words if w not in _STOP_WORDS and len(w) > 3]

    if meaningful_words:
        top_words = [word for word, _ in Counter(meaningful_words).most_common(3)]
        return f"Discussion sur {', '.join(top_words[:2])}"

    return f"Discussion - Segment {chunk_index}"


def create_semantic_chunks_fallback(segments: List[Dict]) -> List[Chunk]:
    """Chunking basique de secours quand Nova n'est pas disponible"""
    logger.info("🔄 Chunking basique de secours avec métadonnées Nova...")

    chunks = []
    current_chunk_segments = []

    for i, segment in enumerate(segments):
        current_chunk_segments.append(segment)
        current_text = " ".join([s["text"] for s in current_chunk_segments])
        current_words = count_words(current_text)

        should_end_chunk = False

        if current_words >= 150:
            should_end_chunk = True
        elif current_words >= 50:
            segment_text = segment["text"].strip()
            if segment_text.endswith(('.', '!', '?')):
                if i + 1 < len(segments):
                    next_segment = segments[i + 1]
                    gap = next_segment["start"] - segment["end"]
                    speaker_change = next_segment["speaker"] != segment["speaker"]
                    if gap > 0.8 or speaker_change:
                        should_end_chunk = True
                else:
                    should_end_chunk = True

        if should_end_chunk and current_chunk_segments:
            chunk = create_chunk_from_segments(current_chunk_segments, len(chunks) + 1)
            theme = _extract_theme_from_text(current_text, len(chunks) + 1)
            chunk.theme = theme
            chunk.reasoning = f"Chunk automatique de {current_words} mots avec analyse du contenu"
            chunk.nova_chunking = {
                "type_contenu": "discussion",
                "speakers_impliques": list(set([seg["speaker"] for seg in current_chunk_segments])),
                "justification": f"Segmentation automatique basée sur {current_words} mots et analyse du contenu",
                "theme_principal": theme
            }
            chunks.append(chunk)
            current_chunk_segments = []

    if current_chunk_segments:
        current_text = " ".join([s["text"] for s in current_chunk_segments])
        current_words = count_words(current_text)
        chunk = create_chunk_from_segments(current_chunk_segments, len(chunks) + 1)
        theme = _extract_theme_from_text(current_text, len(chunks) + 1)
        chunk.theme = theme
        chunk.reasoning = f"Chunk final de {current_words} mots avec analyse du contenu"
        chunk.nova_chunking = {
            "type_contenu": "discussion",
            "speakers_impliques": list(set([seg["speaker"] for seg in current_chunk_segments])),
            "justification": f"Segmentation finale basée sur {current_words} mots et analyse du contenu",
            "theme_principal": theme
        }
        chunks.append(chunk)

    logger.info(f"✅ {len(chunks)} chunks créés avec métadonnées Nova enrichies")
    return chunks
