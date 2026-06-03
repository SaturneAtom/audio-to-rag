import json
import time
import logging
from typing import List, Dict, Any
from config import bedrock_client, MIN_WORDS_PER_CHUNK, MAX_WORDS_PER_CHUNK
from models import Chunk
from chunking import (
    count_words, create_chunk_from_segments, find_chunk_segments,
    divide_chunk_intelligently, divide_chunk_by_words,
    merge_short_chunks_intelligently, create_semantic_chunks_fallback
)

logger = logging.getLogger(__name__)


def is_json_incomplete(text: str) -> bool:
    """Return True if text looks like a truncated JSON response."""
    try:
        cleaned = text.strip()
        for prefix in ("```json", "```"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        json.loads(cleaned.strip())
        return False
    except json.JSONDecodeError as e:
        return "Expecting" in str(e) or "end of JSON input" in str(e)
    except Exception:
        return True


def get_nova_response_with_pagination(prompt: str, max_tokens: int = 8000) -> str:
    """Call Nova and automatically request continuations for truncated JSON responses."""
    response = bedrock_client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1}
    )
    full_response = response["output"]["message"]["content"][0]["text"].strip()

    for iteration in range(5):
        if not is_json_incomplete(full_response):
            break
        logger.debug(f"Response truncated, requesting continuation ({iteration + 1}/5)")
        time.sleep(0.5)
        continuation_prompt = (
            f"Voici le début de ta réponse précédente:\n{full_response}\n\n"
            "Continue exactement où tu t'es arrêté. Donne la SUITE seulement, sans répéter le début.\nSUITE:"
        )
        response = bedrock_client.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": continuation_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1}
        )
        full_response += response["output"]["message"]["content"][0]["text"].strip()
    else:
        logger.warning("Pagination limit reached (5 iterations)")

    return full_response


def _parse_nova_json(response_text: str) -> Any:
    """Strip markdown fences and parse JSON from a Nova response."""
    cleaned = response_text
    for prefix in ("```json", "```"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


def apply_cleaned_text_to_segments(segments: List[Dict], cleaned_text: str) -> List[Dict]:
    """Distribute Nova-cleaned text proportionally across the original segments."""
    if not cleaned_text or not segments:
        return segments

    try:
        original_text = " ".join(seg["text"] for seg in segments)
        if len(cleaned_text) < len(original_text) * 0.5 or len(cleaned_text) > len(original_text) * 2:
            logger.warning("Cleaned text diverges too much from original — keeping original segments")
            return segments

        words_original = original_text.split()
        words_cleaned = cleaned_text.split()
        if not words_cleaned:
            return segments

        word_ratio = len(words_cleaned) / len(words_original)
        cleaned_segments = []
        current_word_index = 0

        for segment in segments:
            count = max(1, int(len(segment["text"].split()) * word_ratio))
            end = min(current_word_index + count, len(words_cleaned))
            cleaned_segment = segment.copy()
            cleaned_segment["text"] = " ".join(words_cleaned[current_word_index:end])
            cleaned_segment["original_text"] = segment["text"]
            cleaned_segments.append(cleaned_segment)
            current_word_index = end

        if current_word_index < len(words_cleaned) and cleaned_segments:
            cleaned_segments[-1]["text"] += " " + " ".join(words_cleaned[current_word_index:])

        return cleaned_segments

    except Exception as e:
        logger.error(f"Failed to apply cleaned text: {e}")
        return segments


def clean_text_with_nova_step1(raw_text: str) -> str:
    """Step 1 — Fix transcription errors, punctuation and filler words via Nova."""
    try:
        prompt = f"""Tu es un expert en nettoyage de transcriptions audio. Nettoie ce texte en corrigeant :
- Les erreurs de transcription
- La ponctuation manquante
- Les répétitions inutiles
- Les hésitations (euh, hmm, etc.)
- Les phrases interrompues

TEXTE À NETTOYER:
{raw_text}

INSTRUCTIONS:
- Garde le sens original intact
- Améliore la lisibilité
- Corrige la ponctuation
- Supprime les éléments parasites
- Retourne SEULEMENT le texte nettoyé, sans commentaires

TEXTE NETTOYÉ:"""

        response = bedrock_client.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 8000, "temperature": 0.1}
        )
        cleaned = response["output"]["message"]["content"][0]["text"].strip()
        logger.info(f"Step 1 done: {len(raw_text)} → {len(cleaned)} chars")
        return cleaned

    except Exception as e:
        logger.error(f"Step 1 failed: {e} — using raw text")
        return raw_text


def extract_global_themes_with_nova_step2(cleaned_text: str) -> Dict[str, Any]:
    """Step 2 — Global macro-analysis: themes, keywords, hot zones, structure."""
    try:
        prompt = f"""Analyse ce texte et crée une carte mentale complète du document.

TEXTE:
{cleaned_text}

Retourne SEULEMENT ce JSON (sans backticks):
{{
"themes_principaux": [
    {{
        "theme": "nom-du-theme",
        "importance": 0.9,
        "mots_cles": ["mot1", "mot2", "mot3"],
        "zone_debut": 0,
        "zone_fin": 500,
        "densite_thematique": "haute"
    }}
],
"mots_cles_globaux": [
    {{
        "mot": "concept-important",
        "frequence": 12,
        "pertinence": 0.85,
        "contextes": ["contexte1", "contexte2"]
    }}
],
"zones_chaudes": [
    {{
        "debut": 100,
        "fin": 300,
        "intensite": 0.8,
        "themes_impliques": ["theme1", "theme2"],
        "raison": "forte-concentration-concepts"
    }}
],
"structure_globale": {{
    "type_conversation": "reunion-travail",
    "progression_narrative": "lineaire",
    "points_cles": ["point1", "point2"],
    "transitions_majeures": [250, 800, 1200]
}}
}}

RÈGLES:
- themes_principaux = maximum 10 thèmes, importance entre 0 et 1
- mots_cles_globaux = maximum 20 mots-clés, avec contextes
- zones_chaudes = zones à forte densité thématique
- positions approximatives en nombre de caractères
- analyse TOUT le texte fourni sans limitation

Si ta réponse est trop longue, je te demanderai la suite."""

        global_analysis = _parse_nova_json(get_nova_response_with_pagination(prompt, max_tokens=10000))

        logger.info(
            f"Step 2 done: {len(global_analysis.get('themes_principaux', []))} themes, "
            f"{len(global_analysis.get('mots_cles_globaux', []))} keywords, "
            f"{len(global_analysis.get('zones_chaudes', []))} hot zones"
        )
        return global_analysis

    except json.JSONDecodeError as e:
        logger.error(f"Step 2 JSON parse error: {e}")
        raise
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        raise


def apply_chunk_analysis(chunk: Chunk, analysis: Dict[str, Any]) -> Chunk:
    """Enrich chunk metadata with Nova analysis without modifying its content."""
    try:
        enriched = Chunk(
            id=chunk.id,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=chunk.text,
            embedding=chunk.embedding,
            speakers_info=chunk.speakers_info.copy(),
            is_topic_transition=chunk.is_topic_transition,
            transition_score=chunk.transition_score,
            cluster_id=chunk.cluster_id,
            theme=analysis.get("theme_principal", chunk.theme),
            reasoning=chunk.reasoning
        )
        enriched.nova_chunking = {
            "theme_principal": analysis.get("theme_principal", ""),
            "themes_secondaires": analysis.get("themes_secondaires", []),
            "pertinence_thematique": analysis.get("pertinence_thematique", 0.5),
            "type_contenu": analysis.get("type_contenu", "discussion"),
            "justification": analysis.get("justification", ""),
            "coherence_score": analysis.get("coherence_score", 0.5),
            "mots_cles_chunk": analysis.get("mots_cles_chunk", []),
            "resume_chunk": analysis.get("resume_chunk", ""),
            "qualite_semantique": analysis.get("qualite_semantique", "moyenne"),
            "suggestions_amelioration": analysis.get("suggestions_amelioration", ""),
            "methode_chunking": "nova-analyse-individuelle",
            "contenu_preserve": True,
            "word_count": count_words(enriched.text)
        }
        return enriched

    except Exception as e:
        logger.error(f"Chunk {chunk.id} enrichment failed: {e}")
        return chunk


def optimize_chunk_with_nova(chunk: Chunk, global_themes: Dict[str, Any], segments: List[Dict]) -> Chunk:
    """Analyse one chunk with Nova to enrich its metadata (content is never modified)."""
    try:
        chunk_segments = find_chunk_segments(chunk, segments)
        if not chunk_segments:
            return chunk

        themes_context = json.dumps([
            {"theme": t["theme"], "mots_cles": t["mots_cles"], "importance": t["importance"]}
            for t in global_themes.get("themes_principaux", [])
        ])
        segments_text = "".join(f"[{i}] {seg['text']}\n" for i, seg in enumerate(chunk_segments))
        current_words = count_words(chunk.text)

        prompt = f"""Analyse ce chunk et identifie sa structure thématique optimale.

CHUNK À ANALYSER ({current_words} mots):
{segments_text}

THÈMES GLOBAUX DISPONIBLES:
{themes_context}

CONTRAINTES:
- NE SUPPRIME AUCUN SEGMENT (analyse seulement)
- Objectif: {MIN_WORDS_PER_CHUNK}-{MAX_WORDS_PER_CHUNK} mots
- Taille actuelle: {current_words} mots

Retourne SEULEMENT ce JSON (sans backticks):
{{
    "theme_principal": "theme-le-plus-pertinent",
    "themes_secondaires": ["theme2", "theme3"],
    "pertinence_thematique": 0.85,
    "type_contenu": "discussion-principale",
    "justification": "pourquoi-ce-theme-est-pertinent",
    "coherence_score": 0.9,
    "mots_cles_chunk": ["mot1", "mot2", "mot3"],
    "resume_chunk": "resume-en-une-phrase",
    "qualite_semantique": "haute",
    "suggestions_amelioration": "comment-ameliorer-sans-supprimer"
}}

RÈGLES:
- GARDE TOUS LES SEGMENTS (aucune suppression)
- Score de pertinence entre 0 et 1"""

        analysis_result = _parse_nova_json(get_nova_response_with_pagination(prompt, max_tokens=5000))
        return apply_chunk_analysis(chunk, analysis_result)

    except json.JSONDecodeError as e:
        logger.error(f"Chunk {chunk.id} JSON parse error: {e}")
        return chunk
    except Exception as e:
        logger.error(f"Chunk {chunk.id} optimisation failed: {e}")
        return chunk


def adjust_chunk_sizes_step3(chunks: List[Chunk], segments: List[Dict]) -> List[Chunk]:
    """Split oversized chunks and flag undersized ones for merging."""
    if not chunks:
        return chunks

    adjusted = []
    for chunk in chunks:
        words = count_words(chunk.text)

        if words > MAX_WORDS_PER_CHUNK:
            sub_chunks = divide_chunk_intelligently(chunk, find_chunk_segments(chunk, segments), MAX_WORDS_PER_CHUNK)
            adjusted.extend(sub_chunks)
            logger.debug(f"Chunk {chunk.id} split: {words}w → {len(sub_chunks)} sub-chunks")
        elif words < MIN_WORDS_PER_CHUNK:
            chunk.nova_chunking["size_adjustment"] = "too_short"
            chunk.nova_chunking["word_count"] = words
            adjusted.append(chunk)
        else:
            chunk.nova_chunking["size_adjustment"] = "optimal"
            chunk.nova_chunking["word_count"] = words
            adjusted.append(chunk)

    final_chunks = merge_short_chunks_intelligently(adjusted)
    for i, chunk in enumerate(final_chunks):
        chunk.id = i + 1

    if final_chunks:
        word_counts = [count_words(c.text) for c in final_chunks]
        logger.info(
            f"Chunks resized: {len(chunks)} → {len(final_chunks)}, "
            f"avg {sum(word_counts) / len(word_counts):.0f}w "
            f"(range {min(word_counts)}–{max(word_counts)})"
        )
    return final_chunks


def create_semantic_chunks_with_nova(segments: List[Dict], global_tags: List[str], global_themes: List[str]) -> List[Chunk]:
    """Direct Nova chunking: Nova groups segments by topic."""
    if not segments:
        return []

    segments_text = "".join(f"[{i}] {seg['text']}\n" for i, seg in enumerate(segments))

    prompt = f"""Analyse ces segments et groupe-les par sujets cohérents:

{segments_text}

Retourne SEULEMENT ce JSON (sans backticks):
{{
"chunks": [
{{
"segment_indices": [0, 1, 2],
"theme": "premier-sujet-traite"
}},
{{
"segment_indices": [3, 4, 5],
"theme": "deuxieme-sujet-traite"
}}
]
}}

RÈGLES:
- Groupe les segments qui parlent du même sujet
- Chaque chunk doit avoir au moins 2 segments
- Traite TOUS les segments fournis
- theme = le sujet spécifique de ce groupe (pas "discussion")
- Utilise des tirets au lieu d'espaces

Si ta réponse est trop longue, je te demanderai la suite."""

    result = _parse_nova_json(get_nova_response_with_pagination(prompt, max_tokens=10000))

    nova_chunks = []
    for i, chunk_info in enumerate(result.get("chunks", [])):
        valid_indices = [idx for idx in chunk_info.get("segment_indices", []) if 0 <= idx < len(segments)]
        if not valid_indices:
            continue
        chunk_segments = [segments[idx] for idx in valid_indices]
        chunk = create_chunk_from_segments(chunk_segments, i + 1)
        chunk.theme = chunk_info.get("theme", f"chunk-{i + 1}")
        chunk.nova_chunking = {
            "theme_principal": chunk.theme,
            "type_contenu": "discussion",
            "justification": "Nova semantic grouping",
            "speakers_impliques": list(set(seg["speaker"] for seg in chunk_segments))
        }
        nova_chunks.append(chunk)

    if not nova_chunks:
        raise RuntimeError("Nova returned no valid chunks")

    logger.info(f"Nova chunking: {len(nova_chunks)} chunks from {len(segments)} segments")
    return nova_chunks


def rechunk_thematically_step3(segments: List[Dict], cleaned_text: str, global_themes: Dict[str, Any]) -> List[Chunk]:
    """Step 3 — Per-chunk Nova analysis followed by size adjustment."""
    try:
        themes = [t["theme"] for t in global_themes.get("themes_principaux", [])]
        logger.info(f"Step 3 start: {len(segments)} segments, {len(themes)} global themes")

        base_chunks = create_semantic_chunks_fallback(segments)
        optimized_chunks = [
            optimize_chunk_with_nova(chunk, global_themes, segments)
            for chunk in base_chunks
        ]
        final_chunks = adjust_chunk_sizes_step3(optimized_chunks, segments)

        logger.info(f"Step 3 done: {len(base_chunks)} base → {len(final_chunks)} final chunks")
        return final_chunks

    except Exception as e:
        logger.error(f"Step 3 failed: {e} — using fallback chunking")
        return create_semantic_chunks_fallback(segments)
