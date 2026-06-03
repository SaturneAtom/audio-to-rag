import json
import logging
import tempfile
import numpy as np
import faiss
from typing import List, Optional
from config import bedrock_client
from models import Chunk

logger = logging.getLogger(__name__)


def generate_titan_embedding(text: str) -> Optional[np.ndarray]:
    """Génère un embedding vectoriel avec Amazon Titan Embed v2"""
    try:
        body = json.dumps({
            "inputText": text,
            "dimensions": 512,
            "normalize": True
        })
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=body,
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read())
        return np.array(response_body['embedding'], dtype=np.float32)

    except Exception as e:
        logger.error(f"Erreur embedding Titan: {e}")
        return None


def regenerate_embeddings_after_enrichment(chunks: List[Chunk]) -> List[Chunk]:
    """Régénère les embeddings en enrichissant le texte avec les métadonnées Nova"""
    logger.info("🔄 Régénération des embeddings avec métadonnées Nova...")

    if not chunks:
        logger.warning("⚠️ Aucun chunk à enrichir")
        return chunks

    updated_chunks = []
    enriched_count = 0

    for i, chunk in enumerate(chunks):
        enriched_text = chunk.text
        enrichment_tags = []

        if chunk.nova_chunking:
            theme = chunk.nova_chunking.get("theme_principal", "")
            content_type = chunk.nova_chunking.get("type_contenu", "")
            speakers = chunk.nova_chunking.get("speakers_impliques", [])

            if theme:
                enriched_text += f" [THEME: {theme}]"
                enrichment_tags.append(f"THEME:{theme}")
            if content_type:
                enriched_text += f" [TYPE: {content_type}]"
                enrichment_tags.append(f"TYPE:{content_type}")
            if speakers:
                enriched_text += f" [SPEAKERS: {', '.join(speakers)}]"
                enrichment_tags.append(f"SPEAKERS:{len(speakers)}")

        if chunk.theme:
            enriched_text += f" [SUBJECT: {chunk.theme}]"
            enrichment_tags.append(f"SUBJECT:{chunk.theme}")

        if enrichment_tags:
            enriched_count += 1

        if i == 0:
            logger.debug(f"Enrichissement chunk 1: tags={enrichment_tags}")

        new_embedding = generate_titan_embedding(enriched_text)

        updated_chunks.append(Chunk(
            id=chunk.id,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=chunk.text,
            embedding=new_embedding,
            speakers_info=chunk.speakers_info.copy() if chunk.speakers_info else [],
            is_topic_transition=chunk.is_topic_transition,
            transition_score=chunk.transition_score,
            cluster_id=chunk.cluster_id,
            theme=chunk.theme,
            reasoning=chunk.reasoning,
            nova_chunking=chunk.nova_chunking.copy() if chunk.nova_chunking else {}
        ))

    logger.info(f"✅ Embeddings régénérés: {enriched_count}/{len(chunks)} chunks enrichis avec métadonnées Nova")
    return updated_chunks


def create_faiss_index(chunks: List[Chunk]) -> Optional[faiss.IndexFlatIP]:
    """Crée un index FAISS pour la recherche vectorielle"""
    try:
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]

        if not embeddings:
            logger.warning("⚠️ Aucun embedding disponible")
            return None

        embeddings_matrix = np.vstack(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_matrix)

        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)

        logger.info(f"✅ Index FAISS créé: {index.ntotal} vecteurs")
        return index

    except Exception as e:
        logger.error(f"❌ Erreur création index FAISS: {e}")
        return None
