import json
import logging
import numpy as np
import faiss
from typing import List, Optional
from config import bedrock_client, TITAN_EMBED_MODEL_ID, EMBEDDING_DIMENSIONS
from models import Chunk
from aws_utils import with_retries

logger = logging.getLogger(__name__)


def generate_titan_embedding(text: str) -> Optional[np.ndarray]:
    """Embed text with Amazon Titan Embed v2. Returns None if the call fails.

    A missing embedding is tolerated while chunks are being built, but
    `create_faiss_index` refuses to index a chunk without one.
    """
    try:
        body = json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True
        })
        response = with_retries(
            lambda: bedrock_client.invoke_model(
                modelId=TITAN_EMBED_MODEL_ID,
                body=body,
                contentType="application/json"
            ),
            "Titan embedding",
        )
        response_body = json.loads(response['body'].read())
        return np.array(response_body['embedding'], dtype=np.float32)

    except Exception as e:
        logger.error(f"Titan embedding failed: {e}")
        return None


def regenerate_embeddings_after_enrichment(chunks: List[Chunk]) -> List[Chunk]:
    """Recompute embeddings on the chunk text augmented with its Nova metadata tags."""
    logger.info("Regenerating embeddings with Nova metadata...")

    if not chunks:
        logger.warning("No chunks to enrich")
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
            logger.debug(f"First chunk enrichment tags: {enrichment_tags}")

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

    logger.info(f"Embeddings regenerated: {enriched_count}/{len(chunks)} chunks enriched with Nova metadata")
    return updated_chunks


def create_faiss_index(chunks: List[Chunk]) -> Optional[faiss.IndexFlatIP]:
    """Build a FAISS inner-product index with one row per chunk, in chunk order.

    Row i of the index always corresponds to chunks[i]. If any chunk has no
    embedding the job fails rather than producing an index whose rows no
    longer line up with the chunks JSON.
    """
    if not chunks:
        logger.warning("No chunks to index")
        return None

    missing = [chunk.id for chunk in chunks if chunk.embedding is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(chunks)} chunks have no embedding (chunk ids: {missing}); "
            "refusing to build a misaligned FAISS index"
        )

    embeddings_matrix = np.vstack([chunk.embedding for chunk in chunks]).astype('float32')
    faiss.normalize_L2(embeddings_matrix)

    index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    logger.info(f"FAISS index created: {index.ntotal} vectors")
    return index
