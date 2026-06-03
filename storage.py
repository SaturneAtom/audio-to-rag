import json
import logging
import tempfile
import faiss
from datetime import datetime
from typing import List, Optional
from config import S3_BUCKET_NAME
from models import Chunk, CustomJSONEncoder
from aws_utils import safe_s3_upload

logger = logging.getLogger(__name__)


def save_results_to_s3(chunks: List[Chunk], user_id: str, reunion_id: str,
                       index: Optional[faiss.IndexFlatIP] = None):
    """Sauvegarde les chunks et l'index FAISS dans S3"""
    try:
        chunks_data = {
            "metadata": {
                "user_id": user_id,
                "reunion_id": reunion_id,
                "total_chunks": len(chunks),
                "creation_date": datetime.now().isoformat(),
                "chunking_strategy": "semantic_intelligent"
            },
            "chunks": [chunk.to_dict() for chunk in chunks]
        }

        chunks_json = json.dumps(chunks_data, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        chunks_key = f"{user_id}/{reunion_id}/chunks/{reunion_id}.json"
        safe_s3_upload(S3_BUCKET_NAME, chunks_key, chunks_json)
        logger.info(f"✅ Chunks sauvegardés dans S3: {chunks_key}")

        index_key = None
        if index is not None:
            with tempfile.NamedTemporaryFile() as tmp_file:
                faiss.write_index(index, tmp_file.name)
                tmp_file.seek(0)
                index_key = f"{user_id}/{reunion_id}/index/{reunion_id}.faiss"
                safe_s3_upload(S3_BUCKET_NAME, index_key, tmp_file.read())
                logger.info(f"✅ Index FAISS sauvegardé dans S3: {index_key}")

        return chunks_key, index_key

    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde S3: {e}")
        raise
