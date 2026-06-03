import os
import re
import json
import logging
import signal
import traceback
import torch
from datetime import datetime
from typing import List, Tuple
from botocore.exceptions import ClientError

from config import (S3_BUCKET_NAME, DYNAMODB_TABLE_NAME, SQS_QUEUE_URL,
                    UNIQUE_MEETINGS_TABLE_NAME, ONE_SHOT_MODE, NOVA_ENHANCED_EMBEDDINGS,
                    VISIBILITY_TIMEOUT, sqs_client, dynamodb)
from models import Chunk, CustomJSONEncoder
from aws_utils import safe_s3_download, safe_s3_upload, safe_dynamodb_update, extend_visibility_timeout
from audio import convert_to_wav, transcribe_audio, perform_diarization, merge_transcription_diarization
from embeddings import regenerate_embeddings_after_enrichment, create_faiss_index
from nova import (clean_text_with_nova_step1, extract_global_themes_with_nova_step2,
                  apply_cleaned_text_to_segments, rechunk_thematically_step3)
from chunking import create_semantic_chunks_fallback
from storage import save_results_to_s3

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
_SAFE_S3_KEY_RE = re.compile(r'^[a-zA-Z0-9_./#@:-]{1,512}$')
_ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".mp4", ".mov", ".webm"}


def _validate_message_fields(body: dict) -> tuple:
    """Validate and return (user_id, reunion_id, s3_key) or raise ValueError."""
    for field in ("userId", "reunionId", "s3Audio"):
        if field not in body:
            raise KeyError(field)
        if not isinstance(body[field], str):
            raise ValueError(f"Field '{field}' must be a string")

    user_id = body["userId"]
    reunion_id = body["reunionId"]
    s3_key = body["s3Audio"]

    if not _SAFE_ID_RE.match(user_id):
        raise ValueError("Invalid userId format")
    if not _SAFE_ID_RE.match(reunion_id):
        raise ValueError("Invalid reunionId format")
    if not _SAFE_S3_KEY_RE.match(s3_key):
        raise ValueError("Invalid s3Audio format")

    return user_id, reunion_id, s3_key


running = True


def signal_handler(sig, frame):
    global running
    running = False
    logger.info(f"Shutdown signal {sig} received")


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def process_audio_intelligent(audio_file: str, user_id: str, reunion_id: str) -> Tuple[List[Chunk], str, str]:
    """Full pipeline: Whisper → Nova 3-step → FAISS → S3."""
    try:
        logger.info(f"Processing '{audio_file}' — meeting={reunion_id}, user={user_id}")

        audio_file = convert_to_wav(audio_file)

        transcript_segments = transcribe_audio(audio_file)
        if not transcript_segments:
            raise RuntimeError("Transcription produced no segments")

        raw_text = " ".join(seg["text"] for seg in transcript_segments)
        cleaned_text = clean_text_with_nova_step1(raw_text)
        global_themes_data = extract_global_themes_with_nova_step2(cleaned_text)

        cleaned_segments = apply_cleaned_text_to_segments(transcript_segments, cleaned_text)
        diarization_segments = perform_diarization(audio_file)
        merged_segments = merge_transcription_diarization(cleaned_segments, diarization_segments)

        chunks = rechunk_thematically_step3(merged_segments, cleaned_text, global_themes_data)
        if not chunks:
            logger.warning("Nova rechunking returned no chunks — using fallback")
            chunks = create_semantic_chunks_fallback(merged_segments)

        if NOVA_ENHANCED_EMBEDDINGS:
            chunks = regenerate_embeddings_after_enrichment(chunks)

        faiss_index = create_faiss_index(chunks)
        chunks_key, index_key = save_results_to_s3(chunks, user_id, reunion_id, faiss_index)

        metadata = {
            "workflow_version": "3-step-nova",
            "global_themes": global_themes_data,
            "cleaned_text_length": len(cleaned_text),
            "original_text_length": len(raw_text),
            "processing_stats": {
                "total_chunks": len(chunks),
                "themes_detected": len(global_themes_data.get("themes_principaux", [])),
                "hot_zones": len(global_themes_data.get("zones_chaudes", [])),
                "global_keywords": len(global_themes_data.get("mots_cles_globaux", []))
            }
        }
        metadata_key = f"{user_id}/{reunion_id}/metadata/{reunion_id}_global_analysis.json"
        safe_s3_upload(S3_BUCKET_NAME, metadata_key,
                       json.dumps(metadata, ensure_ascii=False, indent=2, cls=CustomJSONEncoder))

        logger.info(f"Pipeline complete: {len(chunks)} chunks — meeting={reunion_id}")
        return chunks, chunks_key, index_key

    except Exception as e:
        logger.error(f"Pipeline failed for meeting={reunion_id}: {e}")
        traceback.print_exc()
        raise


def process_message(message) -> bool:
    """Download audio from S3, run pipeline, update DynamoDB."""
    start_time = datetime.now()
    receipt_handle = message["ReceiptHandle"]
    local_audio_path = None

    try:
        extend_visibility_timeout(receipt_handle)

        body = json.loads(message["Body"])
        user_id, reunion_id, s3_key = _validate_message_fields(body)

        logger.info(f"Message received (meeting={reunion_id})")

        raw_ext = os.path.splitext(s3_key)[1].lower()
        ext = raw_ext if raw_ext in _ALLOWED_AUDIO_EXTENSIONS else ".audio"
        local_audio_path = f"/tmp/{reunion_id}{ext}"
        os.makedirs("/tmp", exist_ok=True)
        safe_s3_download(S3_BUCKET_NAME, s3_key, local_audio_path)

        extend_visibility_timeout(receipt_handle, VISIBILITY_TIMEOUT)

        chunks, chunks_s3_key, index_s3_key = process_audio_intelligent(
            local_audio_path, user_id, reunion_id
        )

        update_expression = "SET etat = :s, s3Chunks = :c, processedAt = :p"
        expression_values = {
            ":s": "TRAITÉ_INTELLIGENT",
            ":c": chunks_s3_key,
            ":p": datetime.now().isoformat()
        }
        if index_s3_key:
            update_expression += ", s3Index = :f"
            expression_values[":f"] = index_s3_key

        safe_dynamodb_update(
            DYNAMODB_TABLE_NAME,
            {"userId": user_id, "reunionId": reunion_id},
            update_expression,
            expression_values
        )

        try:
            dynamodb.Table(UNIQUE_MEETINGS_TABLE_NAME).update_item(
                Key={"userId": user_id, "reunionId": reunion_id},
                UpdateExpression="SET etat = :s",
                ExpressionAttributeValues={":s": "TRAITÉ_INTELLIGENT"}
            )
        except Exception as e:
            logger.warning(f"UniqueMeetings update failed for {reunion_id}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Done: {len(chunks)} chunks in {elapsed:.1f}s — meeting={reunion_id}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"Malformed SQS message: {e}")
        return True
    except KeyError as e:
        logger.error(f"Missing required field in message: {e}")
        return True
    except ValueError as e:
        logger.error(f"Invalid message field: {e}")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return False
    finally:
        if local_audio_path:
            try:
                if os.path.exists(local_audio_path):
                    os.remove(local_audio_path)
            except Exception as cleanup_err:
                logger.warning(f"Temp file cleanup failed: {cleanup_err}")


def poll_sqs():
    """Poll SQS once (one-shot mode) or continuously (permanent mode)."""
    mode = "one-shot" if ONE_SHOT_MODE else "permanent"
    logger.info(f"Worker starting in {mode} mode")

    try:
        while running:
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,
                AttributeNames=["All"]
            )
            messages = response.get("Messages", [])

            if not messages:
                logger.info("No messages in queue")
                if ONE_SHOT_MODE:
                    return
                continue

            message = messages[0]
            success = process_message(message)

            if success:
                sqs_client.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
                logger.info("Message deleted from SQS")
            else:
                logger.error("Message processing failed — left in queue for retry")

            if ONE_SHOT_MODE:
                return

    except ClientError as e:
        logger.error(f"AWS error: {e.response.get('Error', {}).get('Code')} — {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    logger.info("Worker stopped")
