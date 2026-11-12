#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Environment loading, shared constants and AWS client initialization."""

import boto3
import os
import logging
import torch
import warnings
from botocore.config import Config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Environment variable '{name}' is required and is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


S3_BUCKET_NAME = _require_env("S3_BUCKET_NAME")
DYNAMODB_TABLE_NAME = _require_env("DYNAMODB_TABLE_NAME")
SQS_QUEUE_URL = _require_env("SQS_QUEUE_URL")
UNIQUE_MEETINGS_TABLE_NAME = os.getenv("UNIQUE_MEETINGS_TABLE_NAME", "UniqueMeetings")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
VISIBILITY_TIMEOUT = int(os.getenv("VISIBILITY_TIMEOUT", "3600"))

NOVA_ENHANCED_EMBEDDINGS = _env_flag("NOVA_ENHANCED_EMBEDDINGS", "true")
NOVA_DIRECT_CHUNKING = _env_flag("NOVA_DIRECT_CHUNKING", "false")
ONE_SHOT_MODE = _env_flag("ONE_SHOT_MODE", "true")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "fr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.getLogger(__name__).info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logging.getLogger(__name__).info(f"GPU detected: {torch.cuda.get_device_name(0)}")

MIN_WORDS_PER_CHUNK = 50
MAX_WORDS_PER_CHUNK = 150

NOVA_MODEL_ID = "amazon.nova-lite-v1:0"
TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSIONS = 512

# Values written to the DynamoDB `etat` attribute. Downstream components
# match on these strings, so they are part of the data contract.
STATUS_PROCESSED = "TRAITÉ_INTELLIGENT"
STATUS_FAILED = "FAILED"

session = boto3.Session(region_name=AWS_REGION)
_boto_config = Config(
    retries={'max_attempts': MAX_RETRIES, 'mode': 'standard'},
    connect_timeout=10,
    read_timeout=300,
)

sqs_client = session.client("sqs", config=_boto_config)
s3_client = session.client("s3", config=_boto_config)
dynamodb = session.resource("dynamodb", config=_boto_config)
bedrock_client = session.client('bedrock-runtime', region_name=AWS_REGION, config=_boto_config)
