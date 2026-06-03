#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
            f"La variable d'environnement '{name}' est obligatoire et n'est pas définie. "
            f"Copiez .env.example vers .env et renseignez les valeurs."
        )
    return value


S3_BUCKET_NAME = _require_env("S3_BUCKET_NAME")
DYNAMODB_TABLE_NAME = _require_env("DYNAMODB_TABLE_NAME")
SQS_QUEUE_URL = _require_env("SQS_QUEUE_URL")
UNIQUE_MEETINGS_TABLE_NAME = os.getenv("UNIQUE_MEETINGS_TABLE_NAME", "UniqueMeetings")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
VISIBILITY_TIMEOUT = int(os.getenv("VISIBILITY_TIMEOUT", "3600"))

NOVA_ENHANCED_EMBEDDINGS = os.getenv("NOVA_ENHANCED_EMBEDDINGS", "true").lower() == "true"
ONE_SHOT_MODE = os.getenv("ONE_SHOT_MODE", "true").lower() == "true"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "fr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.getLogger(__name__).info(f"🖥️ Device utilisé: {DEVICE}")
if torch.cuda.is_available():
    logging.getLogger(__name__).info(f"🎮 GPU détecté: {torch.cuda.get_device_name(0)}")

MIN_WORDS_PER_CHUNK = 50
MAX_WORDS_PER_CHUNK = 150

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
