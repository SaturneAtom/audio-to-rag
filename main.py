#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AudioPipeline — Entry point
"""

import sys
import logging
from config import (AWS_REGION, S3_BUCKET_NAME, DYNAMODB_TABLE_NAME, SQS_QUEUE_URL,
                    DEVICE, MAX_RETRIES, ONE_SHOT_MODE, NOVA_ENHANCED_EMBEDDINGS)
from worker import poll_sqs

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("🔧 Configuration:")
        logger.debug(f"  - Région AWS: {AWS_REGION}")
        logger.debug(f"  - Bucket S3: {S3_BUCKET_NAME}")
        logger.debug(f"  - Table DynamoDB: {DYNAMODB_TABLE_NAME}")
        logger.debug(f"  - URL SQS: {SQS_QUEUE_URL}")
        logger.info(f"  - Device: {DEVICE}")
        logger.info(f"  - Max retries: {MAX_RETRIES}")
        logger.info(f"  - Mode: {'Éphémère (Simple One-Shot)' if ONE_SHOT_MODE else 'Permanent'}")
        logger.info(f"  - Nova Enhanced Embeddings: {'Activé' if NOVA_ENHANCED_EMBEDDINGS else 'Désactivé'}")

        poll_sqs()

        if ONE_SHOT_MODE:
            logger.info("🎯 Container One-Shot terminé")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("👋 Arrêt par l'utilisateur")
    except Exception as e:
        logger.critical(f"❌ Erreur fatale: {str(e)}", exc_info=True)
        sys.exit(1)

    sys.exit(0)
