#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio-to-rag — Entry point
"""

import sys
import logging
from config import (AWS_REGION, S3_BUCKET_NAME, DYNAMODB_TABLE_NAME, SQS_QUEUE_URL,
                    DEVICE, MAX_RETRIES, ONE_SHOT_MODE, NOVA_ENHANCED_EMBEDDINGS,
                    NOVA_DIRECT_CHUNKING)
from worker import poll_sqs

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the worker and return the process exit code."""
    try:
        logger.info("Configuration:")
        logger.debug(f"  - AWS region: {AWS_REGION}")
        logger.debug(f"  - S3 bucket: {S3_BUCKET_NAME}")
        logger.debug(f"  - DynamoDB table: {DYNAMODB_TABLE_NAME}")
        logger.debug(f"  - SQS URL: {SQS_QUEUE_URL}")
        logger.info(f"  - Device: {DEVICE}")
        logger.info(f"  - Max retries: {MAX_RETRIES}")
        logger.info(f"  - Mode: {'one-shot' if ONE_SHOT_MODE else 'permanent'}")
        logger.info(f"  - Nova enhanced embeddings: {'enabled' if NOVA_ENHANCED_EMBEDDINGS else 'disabled'}")
        logger.info(f"  - Nova direct chunking: {'enabled' if NOVA_DIRECT_CHUNKING else 'disabled'}")

        ok = poll_sqs()

        if ONE_SHOT_MODE:
            if not ok:
                logger.error("One-shot container finished with a failure")
                return 1
            logger.info("One-shot container finished")

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
