import time
import logging
from botocore.exceptions import ClientError
from config import s3_client, dynamodb, sqs_client, SQS_QUEUE_URL, MAX_RETRIES, VISIBILITY_TIMEOUT

logger = logging.getLogger(__name__)


def with_retries(operation, description, retries=MAX_RETRIES):
    """Run `operation()` and retry on ClientError with exponential backoff (1s, 2s, 4s...).

    Returns the operation result, or re-raises the last ClientError once
    `retries` attempts have been made.
    """
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except ClientError as e:
            if attempt == attempts:
                logger.error(f"{description} failed after {attempts} attempts: {e}")
                raise
            logger.warning(f"{description} failed, attempt {attempt}/{attempts}: {e}")
            time.sleep(2 ** (attempt - 1))


def safe_s3_download(bucket, key, local_path, retries=MAX_RETRIES):
    """Download an S3 object to a local file, with retries."""
    with_retries(lambda: s3_client.download_file(bucket, key, local_path), "S3 download", retries)
    return True


def safe_s3_upload(bucket, key, content, retries=MAX_RETRIES):
    """Upload str or bytes content to S3, with retries."""
    body = content.encode('utf-8') if isinstance(content, str) else content
    with_retries(lambda: s3_client.put_object(Bucket=bucket, Key=key, Body=body), "S3 upload", retries)
    return True


def safe_dynamodb_update(table_name, key, update_expression, expression_values, retries=MAX_RETRIES):
    """Update a DynamoDB item, with retries."""
    table = dynamodb.Table(table_name)
    with_retries(
        lambda: table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values
        ),
        "DynamoDB update",
        retries,
    )
    return True


def extend_visibility_timeout(receipt_handle, timeout=VISIBILITY_TIMEOUT):
    """Extend the visibility timeout of an SQS message (single attempt, best effort)."""
    try:
        sqs_client.change_message_visibility(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=timeout
        )
        logger.debug(f"Visibility timeout extended by {timeout}s")
        return True
    except ClientError as e:
        logger.warning(f"Could not extend visibility timeout: {e}")
        return False
