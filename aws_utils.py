import time
import logging
from botocore.exceptions import ClientError
from config import s3_client, dynamodb, sqs_client, SQS_QUEUE_URL, MAX_RETRIES, VISIBILITY_TIMEOUT

logger = logging.getLogger(__name__)


def safe_s3_download(bucket, key, local_path, retries=MAX_RETRIES):
    """Télécharge un fichier S3 avec mécanisme de retry"""
    for attempt in range(retries):
        try:
            s3_client.download_file(bucket, key, local_path)
            return True
        except ClientError as e:
            if attempt < retries - 1:
                logger.warning(f"⚠️ Échec téléchargement S3, tentative {attempt+1}/{retries}: {str(e)}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"❌ Échec téléchargement S3 après {retries} tentatives: {str(e)}")
                raise
    return False


def safe_s3_upload(bucket, key, content, retries=MAX_RETRIES):
    """Téléverse un contenu dans S3 avec mécanisme de retry"""
    for attempt in range(retries):
        try:
            if isinstance(content, str):
                s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'))
            else:
                s3_client.put_object(Bucket=bucket, Key=key, Body=content)
            return True
        except ClientError as e:
            if attempt < retries - 1:
                logger.warning(f"⚠️ Échec upload S3, tentative {attempt+1}/{retries}: {str(e)}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"❌ Échec upload S3 après {retries} tentatives: {str(e)}")
                raise
    return False


def safe_dynamodb_update(table_name, key, update_expression, expression_values, retries=MAX_RETRIES):
    """Met à jour DynamoDB avec mécanisme de retry"""
    table = dynamodb.Table(table_name)
    for attempt in range(retries):
        try:
            table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            return True
        except ClientError as e:
            if attempt < retries - 1:
                logger.warning(f"⚠️ Échec MAJ DynamoDB, tentative {attempt+1}/{retries}: {str(e)}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"❌ Échec MAJ DynamoDB après {retries} tentatives: {str(e)}")
                raise
    return False


def extend_visibility_timeout(receipt_handle, timeout=VISIBILITY_TIMEOUT):
    """Prolonge la visibilité d'un message SQS pendant son traitement"""
    try:
        sqs_client.change_message_visibility(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=timeout
        )
        logger.debug(f"✓ Visibilité prolongée de {timeout}s")
        return True
    except ClientError as e:
        logger.warning(f"⚠️ Impossible de prolonger visibilité: {str(e)}")
        return False
