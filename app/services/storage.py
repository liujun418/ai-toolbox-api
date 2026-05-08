"""S3/R2 storage service for file uploads."""

import uuid
from datetime import datetime, timedelta

import boto3
from botocore.config import Config

from app.config import settings


def get_s3_client():
    """Get S3/R2 client."""
    kwargs = {"config": Config(signature_version="s3v4")}
    if settings.AWS_S3_ENDPOINT:
        kwargs["endpoint_url"] = settings.AWS_S3_ENDPOINT
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        **kwargs,
    )


def generate_upload_key(filename: str, user_id: str) -> str:
    """Generate unique storage key for uploaded file."""
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "png"
    return f"uploads/{user_id}/{uuid.uuid4().hex}.{ext}"


def generate_download_key(user_id: str, task_id: str, ext: str = "png") -> str:
    """Generate unique storage key for output file."""
    return f"outputs/{user_id}/{task_id}.{ext}"


async def upload_file(file_bytes: bytes, key: str, content_type: str = "image/png") -> str:
    """Upload file bytes to S3/R2. Returns the file key."""
    client = get_s3_client()
    client.put_object(
        Bucket=settings.AWS_S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return key


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate presigned download URL (valid for 1 hour by default)."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.AWS_S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )


def generate_presigned_upload_url(key: str, content_type: str, expires_in: int = 3600) -> str:
    """Generate presigned upload URL for direct browser-to-S3 upload."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.AWS_S3_BUCKET,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )


def delete_file(key: str) -> None:
    """Delete file from storage."""
    client = get_s3_client()
    client.delete_object(Bucket=settings.AWS_S3_BUCKET, Key=key)
