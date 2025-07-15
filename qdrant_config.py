from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_qdrant_client():
    """Get Qdrant client instance with proper authentication"""
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Qdrant URL or API key not found in environment variables")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )

def get_collection_configs():
    """
    Get collection configurations

    Returns a dictionary with the following structure:

    {
        "cv_embeddings": {
            "vectors_config": models.VectorParams,
            "payload_schema": {
                "user_id": models.PayloadSchemaType.KEYWORD,
                "cv_id": models.PayloadSchemaType.KEYWORD,
                "version": models.PayloadSchemaType.INTEGER,
                "created_at": models.PayloadSchemaType.DATETIME
            }
        },
        "job_embeddings": {
            "vectors_config": models.VectorParams,
            "payload_schema": {
                "job_id": models.PayloadSchemaType.KEYWORD,
                "source_id": models.PayloadSchemaType.KEYWORD,
                "created_at": models.PayloadSchemaType.DATETIME,
                "is_active": models.PayloadSchemaType.INTEGER
            }
        }
    }
    """
    return {
        "cv_embeddings": {
            "vectors_config": models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            ),
            "payload_schema": {
                "user_id": models.PayloadSchemaType.KEYWORD,
                "cv_id": models.PayloadSchemaType.KEYWORD,
                "version": models.PayloadSchemaType.INTEGER,
                "created_at": models.PayloadSchemaType.DATETIME
            }
        },
        "job_embeddings": {
            "vectors_config": models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            ),
            "payload_schema": {
                "job_id": models.PayloadSchemaType.KEYWORD,
                "source_id": models.PayloadSchemaType.KEYWORD,
                "created_at": models.PayloadSchemaType.DATETIME,
                "is_active": models.PayloadSchemaType.INTEGER
            }
        }
    }