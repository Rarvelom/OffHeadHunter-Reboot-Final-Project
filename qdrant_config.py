import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

def get_qdrant_client():
    """
    Crea y devuelve una instancia del cliente Qdrant
    """
    load_dotenv()
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("No se encontró la URL o la API key de Qdrant en las variables de entorno")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )

def get_collection_configs():
    """
    Obtiene las configuraciones de las colecciones

    Retorna un diccionario con la siguiente estructura:
    {
        "cv_embeddings": {
            "vectors_config": models.VectorParams,
        },
        "job_embeddings": {
            "vectors_config": models.VectorParams,
        }
    }
    """
    return {
        "cv_embeddings": {
            "vectors_config": models.VectorParams(
                size=768,  # Dimensión para los embeddings (all-mpnet-base-v2 output)
                distance=models.Distance.COSINE
            )
        },
        "job_embeddings": {
            "vectors_config": models.VectorParams(
                size=768,  # Dimensión para los embeddings (all-mpnet-base-v2 output)
                distance=models.Distance.COSINE
            )
        }
    }