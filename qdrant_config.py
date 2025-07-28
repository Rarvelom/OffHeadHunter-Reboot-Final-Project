import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_qdrant_client(use_http=False, timeout=30.0):
    """
    Crea y devuelve una instancia del cliente Qdrant
    
    Args:
        use_http: Si es True, usa HTTP en lugar de gRPC (más estable para conexiones lentas)
        timeout: Tiempo máximo de espera para las operaciones (en segundos)
    """
    load_dotenv()
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("No se encontró la URL o la API key de Qdrant en las variables de entorno")
    
    # Usar HTTP o gRPC según se especifique
    prefer_grpc = not use_http
    
    logger = logging.getLogger(__name__)
    logger.info(f"Connecting to Qdrant at {qdrant_url} using {'HTTP' if use_http else 'gRPC'} with timeout {timeout}s")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=prefer_grpc,
        timeout=timeout  # Aumentar timeout para evitar errores de Deadline Exceeded
    )

# Constantes de configuración centralizadas
CV_COLLECTION = "cv_embeddings_BGE2"
JOB_COLLECTION = "job_embeddings_BGE2"
VECTOR_DIMENSION = 1024  # Dimensión para embeddings BAAI/bge-m3

def get_collection_configs():
    """
    Obtiene las configuraciones de las colecciones

    Retorna un diccionario con la siguiente estructura:
    {
        "cv_embeddings_BGE2": {
            "vectors_config": models.VectorParams,
        },
        "job_embeddings_BGE2": {
            "vectors_config": models.VectorParams,
        }
    }
    """
    return {
        CV_COLLECTION: {
            "vectors_config": models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=models.Distance.COSINE
            )
        },
        JOB_COLLECTION: {
            "vectors_config": models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=models.Distance.COSINE
            )
        }
    }

def get_collection_names():
    """
    Retorna los nombres de las colecciones configuradas
    """
    return {
        "cv_collection": CV_COLLECTION,
        "job_collection": JOB_COLLECTION
    }