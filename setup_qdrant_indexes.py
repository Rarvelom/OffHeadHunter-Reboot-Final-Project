#!/usr/bin/env python3
"""
Script para configurar índices necesarios en Qdrant para el funcionamiento de OffHeadHunter.
"""
import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_qdrant_client():
    """Obtener un cliente de Qdrant configurado."""
    load_dotenv()
    
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url:
        logger.error("QDRANT_URL no está configurado en las variables de entorno")
        sys.exit(1)
    
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30.0
        )
        return client
    except Exception as e:
        logger.error(f"Error al conectar con Qdrant: {str(e)}")
        sys.exit(1)

def create_scraped_at_index(client, collection_name="job_embeddings_BGE"):
    """Crear un índice para el campo scraped_at en la colección especificada."""
    try:
        # Verificar si la colección existe
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            logger.error(f"La colección {collection_name} no existe en Qdrant")
            return False
        
        # Crear el índice para el campo scraped_at
        logger.info(f"Creando índice para el campo 'scraped_at' en la colección '{collection_name}'...")
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="scraped_at",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True
        )
        
        logger.info("Índice creado exitosamente")
        return True
        
    except Exception as e:
        if "already exists" in str(e):
            logger.info("El índice ya existe, no es necesario crearlo")
            return True
        logger.error(f"Error al crear el índice: {str(e)}")
        return False

def main():
    """Función principal."""
    logger.info("Iniciando configuración de índices de Qdrant")
    
    # Obtener cliente de Qdrant
    client = get_qdrant_client()
    
    # Crear índice para el campo scraped_at
    success = create_scraped_at_index(client)
    
    if success:
        logger.info("Configuración de índices completada exitosamente")
        return 0
    else:
        logger.error("Error al configurar los índices")
        return 1

if __name__ == "__main__":
    sys.exit(main())
