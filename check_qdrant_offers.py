#!/usr/bin/env python3
"""
Script para verificar las ofertas en Qdrant y sus timestamps.
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from datetime import datetime, timezone, timedelta
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
        return None
    
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30.0
        )
        return client
    except Exception as e:
        logger.error(f"Error al conectar con Qdrant: {str(e)}")
        return None

def check_offers():
    """Verificar las ofertas en Qdrant y sus timestamps."""
    client = get_qdrant_client()
    if not client:
        return
    
    # Obtener la colección de ofertas
    collection_name = "job_embeddings_BGE"
    
    try:
        # Obtener información de la colección
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            logger.error(f"La colección {collection_name} no existe en Qdrant")
            return
        
        # Obtener todas las ofertas
        all_offers, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        if not all_offers:
            logger.info("No hay ofertas en la colección")
            return
        
        logger.info(f"Total de ofertas en la colección: {len(all_offers)}")
        
        # Obtener el CV más reciente
        cv_collection = "cv_embeddings_BGE"
        if cv_collection in collection_names:
            try:
                # Primero intentamos obtener todos los CVs sin filtrar por uploaded_at
                cvs, _ = client.scroll(
                    collection_name=cv_collection,
                    limit=10,  # Tomamos varios por si hay alguno sin el campo
                    with_payload=["uploaded_at"],
                    with_vectors=False
                )
                
                # Filtramos los que tienen el campo uploaded_at
                cvs_with_timestamp = [cv for cv in cvs if cv.payload and 'uploaded_at' in cv.payload]
                
                if not cvs_with_timestamp:
                    logger.warning("No se encontraron CVs con el campo 'uploaded_at'")
                    cvs = []
                else:
                    # Ordenamos por uploaded_at manualmente
                    cvs = sorted(cvs_with_timestamp, 
                               key=lambda x: x.payload['uploaded_at'], 
                               reverse=True)
                    cvs = [cvs[0]]  # Tomamos solo el más reciente
                    
            except Exception as e:
                logger.error(f"Error al obtener CVs: {str(e)}")
                cvs = []
            
            if cvs:
                cv = cvs[0]
                cv_time = datetime.fromtimestamp(cv.payload['uploaded_at'], tz=timezone.utc)
                logger.info(f"\nCV más reciente:")
                logger.info(f"- ID: {cv.id}")
                logger.info(f"- Subido el: {cv_time} (UTC)")
                
                # Contar ofertas posteriores al CV
                recent_offers = [
                    o for o in all_offers 
                    if o.payload.get('scraped_at', 0) > cv.payload['uploaded_at']
                ]
                
                logger.info(f"\nOfertas posteriores al CV ({len(recent_offers)}):")
                for i, offer in enumerate(recent_offers[:5], 1):  # Mostrar solo las primeras 5
                    offer_time = datetime.fromtimestamp(offer.payload.get('scraped_at', 0), tz=timezone.utc)
                    logger.info(f"{i}. {offer.payload.get('title', 'Sin título')}")
                    logger.info(f"   - ID: {offer.id}")
                    logger.info(f"   - Empresa: {offer.payload.get('company', 'Desconocida')}")
                    logger.info(f"   - Scraped at: {offer_time} (UTC)")
                
                if len(recent_offers) > 5:
                    logger.info(f"   ... y {len(recent_offers) - 5} ofertas más")
                
                # Mostrar el rango de fechas de las ofertas
                if all_offers:
                    min_time = min(o.payload.get('scraped_at', float('inf')) for o in all_offers)
                    max_time = max(o.payload.get('scraped_at', 0) for o in all_offers)
                    
                    logger.info("\nRango de fechas de las ofertas:")
                    logger.info(f"- Más antigua: {datetime.fromtimestamp(min_time, tz=timezone.utc)} (UTC)")
                    logger.info(f"- Más reciente: {datetime.fromtimestamp(max_time, tz=timezone.utc)} (UTC)")
                    
                    if max_time < cv.payload['uploaded_at']:
                        logger.warning("\n¡ADVERTENCIA: No hay ofertas más recientes que el CV!")
                        logger.warning(f"La oferta más reciente es de {datetime.fromtimestamp(max_time, tz=timezone.utc)} (UTC)")
                        logger.warning(f"El CV se subió el {cv_time} (UTC)")
                    
        # Mostrar información de las ofertas
        logger.info("\nMuestra de ofertas (máximo 5):")
        for i, offer in enumerate(all_offers[:5], 1):
            offer_time = datetime.fromtimestamp(offer.payload.get('scraped_at', 0), tz=timezone.utc)
            logger.info(f"{i}. {offer.payload.get('title', 'Sin título')}")
            logger.info(f"   - ID: {offer.id}")
            logger.info(f"   - Empresa: {offer.payload.get('company', 'Desconocida')}")
            logger.info(f"   - Scraped at: {offer_time} (UTC)")
        
        if len(all_offers) > 5:
            logger.info(f"   ... y {len(all_offers) - 5} ofertas más")
        
    except Exception as e:
        logger.error(f"Error al verificar ofertas: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    check_offers()
