#!/usr/bin/env python3
"""
Script para verificar los CVs en MongoDB y sus timestamps.
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_mongodb_client():
    """Obtener un cliente de MongoDB configurado."""
    load_dotenv()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        logger.error("MONGODB_URI no está configurado en las variables de entorno")
        return None
    
    try:
        client = MongoClient(mongodb_uri)
        # Verificar la conexión
        client.admin.command('ping')
        logger.info("Conexión exitosa a MongoDB")
        return client
    except Exception as e:
        logger.error(f"Error al conectar con MongoDB: {str(e)}")
        return None

def check_cvs():
    """Verificar los CVs en MongoDB y sus timestamps."""
    client = get_mongodb_client()
    if not client:
        return
    
    db_name = os.getenv('MONGODB_DB', 'offheadhunter_db')
    cv_collection_name = 'cv_uploads'
    
    try:
        db = client[db_name]
        
        # Verificar si la colección existe
        if cv_collection_name not in db.list_collection_names():
            logger.error(f"La colección {cv_collection_name} no existe en la base de datos")
            return
        
        cv_collection = db[cv_collection_name]
        
        # Obtener el CV más reciente
        latest_cv = cv_collection.find_one(
            {"uploaded_at": {"$exists": True}},
            sort=[("uploaded_at", -1)]
        )
        
        if not latest_cv:
            logger.warning("No se encontraron CVs con el campo 'uploaded_at' en MongoDB")
            # Mostrar información sobre los CVs disponibles
            all_cvs = list(cv_collection.find().limit(5))
            logger.info(f"\nMuestra de CVs (máximo 5):")
            for i, cv in enumerate(all_cvs, 1):
                logger.info(f"{i}. ID: {cv.get('_id')}")
                logger.info(f"   - Filename: {cv.get('filename')}")
                logger.info(f"   - Campos disponibles: {', '.join(cv.keys())}")
                if 'metadata' in cv and isinstance(cv['metadata'], dict):
                    logger.info(f"   - Metadatos: {', '.join(cv['metadata'].keys())}")
            return
        
        # Mostrar información del CV más reciente
        logger.info("\nCV más reciente en MongoDB:")
        logger.info(f"- ID: {latest_cv['_id']}")
        logger.info(f"- Filename: {latest_cv.get('filename')}")
        logger.info(f"- Subido el: {latest_cv['uploaded_at']} (UTC)")
        
        # Verificar si el CV está en Qdrant
        qdrant_id = latest_cv.get('qdrant_id')
        if not qdrant_id:
            logger.warning("Este CV no tiene un ID de Qdrant asociado")
            return
        
        logger.info(f"- ID de Qdrant: {qdrant_id}")
        
        # Verificar ofertas posteriores a este CV
        check_offers_after_cv(latest_cv['uploaded_at'])
        
    except Exception as e:
        logger.error(f"Error al verificar CVs: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        client.close()

def check_offers_after_cv(cv_upload_time):
    """Verificar ofertas posteriores a la carga de un CV."""
    from qdrant_client import QdrantClient
    from datetime import datetime, timezone
    
    logger.info("\nVerificando ofertas posteriores al CV...")
    
    # Configurar cliente Qdrant
    load_dotenv()
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url:
        logger.error("QDRANT_URL no está configurado en las variables de entorno")
        return
    
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30.0
        )
        
        # Convertir cv_upload_time a timestamp Unix si es un objeto datetime
        if isinstance(cv_upload_time, datetime):
            if cv_upload_time.tzinfo is None:
                cv_upload_time = cv_upload_time.replace(tzinfo=timezone.utc)
            timestamp_threshold = int(cv_upload_time.timestamp())
        else:
            # Asumir que ya es un timestamp
            timestamp_threshold = int(cv_upload_time)
        
        logger.info(f"Buscando ofertas con scraped_at > {cv_upload_time} (timestamp: {timestamp_threshold})")
        
        # Buscar ofertas posteriores al CV
        recent_offers = client.scroll(
            collection_name="job_embeddings_BGE",
            scroll_filter={
                "must": [
                    {
                        "key": "scraped_at",
                        "range": {
                            "gt": timestamp_threshold
                        }
                    }
                ]
            },
            limit=10,  # Limitar a 10 ofertas para la muestra
            with_payload=["title", "company", "scraped_at"],
            with_vectors=False
        )
        
        offers = recent_offers[0]
        logger.info(f"\nSe encontraron {len(offers)} ofertas posteriores al CV")
        
        if offers:
            logger.info("\nMuestra de ofertas posteriores al CV:")
            for i, offer in enumerate(offers, 1):
                offer_time = datetime.fromtimestamp(offer.payload.get('scraped_at', 0), tz=timezone.utc)
                logger.info(f"{i}. {offer.payload.get('title', 'Sin título')}")
                logger.info(f"   - ID: {offer.id}")
                logger.info(f"   - Empresa: {offer.payload.get('company', 'Desconocida')}")
                logger.info(f"   - Scraped at: {offer_time} (UTC)")
        
        # Verificar también el rango de fechas de todas las ofertas
        all_offers = client.scroll(
            collection_name="job_embeddings_BGE",
            limit=1,
            with_payload=["scraped_at"],
            with_vectors=False,
            scroll_filter={
                "must": [
                    {"key": "scraped_at", "range": {"gt": 0}}
                ]
            },
            order_by={"key": "scraped_at", "direction": "asc"}
        )
        
        if all_offers[0]:
            min_time = all_offers[0][0].payload.get('scraped_at', 0)
            max_time = client.scroll(
                collection_name="job_embeddings_BGE",
                limit=1,
                with_payload=["scraped_at"],
                with_vectors=False,
                order_by={"key": "scraped_at", "direction": "desc"}
            )[0][0].payload.get('scraped_at', 0)
            
            logger.info("\nRango de fechas de las ofertas en Qdrant:")
            logger.info(f"- Oferta más antigua: {datetime.fromtimestamp(min_time, tz=timezone.utc)} (UTC)")
            logger.info(f"- Oferta más reciente: {datetime.fromtimestamp(max_time, tz=timezone.utc)} (UTC)")
            logger.info(f"- CV subido el: {cv_upload_time}")
            
            if max_time < timestamp_threshold:
                logger.warning("\n¡ADVERTENCIA: No hay ofertas más recientes que el CV!")
                logger.warning(f"La oferta más reciente es de {datetime.fromtimestamp(max_time, tz=timezone.utc)} (UTC)")
                logger.warning(f"El CV se subió el {cv_upload_time}")
    
    except Exception as e:
        logger.error(f"Error al verificar ofertas en Qdrant: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    check_cvs()
