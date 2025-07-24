#!/usr/bin/env python3
"""
Script para limpiar las colecciones de MongoDB.
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_mongodb():
    """Limpia las colecciones de MongoDB."""
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Obtener la URI de conexión de MongoDB
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            logger.error("MONGODB_URI no está configurado en el archivo .env")
            return False
        
        # Conectar a MongoDB
        client = MongoClient(mongodb_uri)
        db = client.get_database("offheadhunter_db")
        
        # Lista de colecciones a limpiar
        collections_to_clean = [
            "cv_uploads",
            "agent_test_queries",
            "job_offers",
            "job_searches"
        ]
        
        # Limpiar cada colección
        for collection_name in collections_to_clean:
            if collection_name in db.list_collection_names():
                collection = db[collection_name]
                count = collection.count_documents({})
                if count > 0:
                    collection.delete_many({})
                    logger.info(f"Se eliminaron {count} documentos de la colección '{collection_name}'")
                else:
                    logger.info(f"La colección '{collection_name}' ya está vacía")
            else:
                logger.warning(f"La colección '{collection_name}' no existe")
        
        logger.info("Limpieza de MongoDB completada con éxito")
        return True
        
    except Exception as e:
        logger.error(f"Error al limpiar MongoDB: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== LIMPIEZA DE MONGODB ===")
    print("Este script eliminará todos los datos de las colecciones de MongoDB.")
    confirm = input("¿Estás seguro de que deseas continuar? (s/n): ")
    
    if confirm.lower() == 's':
        success = clean_mongodb()
        if success:
            print("\n✅ Limpieza completada con éxito")
        else:
            print("\n❌ Hubo un error durante la limpieza. Revisa los logs para más detalles.")
    else:
        print("Operación cancelada por el usuario")
