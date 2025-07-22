import os
import pandas as pd
from src.job_offers import JobOfferStorage
from src.qdrant_storage import QdrantStorage
from src.text_processing import TextProcessor
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_job_offers():
    """
    Carga las ofertas de trabajo desde el archivo CSV en Qdrant.
    """
    logger.info("Iniciando carga de ofertas de trabajo...")
    job_storage = JobOfferStorage(collection_name="job_embeddings_BGE")
    
    # Ruta al archivo CSV de ofertas
    csv_path = os.path.join(os.path.dirname(__file__), "uploads", "job_descriptions", "job_title_des.csv")
    
    try:
        # Leer el CSV
        df = pd.read_csv(csv_path)
        
        # Procesar cada fila del CSV
        for _, row in df.iterrows():
            try:
                # Crear el diccionario con los datos de la oferta
                job_offer = {
                    "title": row.get("title", "Sin título"),
                    "description": row.get("description", ""),
                    "skills": row.get("skills", ""),
                    "location": row.get("location", ""),
                    "company": row.get("company", "")
                }
                
                # Guardar en Qdrant
                result = job_storage.save_offer(job_offer)
                if result:
                    logger.info(f"Oferta guardada exitosamente: {job_offer['title']}")
                else:
                    logger.error(f"Error al guardar oferta: {job_offer['title']}")
                    
            except Exception as e:
                logger.error(f"Error procesando oferta: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error al leer el archivo CSV: {str(e)}")

def load_cvs():
    """
    Carga los CVs desde el archivo CSV en Qdrant.
    """
    logger.info("Iniciando carga de CVs...")
    cv_storage = QdrantStorage(collection_name="cv_embeddings_BGE")
    text_processor = TextProcessor()
    
    # Ruta al archivo CSV de CVs
    csv_path = os.path.join(os.path.dirname(__file__), "uploads", "cvs", "UpdatedResumeDataSet.csv")
    
    try:
        # Leer el CSV
        df = pd.read_csv(csv_path)
        
        # Procesar cada fila del CSV
        for _, row in df.iterrows():
            try:
                # Extraer el texto del CV
                cv_text = row.get("Resume", "")
                
                # Generar el embedding
                embedding = text_processor.model.encode(cv_text)
                
                # Crear el diccionario con los datos del CV
                cv_data = {
                    "text": cv_text,
                    "embedding": embedding,
                    "metadata": {
                        "category": row.get("Category", ""),
                        "skills": row.get("Skills", "")
                    }
                }
                
                # Guardar en Qdrant
                result = cv_storage.save_document(cv_data)
                if result:
                    logger.info(f"CV guardado exitosamente")
                else:
                    logger.error(f"Error al guardar CV")
                    
            except Exception as e:
                logger.error(f"Error procesando CV: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error al leer el archivo CSV: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("Iniciando proceso de carga de datos...")
        
        # Cargar ofertas de trabajo
        load_job_offers()
        
        # Cargar CVs
        load_cvs()
        
        logger.info("Proceso de carga completado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error en el proceso de carga: {str(e)}")
