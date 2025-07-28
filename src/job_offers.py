"""
Módulo para el manejo específico de ofertas de trabajo en Qdrant.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from qdrant_client.http import models
from src.qdrant_storage import QdrantStorage
from src.text_processing import TextProcessor
from src.pdf_processor import PDFProcessor
from src.utils.time_utils import get_current_utc_timestamp, to_iso_format, to_unix_timestamp
import uuid

logger = logging.getLogger(__name__)

class JobOfferStorage:
    """Clase para manejar el almacenamiento de ofertas de trabajo en Qdrant."""
    
    def __init__(self, collection_name: str = "job_embeddings_BGE"):
        """
        Inicializa el almacenamiento de ofertas.
        
        Args:
            collection_name: Nombre de la colección en Qdrant (por defecto: "job_embeddings")
        """
        self.storage = QdrantStorage(collection_name=collection_name)
        self.text_processor = TextProcessor()
        self.pdf_processor = PDFProcessor()
    
    def save_offer(self, job_offer: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Guarda una oferta de trabajo en Qdrant.
        
        Args:
            job_offer: Diccionario con los datos de la oferta
            
        Returns:
            Dict: Diccionario con la información del embedding o None si hubo un error
        """
        try:
            logger.info(f"Iniciando guardado de oferta: {job_offer.get('title', 'Sin título')}")
            
            # Extraer texto del PDF si está disponible
            pdf_text = ""
            if 'pdf_file' in job_offer and job_offer['pdf_file'] is not None:
                try:
                    logger.info("Procesando PDF adjunto...")
                    pdf_data = self.pdf_processor.process_pdf(job_offer['pdf_file'])
                    pdf_text = pdf_data.get('text', '')
                    logger.info(f"PDF procesado. Texto extraído: {len(pdf_text)} caracteres")
                except Exception as e:
                    logger.error(f"Error al procesar PDF: {e}", exc_info=True)
            
            # Generar embedding del texto del PDF o del título si no hay PDF
            text_to_embed = pdf_text if pdf_text else job_offer.get('title', '') + ' ' + job_offer.get('description', '')
            
            try:
                # Generar el embedding
                embedding = self.text_processor.generate_embeddings([text_to_embed])[0].tolist()
                
                # Generar un nuevo UUID para Qdrant
                point_id = str(uuid.uuid4())
                
                # Preparar metadatos completos usando el método _prepare_metadata
                has_pdf = bool(pdf_text)
                payload = self._prepare_metadata(job_offer, has_pdf)
                
                # Asegurarse de que los campos básicos estén presentes
                if 'user_id' not in payload or not payload['user_id']:
                    payload['user_id'] = job_offer.get('user_id')
                if 'title' not in payload or not payload['title']:
                    payload['title'] = job_offer.get('title', 'Sin título')
                if 'company' not in payload or not payload['company']:
                    payload['company'] = job_offer.get('company', 'Empresa no especificada')
                if 'source_url' not in payload or not payload['source_url']:
                    payload['source_url'] = job_offer.get('url', '')
                if 'has_pdf' not in payload:
                    payload['has_pdf'] = has_pdf
                    
                # Asegurar que los campos de timestamp estén presentes y en el formato correcto
                current_time = get_current_utc_timestamp()
                
                # Establecer created_at si no existe
                if 'created_at' not in payload or not payload['created_at']:
                    payload['created_at'] = current_time
                else:
                    # Convertir a formato unix si es necesario
                    payload['created_at'] = to_unix_timestamp(payload['created_at'])
                
                # Asegurar que scraped_at esté en formato unix
                if 'scraped_at' in payload and payload['scraped_at']:
                    payload['scraped_at'] = to_unix_timestamp(payload['scraped_at'])
                else:
                    payload['scraped_at'] = current_time
                    
                # Añadir campo de actualización
                payload['updated_at'] = current_time
                
                logger.info(f"Insertando oferta en Qdrant con ID: {point_id}")
                
                # Insertar en Qdrant
                self.storage.client.upsert(
                    collection_name=self.storage.collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
                
                logger.info(f"Oferta guardada exitosamente en Qdrant con ID: {point_id}")
                
                # Devolver solo la información del embedding
                return {
                    "embedding_vector_id_qdrant": point_id,
                    "embedding_model": "BAAI/bge-m3"
                }
                
            except Exception as e:
                logger.error(f"Error al guardar en Qdrant: {e}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"Error en save_offer: {e}", exc_info=True)
            return None
    
    def _prepare_metadata(self, job_offer: Dict[str, Any], has_pdf: bool) -> Dict[str, Any]:
        """Prepara los metadatos para almacenar en Qdrant."""
        try:
            # Asegurarnos de que job_offer es un diccionario
            if not isinstance(job_offer, dict):
                logger.error(f"job_offer no es un diccionario: {type(job_offer)}")
                job_offer = {}

            # Extraer ubicaciones de manera segura
            locations = []
            if 'locations' in job_offer and job_offer['locations'] is not None:
                if isinstance(job_offer['locations'], list):
                    locations = [
                        loc.get('city', '') 
                        for loc in job_offer['locations'] 
                        if isinstance(loc, dict) and 'city' in loc
                    ]
                elif isinstance(job_offer['locations'], dict):
                    locations = [job_offer['locations'].get('city', '')]

            # Extraer información salarial de manera segura
            salary_info = {}
            if 'salary_range' in job_offer and job_offer['salary_range']:
                salary = job_offer['salary_range']
                if isinstance(salary, dict):
                    salary_info = {
                        "min": salary.get('min_value'),
                        "max": salary.get('max_value'),
                        "currency": salary.get('currency', 'EUR'),
                        "period": salary.get('period', 'year')
                    }

            # Obtener la fecha actual
            now = datetime.utcnow()
            now_iso = now.isoformat()
            now_timestamp = int(now.timestamp())  # Timestamp en segundos
            
            # Construir metadatos
            metadata = {
                "title": str(job_offer.get('title', '')),
                "company": str(job_offer.get('company', '')),
                "url": str(job_offer.get('url', '')),
                "source": "infojobs",
                "mongodb_id": str(job_offer.get('_id', '')),
                "created_at": now_iso,
                "scraped_at": now_timestamp,  # Añadido para filtrado por timestamp
                "has_pdf": has_pdf,
                "description": str(job_offer.get('description', ''))[:1000],
                "locations": locations,
                "user_id": job_offer.get('user_id', ''),
            }

            if salary_info:
                metadata["salary"] = salary_info

            return metadata

        except Exception as e:
            logger.error(f"Error en _prepare_metadata: {e}", exc_info=True)
            # Devolver un diccionario mínimo en caso de error
            return {
                "title": "Error en metadatos",
                "error": str(e)
            }
    
    def _prepare_text_for_embedding(self, job_offer: Dict[str, Any], pdf_text: str = "") -> str:
        """Combina metadatos y contenido del PDF para generar el embedding."""
        try:
            # Asegurarnos de que job_offer es un diccionario
            if not isinstance(job_offer, dict):
                logger.warning("job_offer no es un diccionario en _prepare_text_for_embedding")
                job_offer = {}

            # Obtener ubicaciones de manera segura
            locations = []
            if 'locations' in job_offer and job_offer['locations'] is not None:
                if isinstance(job_offer['locations'], list):
                    locations = [
                        loc if isinstance(loc, str) else str(loc)
                        for loc in job_offer['locations']
                        if loc  # Solo incluir ubicaciones no vacías
                    ]
                elif isinstance(job_offer['locations'], dict):
                    # Si es un diccionario, extraer valores
                    locations = [str(val) for val in job_offer['locations'].values() if val]
                elif isinstance(job_offer['locations'], str):
                    # Si es una cadena simple, usarla directamente
                    locations = [job_offer['locations']]

            # Construir partes del texto
            parts = [
                f"Título: {job_offer.get('title', '')}",
                f"Empresa: {job_offer.get('company', '')}",
                f"Ubicación: {', '.join(locations) if locations else 'No especificada'}",
                f"Descripción: {job_offer.get('description', '')}",
            ]
            
            # Añadir contenido del PDF (limitado a 2000 caracteres para no exceder el límite de contexto)
            if pdf_text:
                parts.append(f"Contenido del PDF: {pdf_text[:2000]}")
            
            # Añadir etiquetas si existen
            if tags := job_offer.get('tags', []):
                if isinstance(tags, (list, tuple)):
                    parts.append(f"Etiquetas: {', '.join(str(tag) for tag in tags)}")
                else:
                    parts.append(f"Etiquetas: {str(tags)}")
            
            # Añadir información salarial
            if salary := job_offer.get('salary_range', {}):
                if isinstance(salary, dict):
                    min_s = salary.get('min_value', 'N/A')
                    max_s = salary.get('max_value', 'N/A')
                    currency = salary.get('currency', 'EUR')
                    period = salary.get('period', 'year')
                    parts.append(f"Salario: {min_s}-{max_s} {currency}/{period}")
                else:
                    parts.append(f"Salario: {str(salary)}")
            
            return '\n'.join(part for part in parts if part)
            
        except Exception as e:
            logger.error(f"Error en _prepare_text_for_embedding: {e}", exc_info=True)
            # Devolver al menos la información básica en caso de error
            return f"Título: {job_offer.get('title', '')}\nEmpresa: {job_offer.get('company', '')}"
    
    def search_similar_offers(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca ofertas de trabajo similares a un embedding de consulta.
        
        Args:
            query_embedding: Embedding de la consulta.
            limit: Número máximo de resultados a devolver.
            filter_conditions: Condiciones de filtrado (ej: {"company": "Google"}).
            score_threshold: Umbral de similitud mínimo (0-1).
            
        Returns:
            Lista de ofertas similares con sus metadatos y puntuaciones.
        """
        return self.storage.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions,
            score_threshold=score_threshold
        )
    
    def delete_offer(self, offer_id: str) -> bool:
        """
        Elimina una oferta de Qdrant.
        
        Args:
            offer_id: ID de la oferta a eliminar.
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario.
        """
        return self.storage.delete_document(offer_id)
