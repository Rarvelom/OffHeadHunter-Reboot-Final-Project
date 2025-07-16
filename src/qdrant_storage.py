import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantStorage:
    """Clase para manejar el almacenamiento y consulta de embeddings en Qdrant."""
    
    def __init__(self, collection_name: str = "cv_embeddings"):
        """
        Inicializa el cliente de Qdrant.
        
        Args:
            collection_name: Nombre de la colección a utilizar.
        """
        from qdrant_config import get_qdrant_client
        
        self.client = get_qdrant_client()
        self.collection_name = collection_name
        
        # Verificar si la colección existe, si no, crearla
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            logger.warning(f"La colección '{self.collection_name}' no existe. Se creará automáticamente.")
            self._create_collection()
    
    def _create_collection(self):
        """Crea una nueva colección en Qdrant con la configuración adecuada."""
        from qdrant_config import get_collection_configs
        
        configs = get_collection_configs()
        
        if self.collection_name not in configs:
            raise ValueError(f"No se encontró configuración para la colección '{self.collection_name}'")
        
        config = configs[self.collection_name]
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=config["vectors_config"],
            # No es necesario especificar payload_schema aquí, ya que Qdrant es schema-less
        )
        
        logger.info(f"Colección '{self.collection_name}' creada exitosamente.")
    
    def store_embeddings(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        batch_size: int = 32
    ) -> List[str]:
        """
        Almacena los embeddings de los chunks en Qdrant.
        
        Args:
            document_id: ID único del documento.
            chunks: Lista de chunks con embeddings (debe incluir el campo 'embedding' y 'text').
            metadata: Metadatos adicionales para almacenar con los puntos.
            user_id: ID del usuario propietario del documento (opcional).
            batch_size: Tamaño del lote para la carga de puntos.
            
        Returns:
            Lista de IDs de los puntos almacenados.
        """
        if not chunks:
            logger.warning("No se proporcionaron chunks para almacenar.")
            return []
        
        # Preparar los puntos para cargar en Qdrant
        points = []
        stored_ids = []
        
        for i, chunk in enumerate(chunks):
            if "embedding" not in chunk:
                logger.warning(f"El chunk {i} no tiene un embedding. Se omitirá.")
                continue
                
            # Generar un ID único para el punto
            point_id = str(uuid.uuid4())
            stored_ids.append(point_id)
            
            # Crear el payload con metadatos
            payload = {
                "document_id": document_id,
                "text": chunk.get("text", ""),
                "chunk_index": i,
                "num_tokens": chunk.get("num_tokens", 0),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Añadir metadatos adicionales si existen
            if metadata:
                payload.update(metadata)
            
            # Añadir user_id si está disponible
            if user_id:
                payload["user_id"] = user_id
            
            # Crear el punto para Qdrant
            point = PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload
            )
            
            points.append(point)
        
        # Cargar los puntos en lotes
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
        
        logger.info(f"Se almacenaron {len(stored_ids)} chunks del documento {document_id} en Qdrant.")
        return stored_ids
    
    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca los chunks más similares a un embedding de consulta.
        
        Args:
            query_embedding: Embedding de la consulta.
            limit: Número máximo de resultados a devolver.
            filter_conditions: Condiciones de filtrado (ej: {"user_id": "123"}).
            score_threshold: Umbral de similitud mínimo (0-1).
            
        Returns:
            Lista de resultados con los chunks más similares y sus puntuaciones.
        """
        # Construir filtros si se proporcionan
        filter_condition = None
        if filter_conditions:
            must_conditions = []
            
            for field, value in filter_conditions.items():
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            
            filter_condition = Filter(must=must_conditions)
        
        # Realizar la búsqueda
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_condition,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Formatear resultados
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "vector": hit.vector
            }
            results.append(result)
        
        return results
    
    def delete_document(self, document_id: str) -> int:
        """
        Elimina todos los chunks de un documento.
        
        Args:
            document_id: ID del documento a eliminar.
            
        Returns:
            Número de puntos eliminados.
        """
        # Crear filtro para el document_id
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        # Eliminar los puntos que coincidan con el filtro
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_condition
        )
        
        logger.info(f"Se eliminaron los chunks del documento {document_id}.")
        return result.operation_id if hasattr(result, 'operation_id') else 0


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia del almacenamiento
    storage = QdrantStorage(collection_name="cv_embeddings")
    
    # Ejemplo de búsqueda (requiere un embedding de consulta)
    try:
        # Crear un embedding de ejemplo (vector de 1536 dimensiones con valores aleatorios)
        example_embedding = [0.0] * 1536  # Reemplazar con un embedding real
        
        # Buscar documentos similares
        results = storage.search_similar(
            query_embedding=example_embedding,
            limit=3,
            filter_conditions={"user_id": "usuario123"},
            score_threshold=0.7
        )
        
        print(f"Resultados de la búsqueda: {len(results)} documentos encontrados")
        for i, result in enumerate(results):
            print(f"\nResultado {i+1} (score: {result['score']:.3f}):")
            print(f"Document ID: {result['payload'].get('document_id')}")
            print(f"Texto: {result['payload'].get('text', '')[:200]}...")
            
    except Exception as e:
        print(f"Error en la búsqueda: {str(e)}")
