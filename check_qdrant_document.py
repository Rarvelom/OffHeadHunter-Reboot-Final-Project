from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

def get_document_metadata():
    # Cargar variables de entorno
    load_dotenv()
    
    # Inicializar cliente Qdrant
    qdrant_client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # ID del documento que acabamos de subir
    document_id = "8a8f7efb-5a98-43bb-9149-cd1c92ca118e"
    
    try:
        # Obtener el punto de Qdrant
        point = qdrant_client.retrieve(
            collection_name="cv_embeddings",
            ids=[document_id],
            with_vectors=False,
            with_payload=True
        )
        
        if point and len(point) > 0:
            print("\n=== Metadatos del documento en Qdrant ===")
            print(f"ID: {document_id}")
            print("\nPayload (metadatos):")
            for key, value in point[0].payload.items():
                print(f"- {key}: {value[:200]}..." if isinstance(value, str) and len(value) > 200 else f"- {key}: {value}")
            
            # Mostrar información de la colección
            collection_info = qdrant_client.get_collection("cv_embeddings")
            print("\nInformación de la colección:")
            print(f"- Nombre: cv_embeddings")
            print(f"- Tamaño de vectores: {collection_info.config.params.vectors.size if hasattr(collection_info.config.params.vectors, 'size') else 'No disponible'}")
            print(f"- Distancia: {collection_info.config.params.vectors.distance if hasattr(collection_info.config.params.vectors, 'distance') else 'No disponible'}")
            print(f"- Estado: {collection_info.status}")
            print(f"- Número de puntos: {collection_info.points_count}")
        else:
            print(f"No se encontró el documento con ID: {document_id}")
            
    except Exception as e:
        print(f"Error al recuperar el documento: {e}")

if __name__ == "__main__":
    get_document_metadata()
