from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

def check_qdrant_metadata(vector_id):
    """
    Check metadata for a specific vector in Qdrant
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    
    try:
        # Get the point from Qdrant
        points = qdrant_client.retrieve(
            collection_name="cv_embeddings",
            ids=[vector_id],
            with_vectors=False,
            with_payload=True
        )
        
        if not points:
            print(f"No se encontró el vector con ID: {vector_id}")
            return
            
        point = points[0]
        
        print("\n=== Metadatos del CV en Qdrant ===")
        print(f"ID del vector: {vector_id}")
        
        if hasattr(point, 'payload') and point.payload:
            print("\nPayload (metadatos):")
            for key, value in point.payload.items():
                if key == 'text':
                    preview = value[:200] + '...' if len(value) > 200 else value
                    print(f"- {key}: {preview}")
                else:
                    print(f"- {key}: {value}")
        
        # Get collection info
        collection_info = qdrant_client.get_collection("cv_embeddings")
        print("\nInformación de la colección:")
        print(f"- Nombre: cv_embeddings")
        print(f"- Tamaño de vectores: {collection_info.config.params.vectors.size}")
        print(f"- Distancia: {collection_info.config.params.vectors.distance}")
        print(f"- Puntos en la colección: {collection_info.points_count}")
        
    except Exception as e:
        print(f"Error al consultar Qdrant: {e}")

if __name__ == "__main__":
    # Use the vector ID from the last upload
    VECTOR_ID = "a44eacbf-a646-4eb2-9146-9898f331b52c"
    check_qdrant_metadata(VECTOR_ID)
