import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def get_cv_metadata(cv_id):
    """
    Obtiene los metadatos de un CV específico de Qdrant
    """
    # Cargar variables de entorno
    load_dotenv()
    
    # Inicializar cliente Qdrant
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        prefer_grpc=True
    )
    
    # Obtener el punto (documento) específico
    try:
        record = client.retrieve(
            collection_name="cv_embeddings_BGE",
            ids=[cv_id],
            with_vectors=False,
            with_payload=True
        )
        
        if not record:
            print(f"No se encontró el CV con ID: {cv_id}")
            return None
            
        return record[0]
    except Exception as e:
        print(f"Error al obtener el CV: {str(e)}")
        return None

if __name__ == "__main__":
    # ID del CV que se cargó anteriormente
    cv_id = "cb60a3b3-9220-42dd-8dd6-4ea06543e971"
    
    print(f"Obteniendo metadatos del CV con ID: {cv_id}")
    cv_metadata = get_cv_metadata(cv_id)
    
    if cv_metadata:
        print("\n=== METADATOS DEL CV ===")
        print(f"ID: {cv_metadata.id}")
        print("\nPayload:")
        for key, value in cv_metadata.payload.items():
            print(f"- {key}: {value}")
    else:
        print("No se pudieron obtener los metadatos del CV")
