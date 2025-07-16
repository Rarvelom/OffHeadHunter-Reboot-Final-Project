from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
from qdrant_config import get_qdrant_client, get_collection_configs

# Load environment variables
load_dotenv()

def initialize_qdrant():
    """Initialize Qdrant client and create collections"""
    try:
        # Get Qdrant client
        client = get_qdrant_client()
        
        # Get collection configurations
        configs = get_collection_configs()
        
        # Get existing collections
        existing_collections = client.get_collections().collections
        existing_collection_names = [col.name for col in existing_collections]
        
        # Create collections if they don't exist
        for collection_name, config in configs.items():
            try:
                if collection_name not in existing_collection_names:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=config["vectors_config"]
                    )
                    print(f"Collection '{collection_name}' created successfully!")
                else:
                    print(f"Collection '{collection_name}' already exists.")
            except Exception as e:
                print(f"Error creating collection '{collection_name}': {str(e)}")
        
        print("Qdrant initialization completed successfully!")
        return client
    
    except Exception as e:
        print(f"Error initializing Qdrant: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_qdrant()
