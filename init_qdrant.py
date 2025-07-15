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
        
        # Create collections
        for collection_name, config in configs.items():
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=config["vectors_config"],
                    payload_schema=config["payload_schema"]
                )
                print(f"{collection_name} collection created successfully!")
            except Exception as e:
                print(f"{collection_name} collection already exists or error: {str(e)}")
        
        print("Qdrant initialization completed successfully!")
        return client
    
    except Exception as e:
        print(f"Error initializing Qdrant: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_qdrant()
