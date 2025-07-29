from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
from qdrant_config import get_qdrant_client, get_collection_configs

# Load environment variables
load_dotenv()

def initialize_qdrant():
    """Initialize Qdrant client, create collections, and ensure payload indexes exist."""
    try:
        client = get_qdrant_client()
        configs = get_collection_configs()
        
        print("--- Initializing Qdrant Collections and Indexes ---")

        for collection_name, config in configs.items():
            try:
                # 1. Check if collection exists, create if not
                collection_info = client.get_collection(collection_name=collection_name)
                print(f"Collection '{collection_name}' already exists.")
            except Exception:
                print(f"Collection '{collection_name}' not found. Creating...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=config["vectors_config"]
                )
                print(f"Collection '{collection_name}' created successfully.")
                # Refresh info after creation
                collection_info = client.get_collection(collection_name=collection_name)

            # 2. Check for payload index on 'document_id', create if not
            payload_schema = collection_info.payload_schema
            if 'document_id' not in payload_schema:
                print(f"Index for 'document_id' not found in '{collection_name}'. Creating index...")
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                print(f"Index for 'document_id' in '{collection_name}' created successfully.")
            else:
                print(f"Index for 'document_id' already exists in '{collection_name}'.")

        print("\nQdrant initialization completed successfully!")
        return client
    
    except Exception as e:
        print(f"An error occurred during Qdrant initialization: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_qdrant()
