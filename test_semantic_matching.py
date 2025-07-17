import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / 'src'
sys.path.append(str(project_root))  # Add project root to path
sys.path.append(str(src_path))      # Add src directory to path

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_config import get_qdrant_client
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import random
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google's Generative AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

def get_cv_embedding(text: str) -> List[float]:
    """Get embedding for CV text using text-embedding-004 model."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",  
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error getting CV embedding: {str(e)}")
        # Print available models for debugging
        try:
            models = genai.list_models()
            logger.info("Available models:")
            for model in models:
                if 'embed' in model.name.lower():
                    logger.info(f"- {model.name} (supports: {', '.join(method for method in model.supported_generation_methods)})")
        except Exception as e2:
            logger.error(f"Error listing models: {str(e2)}")
        return None

def find_similar_jobs(cv_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find similar jobs for a given CV text."""
    # Get embedding for the CV
    cv_embedding = get_cv_embedding(cv_text)
    if not cv_embedding:
        logger.error("Failed to generate CV embedding")
        return []
    
    # Initialize Qdrant client
    client = get_qdrant_client()
    
    # Search for similar jobs
    search_result = client.search(
        collection_name="job_embeddings",
        query_vector=cv_embedding,
        limit=top_k,
        with_vectors=False,
        with_payload=True,
        score_threshold=0.5  # Adjust threshold as needed
    )
    
    # Format results
    results = []
    for hit in search_result:
        results.append({
            'job_id': hit.id,
            'score': hit.score,
            'title': hit.payload.get('text', '').split(':')[0],
            'description': hit.payload.get('text', '')[:200] + '...',
            'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
        })
    
    return results

def get_random_cv() -> Dict[str, Any]:
    """Get a random CV from the cv_embeddings collection."""
    import random
    client = get_qdrant_client()
    
    # First, get all CVs with minimal data (just IDs)
    result = client.scroll(
        collection_name="cv_embeddings",
        limit=1000,  # Adjust based on your collection size
        with_vectors=False,
        with_payload=False
    )
    
    if not result[0]:
        raise ValueError("No CVs found in the collection")
    
    # Get list of all CV IDs
    all_cv_ids = [point.id for point in result[0]]
    
    if not all_cv_ids:
        raise ValueError("No CV IDs found in the collection")
    
    # Select a random CV ID
    random_cv_id = random.choice(all_cv_ids)
    
    # Get the full CV data for the selected ID
    cv_data = client.retrieve(
        collection_name="cv_embeddings",
        ids=[random_cv_id],
        with_vectors=False
    )
    
    if not cv_data:
        raise ValueError(f"Failed to retrieve CV with ID {random_cv_id}")
    
    cv = cv_data[0]
    return {
        'cv_id': cv.id,
        'text': cv.payload.get('text', ''),
        'metadata': {k: v for k, v in cv.payload.items() if k != 'text'}
    }

def main():
    try:
        # Get a random CV
        print("\nFetching a random CV...")
        cv = get_random_cv()
        print(f"\nCV ID: {cv['cv_id']}")
        print(f"CV Text Preview: {cv['text'][:200]}...")
        
        # Find similar jobs
        print("\nFinding similar jobs...")
        similar_jobs = find_similar_jobs(cv['text'])
        
        # Display results
        print(f"\nTop {len(similar_jobs)} Matching Jobs:")
        print("-" * 80)
        
        for i, job in enumerate(similar_jobs, 1):
            print(f"\nJob #{i}")
            print(f"Match Score: {job['score']:.4f}")
            print(f"Title: {job['title']}")
            print(f"Description: {job['description']}")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
