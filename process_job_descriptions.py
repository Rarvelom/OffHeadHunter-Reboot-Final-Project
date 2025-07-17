import os
import sys
import pandas as pd
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the local modules
from src.qdrant_storage import QdrantStorage
from qdrant_config import get_qdrant_client

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

def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Google's text-embedding-004 model."""
    try:
        # Use the text-embedding-004 model which produces 1536-dimensional vectors
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def process_job_descriptions(csv_path: str, batch_size: int = 10):
    """Process job descriptions from CSV and add them to Qdrant."""
    # Initialize Qdrant storage for job_embeddings collection
    storage = QdrantStorage(collection_name="job_embeddings")
    
    # Read the CSV file
    logger.info(f"Reading job descriptions from {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Process in batches
    total_rows = len(df)
    for i in tqdm(range(0, total_rows, batch_size), desc="Processing job descriptions"):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare chunks for this batch
        chunks = []
        for _, row in batch.iterrows():
            job_title = str(row['Job Title']) if 'Job Title' in row else ""
            job_description = str(row['Job Description']) if 'Job Description' in row else ""
            
            # Combine title and description for embedding
            text = f"{job_title}: {job_description}"
            
            # Skip if text is too short
            if len(text) < 10:
                continue
                
            # Get embedding
            embedding = get_embedding(text)
            if not embedding:
                continue
                
            # Create chunk
            chunk = {
                'text': text,
                'embedding': embedding,
                'num_tokens': len(text.split())  # Approximate token count
            }
            chunks.append(chunk)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Store chunks in Qdrant
        if chunks:
            document_id = f"job_batch_{i//batch_size}"
            metadata = {
                'source': 'job_descriptions',
                'batch_id': i//batch_size,
                'processed_at': pd.Timestamp.now().isoformat()
            }
            
            try:
                stored_ids = storage.store_embeddings(
                    document_id=document_id,
                    chunks=chunks,
                    metadata=metadata
                )
                logger.info(f"Stored {len(stored_ids)} job descriptions in batch {i//batch_size}")
            except Exception as e:
                logger.error(f"Error storing batch {i//batch_size}: {str(e)}")
    
    logger.info("Finished processing job descriptions")

if __name__ == "__main__":
    # Path to the processed job descriptions CSV
    csv_path = "job_title_des_processed.csv"
    
    # Process job descriptions
    process_job_descriptions(csv_path, batch_size=5)  # Small batch size to avoid rate limiting
