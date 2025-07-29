import os
import argparse
import logging
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from typing import List, Dict

# Importar configuración y utilidades centralizadas
from qdrant_config import get_qdrant_client, CV_COLLECTION, JOB_COLLECTION
from src.qdrant_utils import get_all_chunks, get_all_job_chunks, group_chunks_by_doc, list_document_ids

# --- CONFIG ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- HELPERS ---
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot_product / (norm_product + 1e-8))
# --- CORE LOGIC ---
def match_cv_to_jobs(client: QdrantClient, cv_id: str, cv_collection: str, job_collection: str, specific_job_ids: List[str] = None) -> List[Dict[str, any]]:
    """Matches a CV to all available job offers and returns a scored list."""
    # Retrieve CV chunks
    cv_chunks = get_all_chunks(client, cv_collection, cv_id)
    if not cv_chunks:
        logger.warning(f"No chunks found for CV {cv_id}. Cannot perform matching.")
        return []
    logger.info(f"Found {len(cv_chunks)} chunks for CV {cv_id}.")
    
    # Debug - print first chunk to check format
    if len(cv_chunks) > 0:
        logger.info(f"Sample CV chunk: {cv_chunks[0].payload.get('text', '')[:100]}...")
        logger.info(f"CV chunk vector type: {type(cv_chunks[0].vector)}, length: {len(cv_chunks[0].vector)}")

    # Retrieve job chunks
    all_job_chunks = get_all_job_chunks(client, job_collection)
    if not all_job_chunks:
        logger.warning(f"No job offers found in collection {job_collection}. Cannot perform matching.")
        return []
    logger.info(f"Found {len(all_job_chunks)} total job chunks to compare against.")
    
    # Debug - print first job chunk to check format
    if len(all_job_chunks) > 0:
        logger.info(f"Sample job chunk: {all_job_chunks[0].payload.get('text', '')[:100]}...")
        logger.info(f"Job chunk vector type: {type(all_job_chunks[0].vector)}, length: {len(all_job_chunks[0].vector)}")

    # Group job chunks by document ID
    jobs_by_id = group_chunks_by_doc(all_job_chunks)
    logger.info(f"Grouped job chunks into {len(jobs_by_id)} unique job offers.")
    
    # Filter by specific job IDs if provided
    if specific_job_ids:
        logger.info(f"Filtering to specific job IDs: {specific_job_ids}")
        filtered_jobs_by_id = {job_id: chunks for job_id, chunks in jobs_by_id.items() 
                              if job_id in specific_job_ids}
        logger.info(f"After filtering: {len(filtered_jobs_by_id)} job offers to match against")
        if not filtered_jobs_by_id:
            logger.warning(f"None of the specified job IDs found in collection. Available IDs: {list(jobs_by_id.keys())[:10]}...")
            return []
        jobs_by_id = filtered_jobs_by_id
    
    # IMPROVED SCORING ALGORITHM
    job_scores = {}
    job_match_details = {}
    
    for job_id, job_chunks_list in jobs_by_id.items():
        logger.info(f"Processing job {job_id} with {len(job_chunks_list)} chunks")
        
        # 1. For each CV chunk, find best matching job chunk
        cv_chunk_scores = []
        top_matches = []
        
        for cv_idx, cv_chunk in enumerate(cv_chunks):
            chunk_matches = []
            
            for job_idx, job_chunk in enumerate(job_chunks_list):
                try:
                    similarity = cosine_similarity(cv_chunk.vector, job_chunk.vector)
                    chunk_matches.append({
                        "job_chunk_idx": job_idx,
                        "similarity": similarity,
                        "cv_text": cv_chunk.payload.get("text", "")[:100],
                        "job_text": job_chunk.payload.get("text", "")[:100]
                    })
                except Exception as e:
                    logger.error(f"Error computing similarity: {e}")
            
            # Find best match for this CV chunk
            if chunk_matches:
                best_match = max(chunk_matches, key=lambda x: x["similarity"])
                cv_chunk_scores.append(best_match["similarity"])
                top_matches.append(best_match)
            else:
                cv_chunk_scores.append(0.0)
        
        # 2. Calculate weighted score - emphasize top matches more
        if cv_chunk_scores:
            # Sort scores in descending order
            sorted_scores = sorted(cv_chunk_scores, reverse=True)
            
            # Calculate weighted average (higher weight for better matches)
            weighted_total = 0
            weight_sum = 0
            
            for i, score in enumerate(sorted_scores):
                # Weight decreases as we go down the list
                weight = 1.0 / (1 + i * 0.2)  # Weight formula: less reduction for top scores
                weighted_total += score * weight
                weight_sum += weight
            
            weighted_score = weighted_total / weight_sum if weight_sum > 0 else 0
            
            # Store final score
            job_scores[job_id] = weighted_score
            
            # Store match details for debugging
            job_match_details[job_id] = {
                "weighted_score": weighted_score,
                "average_score": sum(cv_chunk_scores) / len(cv_chunk_scores),
                "max_score": max(cv_chunk_scores) if cv_chunk_scores else 0,
                "top_matches": top_matches[:3]  # Keep top 3 matches for debugging
            }
            
            logger.info(f"Job {job_id}: weighted_score={weighted_score:.4f}, avg={job_match_details[job_id]['average_score']:.4f}, max={job_match_details[job_id]['max_score']:.4f}")
    
    # Sort all jobs by weighted score, descending
    sorted_jobs = sorted(job_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Format the output with additional information
    all_matches = []
    for job_id, score in sorted_jobs:
        match_info = {
            "job_id": job_id,
            "score": score,
            # Add these fields for debugging but comment them out in production
            # "max_score": job_match_details[job_id]["max_score"],
            # "avg_score": job_match_details[job_id]["average_score"]
        }
        all_matches.append(match_info)
    
    logger.info(f"Finished matching. Found {len(all_matches)} potential matches.")
    return all_matches

# --- UTILITIES FOR DEBUGGING ---
# Función movida a src/qdrant_utils.py

# --- MAIN EXECUTION ---
def main(args_list=None):
    """Main function to run the job matching script from the command line.
    
    Args:
        args_list: Optional list of command line arguments. If None, sys.argv[1:] is used.
    """
    parser = argparse.ArgumentParser(description="Match a CV to the best job offers.")
    parser.add_argument('--cv_id', type=str, required=True, help='The document ID of the CV to match.')
    parser.add_argument('--cv_collection', type=str, default="cv_embeddings_BGE2", help='Qdrant collection for CVs.')
    parser.add_argument('--job_collection', type=str, default="job_embeddings_BGE2", help='Qdrant collection for jobs.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top job matches to return.')
    # Se espera que los IDs vengan como un solo string separado por comas
    parser.add_argument('--job_ids', type=str, help='Optional: Comma-separated string of specific job IDs to match against')
    args = parser.parse_args(args_list)

    try:
        # Usar cliente centralizado con configuración optimizada
        client = get_qdrant_client(use_http=True, timeout=60.0)
        client.get_collections() # Verify connection
    except Exception as e:
        logger.critical(f"Failed to connect to Qdrant. Error: {e}")
        return

    # Debug document IDs
    logger.info(f"CV ID to find: {args.cv_id}")
    cv_doc_ids = list_document_ids(client, args.cv_collection)
    job_doc_ids = list_document_ids(client, args.job_collection)

    # Try to match with both original and alternative formats
    cv_found = False
    if args.cv_id in cv_doc_ids:
        logger.info(f"Found exact match for CV ID: {args.cv_id}")
        cv_found = True
    elif cv_doc_ids:  # If we have any CV documents
        # Maybe the CV ID has been processed differently (e.g., stem vs. full name)
        logger.warning(f"Could not find exact CV ID: {args.cv_id} in collection")
        logger.info(f"Available CV IDs: {cv_doc_ids}")
        
        # Let's find the most similar document ID
        import difflib
        closest_match = difflib.get_close_matches(args.cv_id, cv_doc_ids, n=1, cutoff=0.1)
        if closest_match:
            alternative_id = closest_match[0]
            logger.warning(f"Using alternative CV ID: {alternative_id} instead of {args.cv_id}")
            args.cv_id = alternative_id
            cv_found = True

    if not cv_found:
        logger.error(f"Could not find any matching CV ID. Aborting match.")
        print("[]")
        return

    # Parse the job_ids string into a list
    specific_job_ids = []
    if args.job_ids:
        specific_job_ids = [job_id.strip() for job_id in args.job_ids.split(',')]

    matches = match_cv_to_jobs(client, args.cv_id, args.cv_collection, args.job_collection, specific_job_ids)

    import json
    # Devolver solo los top_k resultados
    print(json.dumps(matches[:args.top_k]))

if __name__ == "__main__":
    main()