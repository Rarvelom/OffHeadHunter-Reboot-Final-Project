import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient

# Configurar logging
logger = logging.getLogger(__name__)

def get_all_chunks(client: QdrantClient, collection_name: str, doc_id: str) -> List:
    """
    Fetches all chunks for a given document ID from a collection.
    Uses multiple strategies to ensure robust retrieval.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        doc_id: Document ID to fetch chunks for
        
    Returns:
        List of chunks for the specified document
    """
    logger.info(f"Fetching chunks for document_id: '{doc_id}' from collection {collection_name}")
    
    # Get collection info to verify it exists and check vector dimension
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        logger.info(f"Collection info: {collection_info}")
        # Try to determine vector dimension from collection config
        vector_dim = 1024  # Default BGE embedding size
        if hasattr(collection_info.config.params.vectors, 'size'):
            vector_dim = collection_info.config.params.vectors.size
            logger.info(f"Detected vector dimension: {vector_dim}")
    except Exception as e:
        logger.error(f"Error getting collection {collection_name}: {e}")
        return []
        
    # Force doc_id to string type
    doc_id = str(doc_id)
    chunks = []
    
    # Strategy 1: Try direct retrieve by payload filtering with scroll
    try:
        logger.info(f"Strategy 1: Payload filtering with scroll for {doc_id}")
        # Construct filter using match on document_id
        scroll_filter = {"must": [{"key": "document_id", "match": {"value": doc_id}}]}
        chunks, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=True,
            limit=100
        )
        logger.info(f"Strategy 1: Retrieved {len(chunks)} chunks using scroll")
        if chunks:
            return chunks
    except Exception as e:
        logger.warning(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Try search with exact match filter
    try:
        logger.info(f"Strategy 2: Search with exact match filter for {doc_id}")
        dummy_vector = [0.0] * vector_dim
        query_filter = {"must": [{"key": "document_id", "match": {"value": doc_id}}]}
        results = client.search(
            collection_name=collection_name,
            query_vector=dummy_vector,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=100
        )
        logger.info(f"Strategy 2: Retrieved {len(results)} chunks")
        if results:
            return results
    except Exception as e:
        logger.warning(f"Strategy 2 failed: {e}")
        
    # Strategy 3: Brute force - get all chunks, then filter in Python
    try:
        logger.info(f"Strategy 3: Get all chunks and filter in Python for {doc_id}")
        all_chunks, _ = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=1000
        )
        
        # Filter in Python
        filtered_chunks = []
        for chunk in all_chunks:
            if chunk.payload and chunk.payload.get('document_id') == doc_id:
                filtered_chunks.append(chunk)
        
        logger.info(f"Strategy 3: Found {len(filtered_chunks)} chunks out of {len(all_chunks)} total")
        if filtered_chunks:
            return filtered_chunks
    except Exception as e:
        logger.warning(f"Strategy 3 failed: {e}")
    
    logger.error(f"Could not retrieve chunks for document_id: {doc_id}")
    return []


def get_all_job_chunks(client: QdrantClient, collection_name: str) -> List:
    """
    Fetches all chunks from the jobs collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        
    Returns:
        List of all job chunks
    """
    try:
        logger.info(f"Fetching all job chunks from collection {collection_name}")
        
        # Try scroll approach first
        try:
            # No filter needed as we want all job chunks
            chunks, _ = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=True,
                limit=1000  # Adjust limit as needed for total job offers
            )
            logger.info(f"Retrieved {len(chunks)} job chunks using scroll method")
            return chunks
        except Exception as scroll_error:
            logger.warning(f"Scroll approach failed for jobs: {scroll_error}. Trying search approach...")
            
            # Alternative: Use search with no filtering
            results = client.search(
                collection_name=collection_name,
                query_vector=[0.0] * 1024,  # Dummy vector
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            logger.info(f"Retrieved {len(results)} job chunks using search method")
            return results
            
    except Exception as e:
        logger.error(f"Error fetching job chunks from {collection_name}: {e}")
        return []


def group_chunks_by_doc(chunks: List) -> Dict[str, List]:
    """
    Groups a list of chunks by their document_id.
    
    Args:
        chunks: List of chunks with payload containing document_id
        
    Returns:
        Dictionary mapping document_id to list of chunks
    """
    by_doc = {}
    for chunk in chunks:
        doc_id = chunk.payload.get("document_id")
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(chunk)
    return by_doc


def list_document_ids(client: QdrantClient, collection_name: str) -> List[str]:
    """
    List all unique document_ids stored in a collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        
    Returns:
        List of unique document IDs
    """
    logger.info(f"Listing all document_ids in collection {collection_name}")
    
    try:
        # Get all points with payload only
        points, _ = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1000
        )
        
        # Extract unique document IDs from payloads
        doc_ids = set()
        for point in points:
            if point.payload and 'document_id' in point.payload:
                doc_ids.add(point.payload['document_id'])
        
        logger.info(f"Found {len(doc_ids)} unique document IDs in {collection_name}: {sorted(doc_ids)}")
        return sorted(list(doc_ids))
        
    except Exception as e:
        logger.error(f"Error listing document IDs in {collection_name}: {e}")
        return []
