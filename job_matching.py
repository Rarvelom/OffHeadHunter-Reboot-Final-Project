import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BGE_M3_DIMENSION = 1024

class BGE_M3_Embedder:
    """Handles text embeddings using BAAI/bge-m3 model with error handling and device management"""
    
    def __init__(self, model_name: str = BGE_M3_MODEL_NAME):
        """Initialize the BGE-M3 model with error handling and device selection"""
        try:
            logger.info(f"Loading BGE-M3 model: {model_name}")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True
            )
            self.model.max_seq_length = 512  # Optimize for most use cases
            self.dimension = BGE_M3_DIMENSION
            logger.info(f"BGE-M3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {str(e)}")
            raise
        
    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with error handling and batching
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []
            
        try:
            # Process in batches to handle memory constraints
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), 
                         desc="Generating embeddings", 
                         unit="batch"):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch, 
                    batch_size=len(batch),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False
                )
                embeddings.extend(batch_embeddings.tolist())
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return empty embeddings for failed batch to avoid breaking the pipeline
            return [[]] * len(texts)

class QdrantManager:
    """Manages Qdrant vector database operations with project configuration"""
    
    def __init__(self):
        """Initialize Qdrant client with configuration from environment"""
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not self.qdrant_url or not self.qdrant_api_key:
            logger.warning(
                "QDRANT_URL or QDRANT_API_KEY not set in environment. "
                "Using default localhost configuration"
            )
            self.qdrant_url = "http://localhost:6333"
            self.qdrant_api_key = ""
        
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                prefer_grpc=True,
                timeout=30.0
            )
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {str(e)}")
            raise
        
        # Collection configurations
        self.collections = {
            "job_embeddings_BGE": {
                "name": "job_embeddings_BGE",
                "vector_size": BGE_M3_DIMENSION,
                "distance": qdrant_models.Distance.COSINE
            },
            "cv_embeddings_BGE": {
                "name": "cv_embeddings_BGE",
                "vector_size": BGE_M3_DIMENSION,
                "distance": qdrant_models.Distance.COSINE
            }
        }
    
    def ensure_collections_exist(self, vector_size: int = BGE_M3_DIMENSION) -> None:
        """Ensure all required collections exist with proper configuration"""
        try:
            existing_collections = {
                collection.name: collection 
                for collection in self.client.get_collections().collections
            }
            
            for collection_key, collection_config in self.collections.items():
                collection_name = collection_config["name"]
                
                if collection_name in existing_collections:
                    logger.debug(f"Collection {collection_name} already exists")
                    continue
                    
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=collection_config["vector_size"],
                        distance=collection_config["distance"]
                    )
                )
                logger.info(f"Created collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collections exist: {str(e)}")
            raise
    
    def upsert_embeddings(
        self, 
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: List[dict] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Insert or update embeddings in Qdrant with batching and error handling
        
        Args:
            collection_name: Name of the collection to upsert to
            ids: List of document IDs
            embeddings: List of embedding vectors
            payloads: List of metadata payloads (optional)
            batch_size: Number of records to process in each batch
            
        Returns:
            bool: True if operation was successful
        """
        if not payloads:
            payloads = [{} for _ in range(len(ids))]
            
        if len(ids) != len(embeddings) or len(ids) != len(payloads):
            raise ValueError("Length of ids, embeddings, and payloads must match")
            
        try:
            # Process in batches to avoid timeouts
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]
                
                points = [
                    PointStruct(
                        id=str(idx),
                        vector=embedding,
                        payload=payload
                    )
                    for idx, embedding, payload in zip(batch_ids, batch_embeddings, batch_payloads)
                ]
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error upserting embeddings to {collection_name}: {str(e)}")
            return False
    
    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        filter_conditions: Optional[dict] = None
    ) -> List[dict]:
        """
        Search for similar vectors in the collection with filtering
        
        Args:
            collection_name: Name of the collection to search in
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            filter_conditions: Optional filter conditions for the search
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            # Convert filter conditions to Qdrant filter if provided
            qdrant_filter = None
            if filter_conditions:
                must_conditions = []
                for field, value in filter_conditions.items():
                    if isinstance(value, (list, tuple)):
                        must_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                match=qdrant_models.MatchAny(any=value)
                            )
                        )
                    else:
                        must_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                match=qdrant_models.MatchValue(value=value)
                            )
                        )
                qdrant_filter = qdrant_models.Filter(must=must_conditions)
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=min(limit, 100),  # Cap at 100 results
                score_threshold=score_threshold,
                with_vectors=False,
                with_payload=True
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {}
                }
                for hit in search_result
            ]
            
        except Exception as e:
            logger.error(f"Error searching in {collection_name}: {str(e)}")
            return []
            
    def get_collection_info(self, collection_name: str) -> Optional[dict]:
        """Get information about a collection"""
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "name": collection_info.name,
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "config": {
                    "params": collection_info.config.params.dict() if collection_info.config.params else {},
                    "hnsw_config": collection_info.config.hnsw_config.dict() if collection_info.config.hnsw_config else {},
                    "optimizer_config": collection_info.config.optimizer_config.dict() if collection_info.config.optimizer_config else {},
                    "wal_config": collection_info.config.wal_config.dict() if collection_info.config.wal_config else {}
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {str(e)}")
            return None

class GeminiResumeTailor:
    """Handles resume tailoring using Gemini Flash 2.5"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_tailored_resume(
        self,
        original_resume: str,
        job_description: str,
        matching_keywords: List[str],
        missing_keywords: List[str]
    ) -> str:
        """Generate a tailored version of the resume for a specific job"""
        prompt = f"""
        You are an expert resume writer. Please tailor the following resume to better match the job description.
        Focus on emphasizing the matching skills and incorporating missing keywords naturally.
        
        Job Description:
        {job_description}
        
        Matching Keywords (emphasize these):
        {', '.join(matching_keywords)}
        
        Missing Keywords (try to incorporate these naturally):
        {', '.join(missing_keywords)}
        
        Original Resume:
        {original_resume}
        
        Please return ONLY the improved resume text, with no additional commentary or explanations.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating tailored resume: {str(e)}")
            return original_resume

class JobMatcher:
    """Main class for job matching and resume tailoring"""
    
    def __init__(self):
        # Initialize components
        self.embedder = BGE_M3_Embedder()
        self.qdrant = QdrantManager()
        self.resume_tailor = GeminiResumeTailor()
        
        # Ensure Qdrant collections exist
        self.qdrant.ensure_collections_exist(
            vector_size=self.embedder.dimension
        )
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple implementation - can be enhanced with more sophisticated NLP
        words = text.lower().split()
        # Filter out common words and get unique words
        stop_words = {"the", "and", "or", "in", "on", "at", "to", "for", "with"}
        return list(set(word for word in words if word.isalpha() and word not in stop_words))
    
    def calculate_keyword_overlap(
        self, 
        resume_keywords: List[str], 
        job_keywords: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Calculate matching and missing keywords"""
        resume_keywords_set = set(resume_keywords)
        job_keywords_set = set(job_keywords)
        
        matching = list(resume_keywords_set.intersection(job_keywords_set))
        missing = list(job_keywords_set - resume_keywords_set)
        
        return matching, missing
    
    def add_resume(
        self, 
        resume_id: str,
        resume_text: str,
        metadata: dict = None
    ) -> bool:
        """Add a resume to the system"""
        # Generate embedding
        embedding = self.embedder.embed([resume_text])[0]
        
        # Prepare payload
        payload = {
            "text": resume_text,
            "type": "resume",
            "timestamp": datetime.utcnow().isoformat(),
            "keywords": self.extract_keywords(resume_text)
        }
        if metadata:
            payload.update(metadata)
        
        # Store in Qdrant
        return self.qdrant.upsert_embeddings(
            collection_name=self.qdrant.collections["cv_embeddings_BGE"]["name"],
            ids=[resume_id],
            embeddings=[embedding],
            payloads=[payload]
        )
    
    def add_job(
        self,
        job_id: str,
        job_description: str,
        metadata: dict = None
    ) -> bool:
        """Add a job to the system"""
        # Generate embedding
        embedding = self.embedder.embed([job_description])[0]
        
        # Prepare payload
        payload = {
            "text": job_description,
            "type": "job",
            "timestamp": datetime.utcnow().isoformat(),
            "keywords": self.extract_keywords(job_description)
        }
        if metadata:
            payload.update(metadata)
        
        # Store in Qdrant
        return self.qdrant.upsert_embeddings(
            collection_name=self.qdrant.collections["job_embeddings_BGE"]["name"],
            ids=[job_id],
            embeddings=[embedding],
            payloads=[payload]
        )
    
    def find_matching_jobs(
        self,
        resume_id: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[dict]:
        """Find the best matching jobs for a resume"""
        # Get resume embedding
        resume = self.qdrant.client.retrieve(
            collection_name=self.qdrant.collections["cv_embeddings_BGE"],
            ids=[resume_id]
        )
        
        if not resume:
            logger.warning(f"No resume found with ID: {resume_id}")
            return []
            
        resume_embedding = resume[0].vector
        resume_keywords = resume[0].payload.get("keywords", [])
        
        # Search for similar jobs
        matching_jobs = self.qdrant.search_similar(
            collection_name=self.qdrant.collections["job_embeddings_BGE"],
            query_embedding=resume_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Add keyword analysis
        for job in matching_jobs:
            job_keywords = job["payload"].get("keywords", [])
            matching, missing = self.calculate_keyword_overlap(
                resume_keywords, job_keywords
            )
            job["matching_keywords"] = matching
            job["missing_keywords"] = missing
        
        return matching_jobs
    
    def tailor_resume_for_job(
        self,
        resume_id: str,
        job_id: str
    ) -> dict:
        """Generate a tailored version of a resume for a specific job"""
        # Get resume and job data
        resume = self.qdrant.client.retrieve(
            collection_name=self.qdrant.collections["cv_embeddings_BGE"],
            ids=[resume_id]
        )
        
        job = self.qdrant.client.retrieve(
            collection_name=self.qdrant.collections["job_embeddings_BGE"],
            ids=[str(job_id)]
        )
        
        if not resume or not job:
            logger.error("Resume or job not found")
            return None
            
        resume_data = resume[0].payload
        job_data = job[0].payload
        
        # Get keyword analysis
        matching_keywords, missing_keywords = self.calculate_keyword_overlap(
            resume_data.get("keywords", []),
            job_data.get("keywords", [])
        )
        
        # Generate tailored resume
        tailored_resume = self.resume_tailor.generate_tailored_resume(
            original_resume=resume_data["text"],
            job_description=job_data["text"],
            matching_keywords=matching_keywords,
            missing_keywords=missing_keywords
        )
        
        return {
            "original_resume": resume_data["text"],
            "tailored_resume": tailored_resume,
            "job_description": job_data["text"],
            "matching_keywords": matching_keywords,
            "missing_keywords": missing_keywords,
            "original_keywords": resume_data.get("keywords", []),
            "job_keywords": job_data.get("keywords", [])
        }

# Example usage
if __name__ == "__main__":
    # Initialize the matcher
    matcher = JobMatcher()
    
    # Example: Add a resume
    resume_id = "resume_123"
    resume_text = """
    John Doe
    Senior Software Engineer
    
    Skills: Python, Django, Flask, AWS, Docker, Kubernetes, CI/CD
    Experience: 5+ years in backend development
    Education: BS in Computer Science
    """
    
    matcher.add_resume(
        resume_id=resume_id,
        resume_text=resume_text,
        metadata={"name": "John Doe", "email": "john@example.com"}
    )
    
    # Example: Add a job
    job_id = "job_456"
    job_description = """
    Senior Backend Developer
    
    We are looking for an experienced backend developer with:
    - Strong Python skills (Django/Flask)
    - Experience with cloud platforms (AWS/GCP)
    - Knowledge of containerization (Docker, Kubernetes)
    - Experience with CI/CD pipelines
    - Experience with database design and optimization
    
    Nice to have:
    - Experience with microservices architecture
    - Knowledge of machine learning
    """
    
    matcher.add_job(
        job_id=job_id,
        job_description=job_description,
        metadata={"title": "Senior Backend Developer", "company": "TechCorp"}
    )
    
    # Find matching jobs for the resume
    print("Finding matching jobs...")
    matches = matcher.find_matching_jobs(resume_id, top_k=3)
    print(f"Found {len(matches)} matching jobs")
    
    if matches:
        best_match = matches[0]
        print(f"\nBest match: {best_match['payload'].get('title', 'N/A')}")
        print(f"Similarity score: {best_match['score']:.2f}")
        print("Matching keywords:", ", ".join(best_match['matching_keywords']))
        print("Missing keywords:", ", ".join(best_match['missing_keywords']))
        
        # Generate tailored resume for the best match
        print("\nGenerating tailored resume...")
        result = matcher.tailor_resume_for_job(resume_id, best_match['id'])
        
        if result:
            print("\n=== Original Resume Keywords ===")
            print(", ".join(result["original_keywords"][:10]) + "...")
            
            print("\n=== Job Keywords ===")
            print(", ".join(result["job_keywords"][:10]) + "...")
            
            print("\n=== Matching Keywords ===")
            print(", ".join(result["matching_keywords"]))
            
            print("\n=== Missing Keywords ===")
            print(", ".join(result["missing_keywords"]))
            
            print("\n=== Tailored Resume ===")
            print(result["tailored_resume"])