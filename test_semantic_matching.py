import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import yake
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Add the src directory to the Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / 'src'
sys.path.append(str(project_root))  # Add project root to path
sys.path.append(str(src_path))      # Add src directory to path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from pymongo import MongoClient
from qdrant_config import get_qdrant_client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize models
model = SentenceTransformer('BAAI/bge-m3')

# Initialize YAKE! keyword extractor
yake_extractor = yake.KeywordExtractor(
    lan="es",  # Language (supports both English and Spanish)
    n=3,        # Max n-gram size
    dedupLim=0.9,
    top=20,     # Number of keywords to extract
    features=None
)

def preprocess_text(text: str) -> str:
    """Preprocess text by removing special characters and extra whitespace."""
    if not text:
        return ""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    text = ' '.join(text.split())
    return text

def extract_keywords(text: str, extractor_type: str = 'yake') -> List[Dict[str, float]]:
    """
    Extract keywords from text using the specified extractor.
    
    Args:
        text: Input text to extract keywords from
        extractor_type: Type of extractor to use ('yake' or 'tfidf')
        
    Returns:
        List of dictionaries with 'keyword' and 'score' keys
    """
    if not text:
        return []
        
    text = preprocess_text(text)
    
    if extractor_type.lower() == 'yake':
        # Use YAKE! for keyword extraction
        keywords = yake_extractor.extract_keywords(text)
        return [{'keyword': kw[0], 'score': float(kw[1])} for kw in keywords]
    
    elif extractor_type.lower() == 'tfidf':
        # Fallback to TF-IDF based extraction
        words = word_tokenize(text)
        stop_words = set(stopwords.words('spanish') + stopwords.words('english'))
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        
        # Simple word frequency (can be enhanced with scikit-learn's TfidfVectorizer)
        word_freq = Counter(words)
        total_words = len(words)
        
        return [
            {'keyword': word, 'score': count/total_words}
            for word, count in word_freq.most_common(20)
        ]
    
    return []

def get_keyword_overlap(cv_keywords: List[Dict[str, float]], 
                      job_keywords: List[Dict[str, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate the overlap between CV and job keywords.
    
    Args:
        cv_keywords: List of CV keywords with scores
        job_keywords: List of job keywords with scores
        
    Returns:
        Tuple of (match_score, matched_keywords)
    """
    # Convert to sets for easier comparison
    cv_keyword_set = {kw['keyword'] for kw in cv_keywords}
    job_keyword_set = {kw['keyword'] for kw in job_keywords}
    
    # Find common keywords
    common_keywords = cv_keyword_set.intersection(job_keyword_set)
    
    # Calculate match score (Jaccard similarity)
    if not cv_keyword_set and not job_keyword_set:
        return 0.0, []
        
    match_score = len(common_keywords) / len(cv_keyword_set.union(job_keyword_set))
    
    # Get details of matched keywords with their scores
    matched_keywords = []
    for kw in cv_keywords + job_keywords:
        if kw['keyword'] in common_keywords and kw['keyword'] not in [m['keyword'] for m in matched_keywords]:
            matched_keywords.append({
                'keyword': kw['keyword'],
                'cv_score': next((k['score'] for k in cv_keywords if k['keyword'] == kw['keyword']), 0),
                'job_score': next((k['score'] for k in job_keywords if k['keyword'] == kw['keyword']), 0)
            })
    
    return match_score, matched_keywords

def get_cv_embedding(text: str) -> List[float]:
    """Get embedding for CV text using BGE-m3 model."""
    try:
        # Generate embedding using BGE-m3
        embedding = model.encode([text])[0].tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error getting CV embedding: {str(e)}")
        return None

def find_similar_jobs(cv_text: str, top_k: int = 5, include_keywords: bool = True, is_current_search: bool = True) -> Dict[str, Any]:
    """
    Encuentra trabajos similares a un CV usando coincidencia semántica y análisis de palabras clave.
    
    Args:
        cv_text: Texto del CV para buscar coincidencias
        top_k: Número máximo de coincidencias a devolver
        include_keywords: Si se debe incluir análisis de palabras clave
        is_current_search: Si los resultados son de la búsqueda actual o históricos
        
    Returns:
        Dict con:
        - 'matches': Lista de trabajos coincidentes
        - 'source': 'current_search' o 'historical_data'
        - 'warnings': Lista de advertencias
        - 'timestamp': Fecha de la búsqueda
    """
    result = {
        'matches': [],
        'source': 'current_search' if is_current_search else 'historical_data',
        'warnings': [],
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if not cv_text:
        error_msg = "No se proporcionó texto de CV para la búsqueda"
        logger.error(error_msg)
        result['warnings'].append(error_msg)
        return result
    
    # Obtener embedding del CV
    try:
        cv_embedding = get_cv_embedding(cv_text)
        if not cv_embedding:
            error_msg = "No se pudo generar el embedding del CV"
            logger.error(error_msg)
            result['warnings'].append(error_msg)
            return result
    except Exception as e:
        error_msg = f"Error al generar embedding del CV: {str(e)}"
        logger.error(error_msg)
        result['warnings'].append(error_msg)
        return result
    
    # Inicializar cliente Qdrant
    try:
        client = get_qdrant_client()
    except Exception as e:
        error_msg = f"Error al conectar con Qdrant: {str(e)}"
        logger.error(error_msg)
        result['warnings'].append(error_msg)
        return result
    
    # Verificar si hay ofertas en la colección
    try:
        collection_info = client.get_collection("job_embeddings_BGE")
        if collection_info.points_count == 0:
            warning_msg = "No hay ofertas de trabajo disponibles en la base de datos"
            logger.warning(warning_msg)
            result['warnings'].append(warning_msg)
            return result
    except Exception as e:
        error_msg = f"Error al verificar ofertas en Qdrant: {str(e)}"
        logger.error(error_msg)
        result['warnings'].append(error_msg)
        return result
    
    # Buscar ofertas similares usando coincidencia semántica
    try:
        search_result = client.search(
            collection_name="job_embeddings_BGE",
            query_vector=cv_embedding,
            limit=top_k * 2,  # Obtener más resultados para filtrar por palabras clave
            with_vectors=False,
            with_payload=True,
            score_threshold=0.4  # Umbral para considerar coincidencias relevantes
        )
        
        if not search_result:
            warning_msg = "No se encontraron ofertas que coincidan con el perfil"
            logger.warning(warning_msg)
            result['warnings'].append(warning_msg)
            return result
            
    except Exception as e:
        error_msg = f"Error al buscar ofertas similares: {str(e)}"
        logger.error(error_msg)
        result['warnings'].append(error_msg)
        return result
    
    # Extraer palabras clave del CV si está habilitado
    cv_keywords = []
    if include_keywords and cv_text:
        try:
            cv_keywords = extract_keywords(cv_text)
            logger.info(f"Se extrajeron {len(cv_keywords)} palabras clave del CV")
        except Exception as e:
            warning_msg = f"No se pudieron extraer palabras clave del CV: {str(e)}"
            logger.warning(warning_msg)
            result['warnings'].append(warning_msg)
    
    # Procesar resultados
    processed_matches = 0
    for hit in search_result:
        try:
            job_id = hit.id
            job_data = client.retrieve(
                collection_name="job_embeddings_BGE",
                ids=[job_id],
                with_vectors=False
            )
            
            if not job_data or not job_data[0].payload:
                logger.warning(f"Datos de trabajo incompletos para ID {job_id}")
                continue
                
            job = job_data[0]
            payload = job.payload
            
            # Validar campos obligatorios
            job_title = payload.get('title', '')
            if not job_title:
                logger.warning(f"ID {job_id}: Falta el título de la oferta")
                job_title = "Título no disponible"
                
            job_company = payload.get('company', 'Empresa no especificada')
            job_description = payload.get('description', '')
            
            # Asegurarse de que los campos de metadatos existan
            metadata = payload.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Preparar entrada de resultado con todos los campos necesarios
            job_result = {
                'job_id': str(job_id),
                'semantic_score': float(hit.score),
                'title': job_title,
                'company': job_company,
                'description': job_description[:200] + '...' if job_description else 'Descripción no disponible',
                'metadata': {
                    'source_url': metadata.get('source_url', payload.get('source_url', '')),
                    'has_pdf': metadata.get('has_pdf', payload.get('has_pdf', False)),
                    'scraped_at': metadata.get('scraped_at', payload.get('scraped_at', '')),
                    'locations': metadata.get('locations', payload.get('locations', '')),
                    'salary': metadata.get('salary', payload.get('salary', 'No especificado'))
                },
                'keyword_matches': [],
                'keyword_score': 0.0,
                'combined_score': float(hit.score)  # Inicializar con la puntuación semántica
            }
            
            # Análisis de palabras clave si está habilitado y hay palabras clave del CV
            if include_keywords and cv_keywords and job_description:
                try:
                    # Extraer palabras clave de la descripción del trabajo
                    job_keywords = extract_keywords(job_description)
                    
                    # Calcular coincidencia de palabras clave
                    if job_keywords:
                        # Obtener las 10 palabras clave más relevantes del trabajo
                        top_job_keywords = [kw['keyword'] for kw in job_keywords[:10]]
                        
                        # Encontrar coincidencias con las palabras clave del CV
                        cv_keyword_set = {kw['keyword'].lower() for kw in cv_keywords}
                        matched_keywords = []
                        
                        for kw in job_keywords:
                            kw_lower = kw['keyword'].lower()
                            if kw_lower in cv_keyword_set:
                                matched_keywords.append({
                                    'keyword': kw['keyword'],
                                    'score': kw['score']
                                })
                        
                        # Calcular puntuación de palabras clave (0-1)
                        if matched_keywords:
                            # Ponderar por relevancia de las palabras clave
                            total_score = sum(kw['score'] for kw in matched_keywords)
                            max_possible_score = sum(kw['score'] for kw in job_keywords[:10])
                            keyword_score = min(total_score / max_possible_score, 1.0)
                            
                            # Actualizar resultado con análisis de palabras clave
                            job_result.update({
                                'keyword_matches': [{
                                    'keyword': m['keyword'],
                                    'relevance': m['score']
                                } for m in matched_keywords[:5]],  # Mostrar solo las 5 principales coincidencias
                                'keyword_score': keyword_score
                            })
                            
                            # Calcular puntuación combinada (70% semántica, 30% palabras clave)
                            semantic_weight = 0.7
                            keyword_weight = 0.3
                            job_result['combined_score'] = (
                                (job_result['semantic_score'] * semantic_weight) +
                                (keyword_score * keyword_weight)
                            )
                            
                except Exception as e:
                    logger.warning(f"Error en análisis de palabras clave para trabajo {job_id}: {str(e)}")
            
            # Agregar el trabajo a los resultados
            result['matches'].append(job_result)
            processed_matches += 1
            
        except Exception as e:
            logger.error(f"Error procesando trabajo {job_id if 'job_id' in locals() else 'desconocido'}: {str(e)}")
            continue
    
    # Ordenar resultados por puntuación combinada (o semántica si no hay análisis de palabras clave)
    sort_key = 'combined_score' if include_keywords and any('combined_score' in m for m in result['matches']) else 'semantic_score'
    result['matches'] = sorted(
        result['matches'],
        key=lambda x: x.get(sort_key, 0),
        reverse=True
    )[:top_k]  # Limitar al número solicitado de resultados
    
    # Agregar metadatos adicionales
    result['total_matches'] = len(result['matches'])
    result['search_metrics'] = {
        'cv_keywords_count': len(cv_keywords) if include_keywords else 0,
        'jobs_processed': processed_matches,
        'sort_key': sort_key,
        'is_current_search': is_current_search
    }
    
    return result
        
    # No se necesita este código ya que la funcionalidad se ha movido arriba
    # en la nueva implementación
    pass

def main():
    try:
        # Get Qdrant client
        client = get_qdrant_client()
        
        # Get MongoDB client and collections
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI not found in .env")
        
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client["offheadhunter_db"]
        cv_collection = db["cv_uploads"]
        job_collection = db["job_offers"]

        # Get the most recent CV from MongoDB
        print("\nFetching the most recent CV from MongoDB...")
        cv_from_mongo = cv_collection.find_one(
            {},
            sort=[('uploaded_at', -1)]
        )
        
        if not cv_from_mongo:
            raise ValueError("No CVs found in MongoDB")
            
        # Get the CV from Qdrant using the stored ID
        cv_id = cv_from_mongo.get('embedding_vector_id_qdrant')
        if not cv_id:
            raise ValueError("CV in MongoDB doesn't have Qdrant ID")
            
        print(f"\nGetting CV from Qdrant with ID: {cv_id}")
        cv_result = client.retrieve(
            collection_name="cv_embeddings_BGE",
            ids=[cv_id],
            with_payload=True,
            with_vectors=True
        )
        
        if not cv_result:
            raise ValueError(f"CV not found in Qdrant with ID: {cv_id}")
            
        cv = cv_result[0]
        print(f"CV Text Preview: {cv.payload.get('text', '')[:200]}...")

        # Get the most recent job offers from MongoDB
        print("\nFetching the most recent job offers from MongoDB...")
        recent_jobs = list(job_collection.find(
            {},
            sort=[('posted_at', -1)]
        ).limit(5))
        
        if not recent_jobs:
            raise ValueError("No job offers found in MongoDB")
            
        print(f"\nFound {len(recent_jobs)} recent job offers")

        # Get embeddings from Qdrant for the recent jobs
        print("\nGetting job embeddings from Qdrant...")
        offer_embeddings = []
        offer_info = []
        
        for job in recent_jobs:
            offer_id = job.get('embedding_vector_id_qdrant')
            if not offer_id:
                logging.warning(f"Job offer {job.get('_id')} doesn't have Qdrant ID")
                continue
                
            offer_result = client.retrieve(
                collection_name="job_embeddings_BGE",
                ids=[offer_id],
                with_payload=True,
                with_vectors=True
            )
            
            if offer_result:
                offer = offer_result[0]
                offer_embeddings.append(offer.vector)
                offer_info.append({
                    'id': job['_id'],
                    'title': job.get('title', ''),
                    'company': job.get('company', '')
                })

        if not offer_embeddings:
            raise ValueError("No job embeddings found in Qdrant")

        # Get embeddings from Qdrant
        print("\nGetting embeddings from Qdrant...")
        cv_embedding = cv.vector

        # Calculate match scores
        print("\nCalculating match scores...")
        
        # Calculate cosine similarity between CV and each offer
        scores = []
        for i, embedding in enumerate(offer_embeddings):
            score = model.cosine_similarity(cv_embedding, embedding)
            scores.append({
                'offer_id': offer_info[i]['id'],
                'score': float(score),
                'title': offer_info[i]['title'],
                'company': offer_info[i]['company']
            })

        # Sort offers by match score
        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)

        # Print results
        print("\nTop matching job offers:")
        for i, result in enumerate(sorted_scores[:5]):
            print(f"\n{i+1}. Match Score: {result['score']:.4f}")
            print(f"Title: {result['title']}")
            print(f"Company: {result['company']}")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
