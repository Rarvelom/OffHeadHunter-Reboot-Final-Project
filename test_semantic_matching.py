import os
import sys
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.utils.time_utils import get_current_utc_timestamp, to_iso_format, to_unix_timestamp, is_after
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

from qdrant_client import QdrantClient, models
from qdrant_client.http import models as http_models
from qdrant_client.models import MatchValue
from pymongo import MongoClient
from qdrant_config import get_qdrant_client
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize models
model = SentenceTransformer('BAAI/bge-m3', device='cpu')

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

def find_similar_jobs(cv_text: str, user_id: str, top_k: int = None, include_keywords: bool = True, 
                     is_current_search: bool = True, hours_window: int = 24,
                     cv_uploaded_at: str = None) -> Dict[str, Any]:
    """
    Encuentra trabajos similares a un CV usando coincidencia semántica y análisis de palabras clave,
    considerando solo ofertas recientes.
    
    Args:
        cv_text: Texto del CV para buscar coincidencias
        top_k: Número máximo de coincidencias a devolver (None para no limitar)
        include_keywords: Si se debe incluir análisis de palabras clave
        is_current_search: Si los resultados son de la búsqueda actual o históricos
        hours_window: Ventana de tiempo en horas para considerar ofertas recientes (últimas X horas)
        cv_uploaded_at: Fecha de carga del CV en formato ISO. Si se proporciona, solo se devolverán
                       ofertas con scraped_at posterior a esta fecha.
        
    Returns:
        Dict con:
        - 'matches': Lista de trabajos coincidentes
        - 'source': 'current_search' o 'historical_data'
        - 'warnings': Lista de advertencias
        - 'timestamp': Fecha de la búsqueda
    """
    # Usar datetime y timezone del módulo datetime directamente para evitar conflictos de ámbito
    from datetime import datetime as dt, timezone as tz
    current_time = dt.now(tz.utc)
    
    user = user_id

    result = {
        'matches': [],
        'source': 'current_search' if is_current_search else 'historical_data',
        'warnings': [],
        'timestamp': to_iso_format(current_time)
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
    
    # Calcular la fecha límite para ofertas recientes
    
    # Si se proporcionó una fecha de carga del CV, usarla como límite inferior
    # De lo contrario, usar la ventana de tiempo configurada
    if cv_uploaded_at:
        logger.info(f"[DEBUG] Fecha de carga del CV recibida: {cv_uploaded_at} (tipo: {type(cv_uploaded_at)})")
        
        # Convertir a timestamp Unix usando la función utilitaria
        time_threshold = to_unix_timestamp(cv_uploaded_at)
        
        if time_threshold is None:
            logger.warning(f"[WARNING] Formato de fecha de carga del CV inválido: {cv_uploaded_at}. Usando ventana de tiempo.")
            time_threshold = get_current_utc_timestamp() - (hours_window * 3600)  # horas a segundos
        else:
            # Asegurarnos de que es un entero (segundos desde epoch)
            time_threshold = int(time_threshold)
            
            # Convertir a datetime para logging legible
            from datetime import datetime
            dt_utc = datetime.utcfromtimestamp(time_threshold)
            
            logger.info(f"[DEBUG] Timestamp de carga del CV (Unix): {time_threshold}")
            logger.info(f"[DEBUG] Fecha/hora UTC de carga del CV: {dt_utc.isoformat()}")
            logger.info(f"[INFO] Filtrando ofertas posteriores a la carga del CV (UTC): {dt_utc.isoformat()}")
    else:
        # Usar ventana de tiempo si no se proporcionó fecha de carga
        current_time = get_current_utc_timestamp()
        time_threshold = current_time - (hours_window * 3600)  # horas a segundos
        
        logger.info(f"[INFO] No se proporcionó fecha de carga del CV. Usando ventana de {hours_window}h desde ahora.")
        logger.info(f"[DEBUG] Timestamp actual (Unix): {current_time}")
        logger.info(f"[DEBUG] Timestamp umbral (Unix): {time_threshold} (hace {hours_window} horas)")
        
        # Convertir a datetime para logging legible
        from datetime import datetime
        dt_threshold = datetime.utcfromtimestamp(time_threshold)
        dt_current = datetime.utcfromtimestamp(current_time)
        
        logger.info(f"[DEBUG] Fecha/hora umbral (UTC): {dt_threshold.isoformat()}")
        logger.info(f"[DEBUG] Fecha/hora actual (UTC): {dt_current.isoformat()}")
    
    # Buscar ofertas similares usando coincidencia semántica, filtrando por fecha
    try:
        # Primero obtenemos los IDs de las ofertas recientes
        from qdrant_client.models import Filter, FieldCondition, Range
        
        # time_threshold ya debería ser un timestamp Unix (segundos desde epoch)
        # Verificamos el tipo para asegurarnos
        if hasattr(time_threshold, 'timestamp'):
            timestamp_threshold = int(time_threshold.timestamp())
        else:
            # Si ya es un timestamp numérico, lo convertimos a entero
            timestamp_threshold = int(time_threshold)
        
        # Aplicar ajuste de 1 hora al timestamp de carga del CV
        timestamp_threshold -= 1 * 3600  # Restar 1 hora en segundos
        
        # Obtener la fecha actual para referencia usando el alias local
        from datetime import datetime as dt, timezone as tz
        current_time = dt.now(tz.utc)
        
        # Convertir el timestamp a datetime solo para logging
        threshold_dt = dt.fromtimestamp(timestamp_threshold, tz=tz.utc)
        logger.info(f"[INFO] Aplicando ajuste de 1 hora al timestamp de carga del CV")
        logger.info(f"[DEBUG] Timestamp original: {timestamp_threshold + 3600} (sin ajuste)")
        logger.info(f"[DEBUG] Timestamp con ajuste: {timestamp_threshold} (ajuste de -1 hora)")
        
        logger.info(f"[DEBUG] Filtro de fechas - Hora actual: {current_time.isoformat()}")
        logger.info(f"[DEBUG] Filtro de fechas - Umbral de búsqueda: {threshold_dt.isoformat()}")
        logger.info(f"[DEBUG] Filtro de fechas - Timestamp Unix del umbral: {timestamp_threshold}")
        logger.info(f"[DEBUG] Filtro de fechas - Diferencia con ahora: {(current_time - threshold_dt).total_seconds()/3600:.2f} horas")
        
        # Primero, verificar algunas ofertas para ver sus timestamps
        sample_jobs = client.scroll(
            collection_name="job_embeddings_BGE",
            limit=10,  # Revisar 10 ofertas para ver sus fechas
            with_vectors=False,
            with_payload=["scraped_at", "title", "company"],
            order_by=http_models.OrderBy(
                key="scraped_at",
                direction=http_models.Direction.DESC
            )
        )
        
        if sample_jobs and sample_jobs[0]:
            logger.info("\n[DEBUG] MUESTRA DE OFERTAS CON SUS FECHAS DE CARGA (scraped_at):")
            logger.info("-" * 80)
            for i, job in enumerate(sample_jobs[0]):
                job_scraped_at = job.payload.get('scraped_at')
                job_title = job.payload.get('title', 'Sin título')
                company = job.payload.get('company', 'Sin empresa')
                
                # Formatear la fecha legible
                if job_scraped_at:
                    try:
                        job_dt = datetime.fromtimestamp(job_scraped_at, tz=timezone.utc)
                        formatted_time = job_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                        time_diff = (current_time - job_dt).total_seconds() / 3600  # Diferencia en horas
                        
                        logger.info(f"  - Oferta {i+1}: \"{job_title}\"")
                        logger.info(f"    Empresa: {company}")
                        logger.info(f"    Fecha carga: {formatted_time} (hace {time_diff:.2f} horas)")
                        logger.info(f"    Timestamp Unix: {job_scraped_at}")
                        logger.info(f"    ID de usuario: {user}")
                        logger.info("-" * 60)
                    except Exception as e:
                        logger.warning(f"  - Oferta {i+1}: Error al formatear fecha: {str(e)}")
                else:
                    logger.warning(f"  - Oferta {i+1}: Sin fecha de carga (scraped_at) disponible")
        
        # Crear filtro para ofertas recientes (scraped_at > timestamp_threshold)
        time_filter = Filter(
            must=[
                FieldCondition(
                    key="scraped_at",
                    range=Range(
                        gt=timestamp_threshold  # Usar el timestamp ya convertido
                    )
                ),
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user) # NUEVO! Filtrar por ID de usuario
                )
            ]
        )


        logger.info(f"[DEBUG] Buscando ofertas con scraped_at > {timestamp_threshold} (unix timestamp)")
        
        # Buscar ofertas recientes con el filtro de tiempo
        recent_jobs = client.scroll(
            collection_name="job_embeddings_BGE",
            scroll_filter=time_filter,
            limit=1000,  # Límite razonable de ofertas recientes
            with_vectors=False,
            with_payload=["scraped_at", "title", "company", "user_id"]  # Solo necesitamos estos campos inicialmente
        )
        
        recent_job_ids = [job.id for job in recent_jobs[0]]
        
        if not recent_job_ids:
            if cv_uploaded_at:
                # Convertir el timestamp a datetime para mostrarlo en un formato legible
                from datetime import datetime as dt
                dt_obj = dt.fromtimestamp(time_threshold, tz.utc)
                warning_msg = f"No se encontraron ofertas publicadas después de la carga del CV ({dt_obj.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
                logger.warning(warning_msg)
                
                # Obtener todas las ofertas para analizar el rango de fechas
                from qdrant_client import models
                all_jobs = client.scroll(
                    collection_name="job_embeddings_BGE",
                    limit=100,  # Aumentamos el límite para obtener un buen conjunto de ofertas
                    with_vectors=False,
                    with_payload=["scraped_at", "title", "company", "user_id"],
                    order_by=models.OrderBy(
                        key="scraped_at",
                        direction=models.Direction.DESC
                    )
                )
                
                if all_jobs and all_jobs[0]:
                    # Procesar todas las ofertas obtenidas
                    logger.warning("[DEBUG] Muestra de ofertas disponibles (ordenadas por fecha descendente):")
                    for i, job in enumerate(all_jobs[0][:5]):  # Mostrar las primeras 5 como ejemplo
                        job_scraped_at = job.payload.get('scraped_at')
                        job_title = job.payload.get('title', 'Sin título')
                        company = job.payload.get('company', 'Sin empresa')
                        if job_scraped_at:
                            job_dt = datetime.fromtimestamp(job_scraped_at, timezone.utc)
                            logger.warning(f"  - Oferta {i+1}: '{job_title}' en {company} - {job_dt.isoformat()}")
                        else:
                            logger.warning(f"  - Oferta {i+1}: '{job_title}' en {company} - Sin fecha de scraping")
            else:
                warning_msg = f"No se encontraron ofertas en las últimas {hours_window} horas"
                logger.warning(warning_msg)
                
            result['warnings'].append(warning_msg)
            return result
            
        logger.info(f"Se encontraron {len(recent_job_ids)} ofertas en las últimas {hours_window} horas")
        
        # Ahora buscamos solo en las ofertas recientes
        from qdrant_client import models
        
        # Crear el filtro con la condición has_id en el formato correcto
        search_filters = models.Filter(
            must=[
                models.HasIdCondition(has_id=recent_job_ids)
            ]
        )
        
        search_result = client.search(
            collection_name="job_embeddings_BGE",
            query_vector=cv_embedding,
            limit=len(recent_job_ids) if top_k is None else min(top_k * 2, len(recent_job_ids)),
            with_vectors=False,
            with_payload=True,
            score_threshold=0.4,  # Umbral para considerar coincidencias relevantes
            query_filter=search_filters
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
            
            # Verificar si la oferta es reciente según el umbral de tiempo
            # Usar la función utilitaria para comparar timestamps
            job_scraped_at = job.payload.get('scraped_at')
            
            # Verificar si job_scraped_at es un timestamp válido
            if job_scraped_at is None:
                logger.warning(f"Oferta sin campo 'scraped_at', se excluirá de los resultados")
                continue
                
            # Convertir a entero si es float (por si acaso)
            job_scraped_at = int(job_scraped_at) if job_scraped_at is not None else 0
            
            # Usar la función is_after para comparar correctamente
            if is_after(job_scraped_at, time_threshold):
                # La oferta es reciente, incluirla en los resultados
                pass
                
            # Obtener metadatos del payload o de metadata
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
    if result['matches']:
        result['matches'].sort(key=lambda x: x['combined_score'] if 'combined_score' in x else x['semantic_score'], 
                             reverse=True)
        
        # Limitar al número solicitado de resultados, si se especificó
        if top_k is not None and len(result['matches']) > top_k:
            result['matches'] = result['matches'][:top_k]
            
        # Añadir información sobre el filtrado por fecha
        # Convertir el timestamp Unix a datetime para mostrarlo en un formato legible
        from datetime import datetime as dt
        threshold_dt = dt.fromtimestamp(time_threshold, tz=timezone.utc)
        
        result['filters'] = {
            'time_window_hours': hours_window,
            'time_threshold': threshold_dt.isoformat(),  # Ahora es un datetime válido
            'time_threshold_unix': time_threshold,  # Mantener el timestamp original
            'total_recent_jobs': len(recent_job_ids) if 'recent_job_ids' in locals() else 0,
            'matching_recent_jobs': len(result.get('matches', []))
        }
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

def calculate_scores(cv_text: str, job_text: str) -> Dict[str, float]:
    """
    Calculates semantic, keyword, and combined scores for a single CV-job pair.

    Args:
        cv_text: The full text of the CV.
        job_text: The full text of the job description.

    Returns:
        A dictionary containing 'semantic_score', 'keyword_score', and 'combined_score'.
    """
    if not cv_text or not job_text:
        return {'semantic_score': 0.0, 'keyword_score': 0.0, 'combined_score': 0.0}

    # 1. Calculate Semantic Score (Cosine Similarity)
    try:
        cv_embedding = model.encode(cv_text, convert_to_tensor=True)
        job_embedding = model.encode(job_text, convert_to_tensor=True)
        semantic_score = util.cos_sim(cv_embedding, job_embedding).item()
    except Exception as e:
        logger.error(f"Error calculating semantic score: {e}")
        semantic_score = 0.0

    # 2. Calculate Keyword Score
    try:
        cv_keywords = extract_keywords(cv_text)
        job_keywords = extract_keywords(job_text)
        keyword_score, _ = get_keyword_overlap(cv_keywords, job_keywords)
    except Exception as e:
        logger.error(f"Error calculating keyword score: {e}")
        keyword_score = 0.0

    # 3. Calculate Combined Score (Weighted Average)
    # NOTE: Using a 70/30 weighted average. This can be adjusted.
    semantic_weight = 0.7
    keyword_weight = 0.3
    combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)

    return {
        'semantic_score': semantic_score,
        'keyword_score': keyword_score,
        'combined_score': combined_score
    }

if __name__ == "__main__":
    main()
