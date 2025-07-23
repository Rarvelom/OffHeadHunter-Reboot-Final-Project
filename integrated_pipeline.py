import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from job_search_agent import JobSearchAgent  # 1. Obtener criterios del usuario
from src.url_gen import generar_url_infojobs  # 2. Generar URL de búsqueda
from src.ij_jobs_scraper import scrape_jobs  # 3. Scraping de ofertas
from src.ij_pdf_exporter import export_ij_offer_to_pdf  # 4. Exportar PDFs
from test_semantic_matching import (
    extract_keywords,
    get_keyword_overlap,
    get_cv_embedding,
    find_similar_jobs
)
from qdrant_config import get_qdrant_client
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_user_cv_text(qdrant_client) -> dict:
    """
    Obtiene el texto del CV del usuario desde MongoDB y Qdrant.
    
    Args:
        qdrant_client: Cliente de Qdrant para acceder a los vectores
        
    Returns:
        dict: Diccionario con el texto del CV y el ID de Qdrant, o None si hay un error
    """
    try:
        # Conectar a MongoDB
        mongo_client = MongoClient(os.getenv('MONGODB_URI'))
        db = mongo_client.get_database()
        cv_collection = db['cv_uploads']
        
        # Obtener el CV más reciente del usuario
        latest_cv = cv_collection.find_one(
            {},  # Sin filtro para obtener el más reciente
            sort=[('uploaded_at', -1)]  # Ordenar por fecha descendente
        )
        
        if not latest_cv:
            print("No se encontró ningún CV en la base de datos.")
            return None
            
        # Obtener el ID de Qdrant
        qdrant_id = latest_cv.get('embedding_vector_id_qdrant')
        if not qdrant_id:
            print("El CV no tiene un ID de Qdrant asociado.")
            return None
            
        print(f"\n✅ CV encontrado en la base de datos con ID de Qdrant: {qdrant_id}")
            
        # Obtener el texto del CV desde Qdrant
        result = qdrant_client.retrieve(
            collection_name="cv_embeddings_BGE",
            ids=[qdrant_id],
            with_payload=True,
            with_vectors=False
        )
        
        if not result or len(result) == 0:
            print(f"No se encontró el CV con ID {qdrant_id} en Qdrant.")
            return None
            
        # Devolver el texto del CV y el ID de Qdrant
        return {
            'text': result[0].payload.get('text', ''),
            'qdrant_id': str(qdrant_id)
        }
        
    except Exception as e:
        print(f"Error al obtener el CV del usuario: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Paso 1: Obtener criterios del usuario
    print("\n=== PASO 1: Obteniendo criterios de búsqueda ===")
    agent = JobSearchAgent()
    agent.run(reset_profile=True)
    user_profile = agent.user_profile  # Diccionario con los datos

    # Extraer los datos del perfil del usuario
    job_title = user_profile.get('job_title')
    work_modality = user_profile.get('work_modality')
    salary_expectation = user_profile.get('salary_expectation')
    location = user_profile.get('location')

    if not job_title:
        print("No se ha proporcionado un título de trabajo. Abortando.")
        sys.exit(1)

    # Paso 2: Generar URL de búsqueda con todos los parámetros
    url = generar_url_infojobs(
        puesto=job_title,
        modalidad=work_modality,
        salario_minimo=salary_expectation,
        localidades=location
    )
    print(f"URL generada para scraping: {url}")

    # Paso 3: Scraping de ofertas (guarda en MongoDB y devuelve la lista)
    job_offers = scrape_jobs(url)
    print(f"Ofertas extraídas: {len(job_offers)}")

    # Paso 4: Exportar PDFs de las ofertas
    print("\n=== PASO 4: Exportando ofertas a PDF ===")
    for offer in job_offers:
        if offer.get("url"):
            export_ij_offer_to_pdf(offer)
        else:
            print("Oferta sin URL, se omite PDF.")

    # Paso 5: Realizar matching semántico con las ofertas
    print("\n=== PASO 5: Realizando matching semántico con ofertas ===")
    
    try:
        # Obtener el cliente de Qdrant
        qdrant_client = get_qdrant_client()
        
        # Obtener el CV del usuario desde MongoDB y Qdrant
        cv_data = get_user_cv_text(qdrant_client)
        
        if not cv_data or 'text' not in cv_data:
            print("No se pudo obtener el CV del usuario. Saliendo del paso de matching semántico.")
            return
            
        cv_text = cv_data['text']
        qdrant_id = cv_data.get('qdrant_id')
        
        if qdrant_id:
            print(f"🔑 ID de Qdrant del CV: {qdrant_id}")
        else:
            print("⚠️ No se pudo obtener el ID de Qdrant del CV")
        
        # Buscar ofertas similares al CV
        print("\nBuscando ofertas similares a tu perfil...")
        result = find_similar_jobs(cv_text, top_k=5)
        
        # Mostrar advertencias si las hay
        if result.get('warnings'):
            for warning in result['warnings']:
                print(f"Advertencia: {warning}")
        
        # Mostrar resultados
        if not result.get('matches'):
            print("No se encontraron ofertas similares a tu perfil.")
        else:
            print(f"\n=== OFERTAS RECOMENDADAS ({len(result['matches'])}) ===")
            print(f"Origen de los datos: {'Búsqueda actual' if result.get('source') == 'current_search' else 'Datos históricos'}")
            print("-" * 80)
            
            for i, job in enumerate(result['matches'], 1):
                print(f"\n{i}. {job.get('title', 'Título no disponible').upper()}")
                print(f"   {'-' * (len(str(i)) + 2 + len(job.get('title', 'Título no disponible')) + 2)}")
                print(f"   Empresa: {job.get('company', 'No especificada')}")
                
                # Mostrar información de ubicación y salario si está disponible
                metadata = job.get('metadata', {})
                if metadata.get('locations'):
                    print(f"   Ubicación: {metadata.get('locations')}")
                if metadata.get('salary'):
                    print(f"   Salario: {metadata.get('salary')}")
                
                # Mostrar puntuación de coincidencia (combinada o solo semántica)
                score = job.get('combined_score', job.get('semantic_score', 0))
                print(f"\n   Puntuación de coincidencia: {score*100:.1f}%")
                
                # Mostrar palabras clave coincidentes si están disponibles
                if 'keyword_matches' in job and job['keyword_matches']:
                    keywords = [match['keyword'] for match in job['keyword_matches'][:5]]
                    print(f"   Palabras clave coincidentes: {', '.join(keywords)}")
                
                # Mostrar descripción si está disponible
                if job.get('description'):
                    print(f"\n   Resumen: {job.get('description')}")
                
                # Mostrar información adicional si está disponible
                if metadata.get('source_url'):
                    print(f"\n   📌 Más información: {metadata.get('source_url')}")
                if metadata.get('scraped_at'):
                    print(f"   📅 Publicada el: {metadata.get('scraped_at')}")
                
                print("\n" + "-" * 80)
    
    except Exception as e:
        print(f"\nError al realizar el matching semántico: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()