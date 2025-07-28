import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from job_agent_chatbot import AgentChatbot  # 1. Chatbot
from job_search_agent import JobSearchAgent  # 2. Obtener criterios del usuario
from src.url_gen import generar_url_infojobs  # 3. Generar URL de búsqueda
from src.ij_jobs_scraper import scrape_jobs  # 4. Scraping de ofertas
from src.ij_pdf_exporter import export_ij_offer_to_pdf  # 5. Exportar PDFs
from test_semantic_matching import (
    extract_keywords,
    get_keyword_overlap,
    get_cv_embedding,
    find_similar_jobs,
    calculate_scores
)
from qdrant_config import get_qdrant_client
from pymongo import MongoClient
from dotenv import load_dotenv
from job_resume_tailor import (
    extract_keywords,
    calculate_keyword_overlap,
    generate_tailored_resume,
    save_text_as_pdf,
    get_all_chunks, # Reutilizamos esta función si es necesario
    get_job_details_by_id
)

# Load environment variables
load_dotenv()

# --- Directorios ---
BASE_DIR = Path(__file__).parent
TAILORED_CV_DIR = BASE_DIR / "uploads" / "tailored_cvs"
TAILORED_CV_DIR.mkdir(parents=True, exist_ok=True)

def format_history_for_parser(history):
    # Convierte la lista de turnos a texto tipo diálogo
    return "\n".join([f"{role}: {msg}" for role, msg in history])


def get_user_cv_text(qdrant_client, user_profile) -> dict:
    """
    Obtiene el texto del CV del usuario desde MongoDB y Qdrant.
    
    Args:
        qdrant_client: Cliente de Qdrant para acceder a los vectores
        
    Returns:
        dict: Diccionario con el texto del CV y el ID de Qdrant, o None si hay un error
    """
    try:
        # Obtener el último CV del usuario desde MongoDB
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            print("Error: MONGODB_URI no está configurado en el archivo .env")
            return None
            
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client["offheadhunter_db"]
        cv_collection = db["cv_uploads"]
        
        # Buscar el CV del actual usuario
        latest_cv = cv_collection.find_one({"user_id": user_profile["user_id"]})
        
        if not latest_cv:
            print("No se encontró ningún CV para el usuario.")
            return None
            
        # Obtener el ID de Qdrant del documento del CV
        # Usamos 'embedding_vector_id_qdrant' que es el campo que se usa al guardar el CV
        qdrant_id = latest_cv.get('embedding_vector_id_qdrant')
        if not qdrant_id:
            print("No se encontró el ID de Qdrant en el documento del CV. Campos disponibles:", latest_cv.keys())
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
            
        # Get the uploaded_at timestamp from the CV document
        uploaded_at = latest_cv.get('uploaded_at')
        
        # If it's a datetime object, convert to ISO format string
        if hasattr(uploaded_at, 'isoformat'):
            uploaded_at = uploaded_at.isoformat()
            
        # Return the CV text, Qdrant ID, and upload timestamp
        return {
            'text': result[0].payload.get('text', ''),
            'qdrant_id': str(qdrant_id),
            'uploaded_at': uploaded_at
        }
        
    except Exception as e:
        print(f"Error al obtener el CV del usuario: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=== PASO 1: Conversación con el chatbot ===")
    chatbot = AgentChatbot()
    history = chatbot.run_conversation()
    history_text = format_history_for_parser(history)

    print("\n=== PASO 2:Extrayendo perfil estructurado... ===\n")
    parser = JobSearchAgent()
    user_profile = parser.parse_profile_from_text(history_text)
    if user_profile:
        print("\nPerfil extraído en formato JSON:")
        print(json.dumps(user_profile, indent=2, ensure_ascii=False))
    else:
        print("No se pudo extraer el perfil correctamente.")
    # parser.save_profile()
    parser.upload_cv()

    print("\n=== PASO 3: Generando URL ===\n")
    url = generar_url_infojobs(user_profile["job_title"], user_profile["work_modality"], user_profile["salary_expectation"], user_profile["location"])
    print("URL generada:", url)

    # Paso 4: Scraping de ofertas (guarda en MongoDB y devuelve la lista)
    print("\n=== PASO 4: Scraping de ofertas ===\n")
    job_offers = scrape_jobs(url)
    max_offers = len(job_offers)
    print(f"Ofertas extraídas: {max_offers}")

    # Paso 5: Exportar PDFs de las ofertas
    print("\n=== PASO 5: Exportando ofertas a PDF ===")

    if max_offers == 0:
        print("No hay ofertas para exportar.")
    else:
        while True:
            try:
                num_to_export = int(input(f"¿Cuántas ofertas quieres exportar a PDF? (1-{max_offers}): "))
                if 1 <= num_to_export <= max_offers:
                    break
                else:
                    print(f"Por favor, introduce un número entre 1 y {max_offers}.")
            except ValueError:
                print("Entrada no válida. Por favor, introduce un número.")

        for offer in job_offers[:num_to_export]:
            if offer.get("url"):
                export_ij_offer_to_pdf(offer, user_profile)
            else:
                print("Oferta sin URL, se omite PDF.")

    # Paso 6: Realizar matching semántico con las ofertas
    print("\n=== PASO 6: Realizando matching semántico con ofertas ===")
    
    try:
        # Obtener el cliente de Qdrant
        qdrant_client = get_qdrant_client()
        
        # Obtener el CV del usuario desde MongoDB y Qdrant
        cv_data = get_user_cv_text(qdrant_client, user_profile)
        
        if not cv_data or 'text' not in cv_data:
            print("No se pudo obtener el CV del usuario. Saliendo del paso de matching semántico.")
            return
            
        cv_text = cv_data['text']
        qdrant_id = cv_data.get('qdrant_id')
        
        if qdrant_id:
            print(f"🔑 ID de Qdrant del CV: {qdrant_id}")
        else:
            print("⚠️ No se pudo obtener el ID de Qdrant del CV")
        
        # Buscar ofertas similares al CV (solo las publicadas después de que se subió el CV)
        print("\nBuscando ofertas similares a tu perfil...")
        result = find_similar_jobs(
            cv_text,
            user_profile['user_id'], 
            top_k=5,
            cv_uploaded_at=cv_data.get('uploaded_at')
        )
        
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
            
            # --- PASO 7: ADAPTAR CV PARA UNA OFERTA (NUEVO) ---
            print("\n=== PASO 7: Adaptar CV para una oferta ===")
            
            selected_index = -1
            while True:
                try:
                    choice = input(f"Introduce el número de la oferta para la que quieres adaptar tu CV (1-{len(result['matches'])}), o 0 para omitir: ")
                    selected_index = int(choice)
                    
                    if selected_index == 0:
                        print("\nProceso de adaptación de CV omitido.")
                        break

                    if 1 <= selected_index <= len(result['matches']):
                        # Obtener la oferta y los IDs necesarios
                        selected_job = result['matches'][selected_index - 1]
                        cv_id = cv_data.get('qdrant_id')
                        job_id = selected_job.get('job_id') # CORRECCIÓN: Usar 'job_id' en lugar de 'id'

                        if not job_id:
                            print("Error: La oferta seleccionada no tiene un ID válido.")
                            break

                        print(f"\nAdaptando CV '{cv_id}' para la oferta '{job_id}'...")

                        # 1. Obtener textos completos de Qdrant
                        cv_full_text = cv_data.get('text')
                        # Asegúrate de que la colección de jobs es la correcta
                        job_chunks = get_all_chunks(qdrant_client, "job_embeddings_BGE", job_id)
                        job_full_text = '\n'.join([c.payload['text'] for c in job_chunks])

                        # 2. Analizar palabras clave
                        cv_keywords = extract_keywords(cv_full_text)
                        job_keywords = extract_keywords(job_full_text)
                        matching_keys, missing_keys = calculate_keyword_overlap(cv_keywords, job_keywords)

                        # 3. Generar CV adaptado con IA
                        print("\n🤖 Llamando a la IA para generar el nuevo CV... (esto puede tardar un momento)")
                        tailored_resume_text = generate_tailored_resume(
                            cv_full_text, 
                            job_full_text, 
                            matching_keys, 
                            missing_keys
                        )

                        # 4. Guardar el resultado como PDF
                        output_filename = f"CV_{cv_id}_adaptado_para_{job_id}.pdf"
                        output_path = TAILORED_CV_DIR / output_filename
                        save_text_as_pdf(tailored_resume_text, output_path)
                        
                        print("\n--- Proceso de Adaptación Finalizado ---")
                        print(f"\n✅ ¡Éxito! El CV adaptado ha sido guardado en: {output_path}")
                        break # Salir del bucle while
                    else:
                        print(f"Selección inválida. Por favor, introduce un número entre 1 y {len(result['matches'])}.")
                except ValueError:
                    print("Entrada no válida. Por favor, introduce un número.")
                except Exception as e:
                    print(f"Ocurrió un error durante la adaptación: {e}")
                    break

            # --- PASO 8: RECALCULAR Y COMPARAR MATCHING SCORE ---
            print("\n--- PASO 8: COMPARANDO MEJORA DEL MATCHING ---")
            job_full_text = get_job_details_by_id(job_id)

            if job_full_text:
                # Calcular la nueva puntuación con el CV adaptado
                new_scores = calculate_scores(tailored_resume_text, job_full_text)
                new_combined_score = new_scores.get('combined_score', 0)

                # Obtener la puntuación original de la oferta seleccionada
                original_score = selected_job.get('combined_score', 0)

                # Calcular la mejora
                improvement = new_combined_score - original_score
                improvement_percent = (improvement / original_score * 100) if original_score > 0 else 0

                # Mostrar la comparación
                print("\n" + "="*25 + " ANÁLISIS DE MEJORA " + "="*25)
                print(f"Puntuación Original:          {original_score*100:.1f}%")
                print(f"Puntuación con CV Adaptado:   {new_combined_score*100:.1f}%")
                print("-" * 68)
                if improvement > 0:
                    print(f"¡Mejora de {improvement*100:.1f} puntos porcentuales! (+{improvement_percent:.1f}%)")
                elif improvement < 0:
                    print(f"La puntuación ha disminuido en {abs(improvement)*100:.1f} puntos.")
                else:
                    print("La puntuación no ha cambiado.")
                print("=" * 68 + "\n")
            else:
                print("No se pudo obtener el texto completo de la oferta para recalcular la puntuación.")

        # else:
        #     print("No se pudo generar el CV adaptado, por lo que no se puede comparar la puntuación.")

    except Exception as e:
        print(f"\nError al realizar el matching semántico: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()