import sys
import os
import subprocess
from pathlib import Path
import json
from job_agent_chatbot import AgentChatbot
from job_search_agent import JobSearchAgent
from src.url_gen import generar_url_infojobs
from src.ij_jobs_scraper import scrape_jobs
from src.ij_pdf_exporter import export_ij_offer_to_pdf
from process_documents import process_document_file
from src.text_processing import TextProcessor
from src.qdrant_storage import QdrantStorage
from job_matching import match_cv_to_jobs, get_qdrant_client
from job_resume_tailor import run_resume_tailoring
from qdrant_config import get_qdrant_client
import concurrent.futures

# --- Configuracion de Directorios ---
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
CV_DIR = UPLOADS_DIR / "cvs"
JOBS_DIR = UPLOADS_DIR / "job_descriptions"
TAILORED_CV_DIR = UPLOADS_DIR / "tailored_cvs"

# Asegurarse de que los directorios existan
CV_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
TAILORED_CV_DIR.mkdir(exist_ok=True)

def format_history_for_parser(history):
    # Convierte la lista de turnos a texto tipo diálogo
    return "\n".join([f"{role}: {msg}" for role, msg in history])

def select_jobs_interactively(job_offers: list) -> list:
    """Muestra una lista de ofertas y pide al usuario que seleccione cuáles procesar."""
    print("\n--- Selección de Ofertas de Trabajo ---")
    if not job_offers:
        print("No se encontraron ofertas.")
        return []

    for i, offer in enumerate(job_offers):
        title = offer.get('title', 'Sin título')
        company = offer.get('company', 'Empresa no especificada')
        print(f"[{i + 1:2d}] {title[:70]:<70} | {company}")

    print("\nSeleccione las ofertas que desea procesar.")
    print("Puede seleccionar varias separadas por comas (ej: 1, 3, 5) o un rango (ej: 1-5).")
    print("Presione Enter para seleccionar todas las ofertas.")

    selected_offers = []
    while True:
        try:
            raw_input = input(f"Seleccione (1-{len(job_offers)}): ")
            if not raw_input:
                return job_offers

            selected_indices = set()
            parts = raw_input.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start - 1, end))
                else:
                    selected_indices.add(int(part) - 1)
            
            if all(0 <= i < len(job_offers) for i in selected_indices):
                selected_offers = [job_offers[i] for i in sorted(list(selected_indices))]
                print(f"\n✅ {len(selected_offers)} ofertas seleccionadas.")
                break
            else:
                print("Selección inválida. Asegúrese de que todos los números están en el rango.")
        except (ValueError, IndexError):
            print("Entrada inválida. Por favor, introduzca números o rangos válidos.")
            
    return selected_offers

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
    parser.upload_cv()

    print("\n=== PASO 3: Generando URL ===\n")
    url = generar_url_infojobs(user_profile["job_title"], user_profile["work_modality"], user_profile["salary_expectation"], user_profile["location"])
    print("URL generada:", url)
    
    
    # Paso 3: Scraping de ofertas (guarda en MongoDB y devuelve la lista)
    job_offers = scrape_jobs(url)
    print(f"Ofertas extraídas: {len(job_offers)}")

    # PASO 3.5: Selección interactiva de ofertas
    selected_jobs = select_jobs_interactively(job_offers)

    if not selected_jobs:
        print("No se seleccionaron ofertas. Finalizando el proceso.")
        sys.exit(0)

    # --- PASO 4: Exportar ofertas seleccionadas a PDF ---
    print(f"\n--- Exportando {len(selected_jobs)} ofertas seleccionadas a PDF ---")
    new_job_files = []
    for offer in selected_jobs:
        try:
            export_result = export_ij_offer_to_pdf(offer, JOBS_DIR)
            # Comprobar si la exportación fue exitosa y si el resultado es un diccionario
            if export_result and isinstance(export_result, dict) and export_result.get('success'):
                pdf_filename = export_result.get('pdf_filename')
                if pdf_filename:
                    # Guardamos la metadata que puede ser útil después
                    new_job_files.append({
                        'offer_id': offer.get('id'),
                        'title': offer.get('title'),
                        'pdf_filename': pdf_filename
                    })
                    print(f"  -> Exportado: {pdf_filename}")
            else:
                 print(f"  -> Error exportando oferta {offer.get('id', 'N/A')}: No se recibió un resultado válido.")
        except Exception as e:
            print(f"  -> Error exportando oferta {offer.get('id', 'N/A')}: {e}")

    if not new_job_files:
        print("No se pudo exportar ninguna oferta a PDF. Abortando.")
        sys.exit(1)

    # --- INICIO DE LA INTEGRACIÓN DEL NUEVO FLUJO ---
    print("\n" + "="*60)
    print("PASO 5: PROCESANDO DOCUMENTOS Y GENERANDO EMBEDDINGS")
    print("="*60)

    # El CV ya ha sido procesado y sus embeddings generados por JobSearchAgent.
    cv_path = user_profile.get('cv_path')
    if not (cv_path and Path(cv_path).exists()):
        print("Error: El agente no pudo procesar o encontrar la ruta del CV. Abortando.")
        sys.exit(1)

    cv_id = Path(cv_path).stem
    print(f"CV base para el matching: '{cv_id}'")

    # Inicializar el TextProcessor una sola vez para reutilizar el modelo de embeddings
    print("\nInitializing TextProcessor (this may take a moment)...")
    text_processor = TextProcessor()
    print("TextProcessor initialized.")

    # Procesar SOLO las ofertas recién scrapeadas para generar sus embeddings
    print(f"\n🔄 Procesando {len(new_job_files)} oferta(s) de trabajo nueva(s) en paralelo...")
    newly_processed_job_ids = []
    job_storage = QdrantStorage(collection_name='job_embeddings_BGE2') # Instancia única para todas las ofertas

    # Wrapper para poder usarlo con ThreadPoolExecutor
    def process_job_file_wrapper(job_meta):
        job_filename = job_meta.get('pdf_filename')
        if not job_filename:
            return None
        job_path = JOBS_DIR / Path(job_filename).name
        
        result = process_document_file(
            file_path=job_path,
            collection_name='job_embeddings_BGE2',
            document_type='job',
            text_processor=text_processor,
            storage=job_storage
        )

        if result.get("success"):
            doc_id = Path(result.get('file_name', '')).stem
            if doc_id:
                return doc_id
        else:
            print(f"⚠️  Error processing {job_path.name}: {result.get('error')}")
        return None

    # Usar ThreadPoolExecutor para procesar en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_job = {executor.submit(process_job_file_wrapper, job_meta): job_meta for job_meta in new_job_files}
        for future in concurrent.futures.as_completed(future_to_job):
            doc_id = future.result()
            if doc_id:
                newly_processed_job_ids.append(doc_id)
    
    if not newly_processed_job_ids:
        print("❌ No se pudo procesar ninguna de las nuevas ofertas. Abortando.")
        sys.exit(1)

    print(f"\n✅ Procesamiento completado: {len(newly_processed_job_ids)} ofertas listas para matching")

    print("\n" + "="*60)
    print("PASO 6: REALIZANDO MATCHING DE CV CON OFERTAS RECIÉN SCRAPEADAS")
    print("="*60)
    print(f"🎯 Matching SOLO contra las {len(newly_processed_job_ids)} ofertas recién scrapeadas")
    print("📋 Ofertas objetivo:")
    for i, job_id in enumerate(newly_processed_job_ids, 1):
        print(f"  {i}. {job_id}")

    cv_id = Path(cv_path).stem
    
    # Llamada directa a la función de matching
    print("\n🔍 Ejecutando matching...")
    qdrant_client = get_qdrant_client() # Obtener cliente Qdrant
    matching_results = match_cv_to_jobs(
        client=qdrant_client,
        cv_id=cv_id,
        cv_collection="cv_embeddings_BGE2",
        job_collection="job_embeddings_BGE2",
        specific_job_ids=newly_processed_job_ids
    )

    # Limitar los resultados a top_k si es necesario (la función devuelve todos los resultados ordenados)
    top_k = len(newly_processed_job_ids)
    matching_results = matching_results[:top_k]

    if not matching_results:
        print("No se encontraron ofertas de trabajo coincidentes.")
        sys.exit(0)

    print("\n--- Mejores Ofertas Encontradas ---")
    for i, res in enumerate(matching_results):
        print(f"{i + 1}. Job ID: {res['job_id']}, Score: {res['score']:.4f}")

    print("\n" + "="*60)
    print("PASO 7: SELECCIONE UNA OFERTA PARA ADAPTAR SU CV")
    print("="*60)

    selected_index = -1
    while selected_index < 0 or selected_index >= len(matching_results):
        try:
            choice = input(f"Introduzca el número de la oferta (1-{len(matching_results)}): ")
            selected_index = int(choice) - 1
            if not (0 <= selected_index < len(matching_results)):
                print("Selección inválida. Inténtelo de nuevo.")
        except ValueError:
            print("Por favor, introduzca un número válido.")

    # Extraer el job_id y el score del resultado seleccionado
    selected_match = matching_results[selected_index]
    selected_job_id = selected_match['job_id']
    initial_score = selected_match['score']

    print("\n" + "="*60)
    print(f"PASO 8: ADAPTANDO CV PARA LA OFERTA '{selected_job_id}'")
    print(f"(Puntuación inicial: {initial_score:.4f})")
    print("="*60)

    try:
        tailored_cv_path = run_resume_tailoring(
            cv_id=cv_id,
            job_id=selected_job_id,
            initial_score=initial_score,
            output_dir=str(TAILORED_CV_DIR),
            client=qdrant_client # Reutilizar el cliente Qdrant
        )
    except Exception as e:
        print(f"\n❌ Error durante la adaptación del CV: {e}")
        sys.exit(1)

    # --- PASO 9: Procesar el nuevo CV adaptado --- 
    print(f"\n🔄 Procesando el CV adaptado: {tailored_cv_path.name}")
    
    # Llamada directa a la función de procesamiento
    cv_storage = QdrantStorage(collection_name='cv_embeddings_BGE2')
    result = process_document_file(
        file_path=tailored_cv_path,
        collection_name='cv_embeddings_BGE2',
        document_type='cv',
        text_processor=text_processor,  # Reutilizar la instancia de TextProcessor
        storage=cv_storage 
    )

    if not result.get("success"):
        print(f"❌ Error al procesar el CV adaptado: {result.get('error')}")
        sys.exit(1)

    tailored_cv_id = Path(result.get('file_name', '')).stem
    print(f"✅ CV adaptado procesado. Nuevo ID: '{tailored_cv_id}'")


    # --- PASO 10: Re-matching y comparación --- 
    # Llamada directa a la función de matching para el re-matching
    print("\n🔍 Ejecutando re-matching con el CV adaptado...")
    rematch_data = match_cv_to_jobs(
        client=qdrant_client,
        cv_id=tailored_cv_id,
        cv_collection="cv_embeddings_BGE2",
        job_collection="job_embeddings_BGE2",
        specific_job_ids=[selected_job_id] # Asegurarse de que es una lista
    )

    # Tomar solo el primer resultado, ya que es contra una sola oferta
    rematch_data = rematch_data[0] if rematch_data else None

    if not rematch_data:
        print("Error: No se pudo obtener el nuevo score tras la adaptación.")
    else:
        # 3. Mostrar comparación de scores
        original_score = matching_results[selected_index]['score']
        new_score = rematch_data['score']

        print("\n--- Comparación de Puntuaciones de Matching ---")
        print(f"Oferta: {selected_job_id}")
        print(f"Puntuación del CV Original: {original_score:.4f}")
        print(f"Puntuación del CV Adaptado:  {new_score:.4f}")
        if new_score > original_score:
            print(f"🎉 ¡Mejora de {(new_score - original_score):.4f} puntos!")
        else:
            print("El score no ha mejorado. Puede que la oferta ya fuera un buen match.")

    print("\n¡Proceso completado!")

if __name__ == "__main__":
    main()