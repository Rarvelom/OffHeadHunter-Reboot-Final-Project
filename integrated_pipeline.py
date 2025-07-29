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

    # Paso 4: Exportar PDFs de las ofertas y capturar nombres de archivos generados
    new_job_files = []  # Lista para rastrear las ofertas recién scrapeadas
    print(f"\nExportando {len(job_offers)} ofertas a PDF...")
    
    for i, offer in enumerate(job_offers, 1):
        if offer.get("url"):
            print(f"  [{i}/{len(job_offers)}] Exportando oferta: {offer.get('title', 'Sin título')[:50]}...")
            # Capturar el nombre del archivo PDF generado
            pdf_result = export_ij_offer_to_pdf(offer, output_dir=str(JOBS_DIR))
            if pdf_result and pdf_result.get('success'):
                new_job_files.append(pdf_result) 
                pdf_name = Path(pdf_result['pdf_filename']).name
                print(f"    ✅ PDF generado: {pdf_name}")
        else:
            print(f"  [{i}/{len(job_offers)}] ⚠️  Oferta sin URL, se omite PDF.")
    
    print(f"\n📊 Resumen: {len(new_job_files)} ofertas exportadas exitosamente")
    if not new_job_files:
        print("❌ No se generaron PDFs de ofertas. Abortando proceso.")
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

    # Procesar SOLO las ofertas recién scrapeadas para generar sus embeddings
    print(f"\n🔄 Procesando {len(new_job_files)} oferta(s) de trabajo nueva(s)...")
    newly_processed_job_ids = []
    for job_meta in new_job_files:
        job_filename = job_meta.get('pdf_filename')
        if not job_filename:
            continue
        job_path = JOBS_DIR / Path(job_filename).name
        
        result = subprocess.run([
            sys.executable, "process_documents.py", str(job_path),
            '--document-type', 'job', '--collection', 'job_embeddings_BGE2'
        ], check=True, capture_output=True, text=True)
        
        doc_id = None
        for line in result.stdout.strip().split('\n'):
            if line.startswith("PROCESSED_DOC_ID:"):
                doc_id = line.split(":", 1)[1].strip()
                break
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
    # Construir comando con los job IDs específicos
    matching_cmd = [
        sys.executable, "job_matching.py",
        "--cv_id", cv_id,
        "--top_k", str(len(newly_processed_job_ids)),  # Mostrar todas las ofertas nuevas
        "--job_ids"
    ] + newly_processed_job_ids  # Añadir los job IDs específicos
    
    print(f"\n🔍 Ejecutando matching con comando: {' '.join(matching_cmd[-6:])}...")  # Mostrar últimos argumentos
    result = subprocess.run(matching_cmd, capture_output=True, text=True, check=True)

    matching_results = json.loads(result.stdout.strip())

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

    tailor_result = subprocess.run([
        sys.executable, "job_resume_tailor.py",
        "--cv_id", cv_id,
        "--job_id", selected_job_id,
        "--output_dir", str(TAILORED_CV_DIR),
        "--initial_score", str(initial_score) # Pasar el score inicial
    ], check=True, capture_output=True, text=True)

    # Extraer la ruta del CV adaptado de la salida del script
    tailored_cv_path_str = ""
    for line in tailor_result.stdout.strip().split('\n'):
        if line.startswith("TAILORED_CV_PATH:"):
            tailored_cv_path_str = line.split(":", 1)[1].strip()
            break
    
    if not tailored_cv_path_str:
        print("Error: No se pudo encontrar la ruta del CV adaptado en la salida del script.")
        sys.exit(1)

    tailored_cv_path = Path(tailored_cv_path_str)
    print(f"\nEl CV adaptado ha sido generado en: {tailored_cv_path}")

    # --- PASO 9: RE-EVALUAR EL MATCHING CON EL CV ADAPTADO ---
    print("\n" + "="*60)
    print("PASO 9: RE-EVALUAR EL MATCHING CON EL CV ADAPTADO")
    print("="*60)

    # 1. Procesar el CV adaptado para generar sus embeddings
    print(f"\nProcesando el CV adaptado en: {tailored_cv_path}...")
    subprocess.run([
        sys.executable, "process_documents.py",
        str(tailored_cv_path),
        '--document-type', 'cv',
        '--collection', 'cv_embeddings_BGE2'  # Usar la misma colección
    ], check=True)

    # 2. Re-ejecutar el matching con el ID del nuevo CV
    tailored_cv_id = tailored_cv_path.stem
    print(f"\nRe-ejecutando matching para el CV adaptado ('{tailored_cv_id}') y la oferta ('{selected_job_id}')...")
    rematch_cmd = [
        sys.executable, "job_matching.py",
        "--cv_id", tailored_cv_id,
        "--job_ids", selected_job_id  # Solo contra la oferta seleccionada
    ]
    rematch_result = subprocess.run(rematch_cmd, capture_output=True, text=True, check=True)
    rematch_data = json.loads(rematch_result.stdout.strip())

    # 3. Mostrar comparación de scores
    original_score = matching_results[selected_index]['score']
    new_score = rematch_data[0]['score'] if rematch_data else 0

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