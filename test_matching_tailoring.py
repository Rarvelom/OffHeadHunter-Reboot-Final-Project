import sys
import os
from pathlib import Path
import json
import io
import re
from pypdf import PdfReader

# --- Configuracion de Directorios y Colecciones ---
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
CV_DIR = UPLOADS_DIR / "cvs"
JOBS_DIR = UPLOADS_DIR / "job_descriptions"
TAILORED_CV_DIR = UPLOADS_DIR / "tailored_cvs"

# Ruta al intérprete de Python del entorno virtual
PYTHON_EXEC = str(BASE_DIR / "offheadhunter_env" / "bin" / "python")

# Importar configuración centralizada
from qdrant_config import CV_COLLECTION, JOB_COLLECTION

# Asegurarse de que los directorios existan
CV_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
TAILORED_CV_DIR.mkdir(exist_ok=True)

# --- Importaciones de scripts ---
# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(str(BASE_DIR))
from src.qdrant_storage import QdrantStorage
import subprocess
from job_matching import main as job_matching_main
from job_resume_tailor import main as job_resume_tailor_main

def select_multiple_files_from_dir(directory: Path, file_type: str) -> list[Path]:
    """Muestra los archivos de un directorio y pide al usuario que seleccione uno o más."""
    print(f"\n--- Seleccionar {file_type} ---")
    files = sorted(list(directory.glob("*.pdf")))
    if not files:
        print(f"No se encontraron archivos en {directory}. Abortando.")
        sys.exit(1)

    for i, f in enumerate(files):
        print(f"[{i + 1}] {f.name}")
    
    print(f"\nSeleccione los {file_type} que desea procesar (ej: 1, 3, 5) o presione Enter para seleccionar todos.")
    
    selected_files = []
    while True:
        try:
            raw_input = input(f"Seleccione (1-{len(files)}): ")
            if not raw_input:
                return files
            
            selected_indices = [int(i.strip()) - 1 for i in raw_input.split(',')]
            
            if all(0 <= i < len(files) for i in selected_indices):
                selected_files = [files[i] for i in selected_indices]
                break
            else:
                print("Selección inválida. Asegúrese de que todos los números están en el rango.")
        except (ValueError, IndexError):
            print("Entrada inválida. Por favor, introduzca números separados por comas.")
            
    return selected_files

def select_file_from_dir(directory: Path, file_type: str) -> Path:
    """Muestra los archivos de un directorio y pide al usuario que seleccione uno."""
    print(f"\n--- Seleccionar {file_type} ---")
    files = sorted(list(directory.glob("*.pdf")))
    if not files:
        print(f"No se encontraron archivos en {directory}. Abortando.")
        sys.exit(1)

    for i, f in enumerate(files):
        print(f"[{i + 1}] {f.name}")

    choice = -1
    while choice < 1 or choice > len(files):
        try:
            raw_choice = input(f"Seleccione un {file_type} (1-{len(files)}): ")
            choice = int(raw_choice)
        except (ValueError, IndexError):
            print("Entrada inválida.")

    return files[choice - 1]

def extract_keywords_from_text(text: str) -> set:
    """Extrae un conjunto de palabras clave de un texto, eliminando palabras comunes y cortas."""
    words = re.findall(r'\b\w{3,}\b', text.lower()) # Palabras de 3 o más letras
    stop_words = {
        "the", "and", "or", "in", "on", "at", "to", "for", "with", "a", "an", "of", "as", "is", "are", "was", "were",
        "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "en", "con", "por", "para", "su", "sus"
    }
    return {word for word in words if word not in stop_words and not word.isdigit()}

def read_md_text(md_path: Path) -> str:
    """Lee todo el texto de un archivo Markdown."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error al leer el archivo Markdown {md_path}: {e}")
        return ""

def read_pdf_text(pdf_path: Path) -> str:
    """Lee todo el texto de un archivo PDF."""
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error al leer el PDF {pdf_path}: {e}")
        return ""

def analyze_keyword_changes(original_cv_text, tailored_cv_text, job_text):
    """Analiza y muestra los cambios de palabras clave entre los CVs y la oferta."""
    job_keywords = extract_keywords_from_text(job_text)
    original_cv_keywords = extract_keywords_from_text(original_cv_text)
    tailored_cv_keywords = extract_keywords_from_text(tailored_cv_text)

    # Palabras clave de la oferta que ahora están en el CV adaptado (y no estaban antes)
    added_keywords = (job_keywords & tailored_cv_keywords) - original_cv_keywords
    # Palabras clave de la oferta que ya estaban en el CV original
    existing_keywords = job_keywords & original_cv_keywords
    # Palabras clave de la oferta que aún faltan en el CV adaptado
    missing_keywords = job_keywords - tailored_cv_keywords

    print("\n" + "="*80)
    print("🔍 ANÁLISIS DE PALABRAS CLAVE (ATS)")
    print("="*80)
    print(f"Total de palabras clave relevantes en la oferta: {len(job_keywords)}")
    print("-" * 40)
    
    print(f"✅ Palabras clave de la oferta AÑADIDAS al CV ({len(added_keywords)}):")
    if added_keywords:
        print(", ".join(sorted(list(added_keywords))))
    else:
        print("(Ninguna)")

    print(f"\n👍 Palabras clave de la oferta que YA ESTABAN en el CV ({len(existing_keywords)}):")
    if existing_keywords:
        print(", ".join(sorted(list(existing_keywords))))
    else:
        print("(Ninguna)")

    print(f"\n⚠️ Palabras clave de la oferta que AÚN FALTAN en el CV ({len(missing_keywords)}):")
    if missing_keywords:
        print(", ".join(sorted(list(missing_keywords))))
    else:
        print("¡Excelente! Todas las palabras clave de la oferta están incluidas.")
    print("="*80)

def select_job_offer(matching_results: list) -> dict:
    """Muestra los resultados del matching y pide al usuario que seleccione una oferta."""
    print("\n--- Seleccionar Oferta para Tailoring ---")
    for i, res in enumerate(matching_results):
        print(f"[{i + 1}] Job ID: {res['job_id']}, Score: {res['score']:.4f}")
    
    choice = -1
    while choice < 1 or choice > len(matching_results):
        try:
            raw_choice = input(f"Seleccione una oferta (1-{len(matching_results)}): ")
            choice = int(raw_choice)
        except (ValueError, IndexError):
            print("Entrada inválida.")
    
    return matching_results[choice - 1]

def main():
    print("\n" + "="*80)
    print("🚀 PRUEBA DEL SISTEMA DE CHUNKING SEMÁNTICO OPTIMIZADO")
    print("="*80)
    print("📋 Configuración:")
    print(f"   • CVs: 1500 tokens/chunk, 300 tokens overlap (semántico inteligente)")
    print(f"   • Ofertas: 1000 tokens/chunk, 200 tokens overlap (semántico inteligente)")
    print(f"   • Estrategia: Preservar secciones completas cuando sea posible")
    print("="*80 + "\n")
    
    # 0. Limpieza de colecciones deshabilitada por el usuario.
    # print("--- PASO 0: Limpiando colecciones antiguas ---")
    # storage = QdrantStorage()
    # storage.delete_collection(CV_COLLECTION)
    # storage.delete_collection(JOB_COLLECTION)
    # print(f"Colecciones '{CV_COLLECTION}' y '{JOB_COLLECTION}' eliminadas.")

    # 1. Seleccionar y procesar CV
    cv_path = select_file_from_dir(CV_DIR, "CV")
    cv_doc_id = cv_path.stem
    print(f"\n🔄 PASO 2: Procesando Documentos (CV y Ofertas)")
    print("-" * 60)
    # Procesar el CV seleccionado
    print(f"Procesando CV: {cv_path.name}")
    subprocess.run([PYTHON_EXEC, 'process_documents.py', str(cv_path), '--collection', CV_COLLECTION, '--document-type', 'cv'], check=True)
    
    # Seleccionar y procesar las ofertas de trabajo deseadas
    selected_job_files = select_multiple_files_from_dir(JOBS_DIR, "ofertas de trabajo")
    selected_job_ids = []
    print(f"\nProcesando {len(selected_job_files)} oferta(s) seleccionada(s)...")
    for job_file in selected_job_files:
        print(f"  - {job_file.name}")
        subprocess.run([PYTHON_EXEC, 'process_documents.py', str(job_file), '--collection', JOB_COLLECTION, '--document-type', 'job'], check=True)
        selected_job_ids.append(job_file.stem)

    # 3. Ejecutar matching
    print(f"\n🎯 PASO 3: Realizando Matching Semántico CV ↔ Ofertas")
    print("-" * 60)
    old_stdout_match = sys.stdout
    redirected_output_match = sys.stdout = io.StringIO()
    # Ejecutar matching solo contra las ofertas seleccionadas
    job_matching_main([
        '--cv_id', cv_doc_id,
        '--job_ids', ','.join(selected_job_ids),
        '--cv_collection', CV_COLLECTION,
        '--job_collection', JOB_COLLECTION
    ])
    sys.stdout = old_stdout_match
    match_output = redirected_output_match.getvalue()
    try:
        matching_results = json.loads(match_output)
    except json.JSONDecodeError:
        print("❌ Error: El script de matching no produjo un JSON válido.")
        print("Salida recibida:", match_output)
        return

    if not matching_results:
        print("⚠️  No se encontraron ofertas de trabajo coincidentes.")
        return

    print(f"\n✅ Resultados del Matching Semántico (Top {len(matching_results)})")
    print("=" * 60)
    # Ordenar resultados por score descendente
    matching_results.sort(key=lambda x: x['score'], reverse=True)
    for i, match in enumerate(matching_results, 1):
        score_emoji = "🔥" if match['score'] > 0.8 else "⭐" if match['score'] > 0.6 else "📋"
        print(f"  {i}. {score_emoji} Oferta: {match['job_id']}, Score: {match['score']:.4f}")

    # 4. Seleccionar oferta para tailoring
    selected_match = select_job_offer(matching_results)
    if not selected_match:
        return

    selected_job_id = selected_match['job_id']
    initial_score = selected_match['score']

    # 5. Ejecutar tailoring y capturar la ruta del nuevo CV
    print(f"\n✨ PASO 4: Adaptando CV con IA para la Oferta Seleccionada")
    print(f"🎯 Oferta objetivo: {selected_job_id} (Score inicial: {initial_score:.4f})")
    print("-" * 60)

    old_stdout_tailor = sys.stdout
    redirected_output_tailor = sys.stdout = io.StringIO()
    job_resume_tailor_main([
        '--cv_id', cv_doc_id,
        '--job_id', selected_job_id,
        '--output_dir', str(TAILORED_CV_DIR),
        '--initial_score', str(initial_score) # Pasar el score
    ])
    sys.stdout = old_stdout_tailor
    tailor_output = redirected_output_tailor.getvalue()
    print(tailor_output) # Mostrar la salida original del script de tailoring

    tailored_cv_path_str = ""
    for line in tailor_output.strip().split('\n'):
        if line.startswith("TAILORED_CV_PATH:"):
            tailored_cv_path_str = line.split(":", 1)[1].strip()
            break
    
    if not tailored_cv_path_str:
        print("❌ Error: No se pudo encontrar la ruta del CV adaptado en la salida.")
        return

    tailored_cv_path = Path(tailored_cv_path_str)
    tailored_cv_id = tailored_cv_path.stem

    # 6. Procesar el CV adaptado
    print(f"\n🔄 PASO 5: Procesando el CV Adaptado para Re-evaluación")
    print(f"📄 Archivo: {tailored_cv_path.name}")
    print("-" * 60)
    subprocess.run([PYTHON_EXEC, 'process_documents.py', str(tailored_cv_path), '--collection', CV_COLLECTION, '--document-type', 'cv'], check=True)

    # 7. Re-ejecutar matching
    print(f"\n🎯 PASO 6: Re-evaluando el Matching Semántico")
    print(f"📄 CV Adaptado: {tailored_cv_id}")
    print(f"🎯 Oferta: {selected_job_id}")
    print("-" * 60)
    
    old_stdout_rematch = sys.stdout
    redirected_output_rematch = sys.stdout = io.StringIO()
    job_matching_main(['--cv_id', tailored_cv_id, '--job_ids', selected_job_id, '--cv_collection', CV_COLLECTION, '--job_collection', JOB_COLLECTION])
    sys.stdout = old_stdout_rematch
    rematch_output = redirected_output_rematch.getvalue()

    try:
        rematch_results = json.loads(rematch_output)
    except json.JSONDecodeError:
        print("❌ Error: El script de re-matching no produjo un JSON válido.")
        print("Salida recibida:", rematch_output)
        return

    # 8. Mostrar comparación
    original_score = next((m['score'] for m in matching_results if m['job_id'] == selected_job_id), 0)
    new_score = rematch_results[0]['score'] if rematch_results else 0

    print("\n" + "="*80)
    print("📊 COMPARACIÓN DE PUNTUACIONES (ANTES Y DESPUÉS)")
    print("="*80)
    print(f"🎯 Oferta: {selected_job_id}")
    print(f"   📄 CV Original: {cv_doc_id}")
    print(f"   📄 CV Adaptado: {tailored_cv_id}")
    print("-" * 40)
    print(f"   ⭐ Puntuación Original: {original_score:.4f}")
    print(f"   🔥 Puntuación Adaptada:  {new_score:.4f}")
    print("-" * 40)
    if new_score > original_score:
        print(f"   🎉 ¡Mejora de {(new_score - original_score):.4f} puntos!")
    elif new_score == original_score:
        print("   😐 La puntuación no cambió.")
    else:
        print("   ⚠️ La puntuación ha disminuido.")
    print("="*80)

    # 9. Analizar y mostrar los cambios de palabras clave
    original_cv_text = read_pdf_text(cv_path)
    # El CV adaptado ahora es un .md
    tailored_cv_text = read_md_text(tailored_cv_path)
    # Construir la ruta al PDF de la oferta de trabajo a partir de su ID
    job_offer_path = JOBS_DIR / f"{selected_job_id}.pdf"
    job_text = read_pdf_text(job_offer_path)

    if original_cv_text and tailored_cv_text and job_text:
        analyze_keyword_changes(original_cv_text, tailored_cv_text, job_text)


if __name__ == "__main__":
    main()
