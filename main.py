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
from job_matching import match_cv_to_jobs
from job_resume_tailor import run_resume_tailoring
from qdrant_config import get_qdrant_client
import concurrent.futures
import streamlit as st
from pathlib import Path
import tempfile

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

# --- Estado inicial ---
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.chat_history = []
    st.session_state.cv_path = None

st.set_page_config(page_title="OffHeadHunter", layout="wide")
st.title("🎯 OffHeadHunter - Pipeline Laboral")

# ====================================
#   Paso 1: Chat + Upload CV
# ====================================
if st.session_state.step == 1:
    st.header("Paso 1: Conversa con el asistente y sube tu CV")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chatbot")
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = AgentChatbot()
            st.session_state.chat, st.session_state.chat_history = st.session_state.chatbot.start_session()
            st.session_state.finished_chat = False

        for role, msg in st.session_state.chat_history:
            with st.chat_message("user" if role == "Usuario" else "assistant"):
                st.markdown(msg)

        if not st.session_state.finished_chat:
            user_input = st.chat_input("Escribe tus preferencias laborales...")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                response, st.session_state.chat_history, finished = st.session_state.chatbot.send_message(
                    st.session_state.chat, st.session_state.chat_history, user_input
                )
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.finished_chat = finished

    with col2:
        st.subheader("📄 Sube tu CV")
        uploaded_cv = st.file_uploader("Selecciona tu CV (PDF o DOCX)", type=["pdf", "docx"])
        if uploaded_cv:
            tmp_dir = tempfile.gettempdir()
            cv_path = Path(tmp_dir) / uploaded_cv.name
            with open(cv_path, 'wb') as f:
                f.write(uploaded_cv.read())
            st.session_state.cv_path = str(cv_path)
            st.success(f"CV cargado: {uploaded_cv.name}")

    if st.session_state.finished_chat and st.session_state.cv_path:
        if st.button("➡️ Continuar a scraping de ofertas"):
            st.session_state.step = 2
            st.rerun()

# ====================================
#   Paso 2: Scraping + selección
# ====================================
elif st.session_state.step == 2:
    st.header("Paso 2: Ofertas encontradas")

    # Generar URL de InfoJobs según perfil (mock aquí, integrar real luego)
    search_url = "https://www.infojobs.net/ofertas-trabajo?keyword=developer"
    if 'job_offers' not in st.session_state:
        st.session_state.job_offers = scrape_jobs(search_url)

    offers = st.session_state.job_offers
    if not offers:
        st.warning("No se encontraron ofertas. Intenta con otros criterios.")
        if st.button("🔄 Reiniciar"):
            st.session_state.clear()
            st.rerun()

    st.write("Selecciona las ofertas que te interesen:")

    selected_indices = []
    for idx, offer in enumerate(offers):
        with st.expander(f"{offer.get('title', 'Sin título')} – {offer.get('company', 'Empresa')}"):
            st.markdown(f"**Ubicación:** {', '.join(offer.get('locations', []))}")
            st.markdown(f"**Descripción:** {offer.get('description', '')[:250]}...")
            checked = st.checkbox("Seleccionar", key=f"offer_{idx}")
            if checked:
                selected_indices.append(idx)

    if selected_indices:
        if st.button("➡️ Procesar ofertas seleccionadas"):
            st.session_state.selected_offers = [offers[i] for i in selected_indices]
            st.session_state.exported_jobs = []

            # --- Exportar a PDF y MongoDB ---
            for offer in st.session_state.selected_offers:
                result = export_ij_offer_to_pdf(offer, output_dir="uploads/job_descriptions")
                if result and result.get('success'):
                    st.session_state.exported_jobs.append(result)

            if st.session_state.exported_jobs:
                st.success(f"{len(st.session_state.exported_jobs)} ofertas exportadas correctamente.")
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("No se pudo exportar ninguna oferta. Intenta de nuevo.")

# ====================================
#   Paso 3: Procesar PDFs y Matching
# ====================================
elif st.session_state.step == 3:
    st.header("Paso 3: Matching CV ↔ Ofertas")

    if 'matching_results' not in st.session_state:
        st.info("🔄 Procesando documentos y generando embeddings...")

        cv_path = st.session_state.cv_path
        cv_id = Path(cv_path).stem

        # Inicializar TextProcessor y QdrantStorage una sola vez
        text_processor = TextProcessor()
        job_storage = QdrantStorage(collection_name='job_embeddings_BGE2')

        newly_processed_job_ids = []

        def process_job(job_meta):
            pdf_filename = job_meta.get('pdf_filename')
            if not pdf_filename:
                return None
            job_path = JOBS_DIR / Path(pdf_filename).name
            result = process_document_file(
                file_path=job_path,
                collection_name='job_embeddings_BGE2',
                document_type='job',
                text_processor=text_processor,
                storage=job_storage
            )
            if result.get("success"):
                return Path(result.get('file_name', '')).stem
            return None

        # Procesar ofertas en paralelo
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, job_meta) for job_meta in st.session_state.exported_jobs]
            for future in concurrent.futures.as_completed(futures):
                job_id = future.result()
                if job_id:
                    newly_processed_job_ids.append(job_id)

        st.session_state.job_ids = newly_processed_job_ids

        # Matching
        st.info("🔍 Ejecutando matching...")
        qdrant_client = get_qdrant_client()
        results = match_cv_to_jobs(
            client=qdrant_client,
            cv_id=cv_id,
            cv_collection="cv_embeddings_BGE2",
            job_collection="job_embeddings_BGE2",
            specific_job_ids=newly_processed_job_ids
        )
        st.session_state.matching_results = results

    # Mostrar resultados
    if st.session_state.matching_results:
        st.subheader("Resultados de Matching")
        choices = []
        for res in st.session_state.matching_results:
            job_id = res['job_id']
            score = res['score']
            choices.append(f"{job_id} (Score: {score:.2f})")

        selected = st.radio("Selecciona una oferta para adaptar tu CV:", choices)
        if selected and st.button("➡️ Adaptar CV"):
            st.session_state.selected_job_for_tailoring = selected.split(" (Score")[0]
            st.session_state.initial_score = float(selected.split("Score: ")[1][:-1])
            st.session_state.step = 4
            st.rerun()
    else:
        st.error("❌ No se encontraron resultados de matching.")

# ====================================
#   Paso 4: Tailoring + Comparación
# ====================================
elif st.session_state.step == 4:
    st.header("Paso 4: CV Adaptado y Comparación de Scores")

    cv_path = st.session_state.cv_path
    cv_id = Path(cv_path).stem
    job_id = st.session_state.selected_job_for_tailoring
    initial_score = st.session_state.initial_score

    st.info(f"🔧 Adaptando CV '{cv_id}' para oferta '{job_id}'...")

    if 'tailored_cv_path' not in st.session_state:
        try:
            qdrant_client = get_qdrant_client()
            tailored_cv_path = run_resume_tailoring(
                cv_id=cv_id,
                job_id=job_id,
                initial_score=initial_score,
                output_dir=str(TAILORED_CV_DIR),
                client=qdrant_client
            )
            st.session_state.tailored_cv_path = tailored_cv_path
        except Exception as e:
            st.error(f"❌ Error al adaptar CV: {e}")

    if 'tailored_cv_path' in st.session_state:
        # Procesar CV adaptado para generar embeddings
        st.info("🔄 Procesando CV adaptado...")
        cv_storage = QdrantStorage(collection_name='cv_embeddings_BGE2')
        result = process_document_file(
            file_path=st.session_state.tailored_cv_path,
            collection_name='cv_embeddings_BGE2',
            document_type='cv'
        )
        tailored_cv_id = Path(result.get('file_name', '')).stem

        # Re-matching
        st.info("🔍 Calculando nuevo score...")
        qdrant_client = get_qdrant_client()
        rematch_data = match_cv_to_jobs(
            client=qdrant_client,
            cv_id=tailored_cv_id,
            cv_collection="cv_embeddings_BGE2",
            job_collection="job_embeddings_BGE2",
            specific_job_ids=[job_id]
        )
        new_score = rematch_data[0]['score'] if rematch_data else 0.0

        # Vista previa CV
        st.subheader("📄 Vista previa CV Adaptado")
        try:
            content = Path(st.session_state.tailored_cv_path).read_text(encoding='utf-8')
        except:
            content = "(No se pudo cargar el contenido)"
        st.text_area("Contenido CV Adaptado", content, height=300)

        # Comparación de scores
        st.subheader("📊 Comparación de Scores")
        col1, col2 = st.columns(2)
        col1.metric("Score Original", f"{initial_score:.2f}")
        col2.metric("Score Adaptado", f"{new_score:.2f}")

        if st.button("🔄 Reiniciar"):
            st.session_state.clear()
            st.rerun()
