import sys
import os
import subprocess
from pathlib import Path
import time
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
import streamlit.components.v1 as components
from qdrant_config import get_qdrant_client
import concurrent.futures
import streamlit as st
from pathlib import Path
import tempfile
import base64
from bs4 import BeautifulSoup
import re
from PIL import Image
import io

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

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Estado inicial ---
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.chat_history = []
    st.session_state.cv_path = None

st.set_page_config(page_title="OffHeadHunter", layout="wide")
# st.title("OffHeadHunter - Pipeline Laboral")
st.image("logo.png", width=450)  # puedes ajustar el tamaño

background_img = get_base64("background.png")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255,255,255,0.5), rgba(255,255,255,0.5)), url("data:image/png;base64,{background_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .stButton button {
        background-color: #ea833d;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 24px;
        font-weight: 800;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #BF6C32;
        color: white;
        font-weight: 800;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================================
#   Paso 1: Chat + Upload CV
# ====================================
if st.session_state.step == 1:
    st.header("Your new job starts here")
    col1, col2 = st.columns([2, 1])

    with col1:
        # ✅ Estilo CSS para la ventana del chat
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        .chat-window {
            height: 60vh;
            min-height: 200px;
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            background-color: #ffffff;
            border: 1px solid #ea833d;
            border-radius: 8px;
            box-shadow: 0 8px 8px rgba(0, 0, 0, 0.2); /* X-offset, Y-offset, blur, color */
            margin-bottom: 10px;
            scroll-behavior: smooth;
            display: flex;
            flex-direction: column;
        }
        .chat-messages {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            
        }
        .chat-bottom {
            float: left;
            clear: both;
            height: 0px;
        }
        .user-msg {
            font-family: 'Montserrat', sans-serif;
            font-size: 16px;
            font-weight: 500;
            color: #000000;
            background-color: #ffccff50;
            padding: 6px 10px;
            border-radius: 6px;
            margin-bottom: 4px;
            width: fit-content;
            max-width: 80%;
            align-self: flex-end;
            margin-bottom: 20px;
        }
        .bot-msg {
            font-family: 'Montserrat', sans-serif;
            font-size: 16px;
            font-weight: 500;
            color: #000000;
            background-color: #ffcc9930;
            padding: 6px 10px;
            border-radius: 6px;
            margin-bottom: 4px;
            width: fit-content;
            max-width: 80%;
            align-self: flex-start;
            margin-bottom: 20px;
        }
        div[data-testid="stFileDropzone"] {
            background-color: white !important;
        }
        </style>
        <script>
        var chatWindow = document.getElementById('chat-window');
        chatWindow.scrollTop = chatWindow.scrollHeight;
        </script>
        """, unsafe_allow_html=True)

        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = AgentChatbot()
            st.session_state.chat, st.session_state.chat_history = st.session_state.chatbot.start_session()
            st.session_state.finished_chat = False

        # ✅ Historial renderizado como HTML dentro del contenedor con auto-scroll
        chat_html = '''
        <div class="chat-window" id="chat-window">
        '''
        
        for role, msg in st.session_state.chat_history:
            if role == "Usuario":
                chat_html += f'<div class="user-msg">{msg}</div>'
            else:
                chat_html += f'<div class="bot-msg">{msg}</div>'
                
        chat_html += '''
        </div>
        '''
        
        st.markdown(chat_html, unsafe_allow_html=True)

        # ✅ Input debajo
        if not st.session_state.finished_chat:
            user_input = st.chat_input("Escribe tus preferencias laborales...")
            if user_input:
                # Dejar que send_message actualice el historial
                response, st.session_state.chat_history, finished = st.session_state.chatbot.send_message(
                    st.session_state.chat, st.session_state.chat_history, user_input
                )
                st.session_state.finished_chat = finished
                st.rerun()

    with col2:
        st.subheader("Upload your CV")
        uploaded_cv = st.file_uploader("Select your CV (PDF or DOCX)", type=["pdf", "docx"])
        st.markdown("""
        <style>
        /* Contenedor del uploader */
        .stFileUploader {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
        }
        /* Texto dentro del uploader */
        .stFileUploader label {
            font-weight: bold;
            color: #000000;
        }
        </style>
        """, unsafe_allow_html=True)

        if uploaded_cv:
            tmp_dir = tempfile.gettempdir()
            cv_path = Path(tmp_dir) / uploaded_cv.name
            with open(cv_path, 'wb') as f:
                f.write(uploaded_cv.read())
            st.session_state.cv_path = str(cv_path)
            st.success(f"CV cargado: {uploaded_cv.name}")

    if st.session_state.finished_chat and st.session_state.cv_path:
        if st.button("Continuar con la búsqueda de ofertas"):
            st.session_state.step = 2
            st.rerun()

# ====================================
#   Paso 2: Scraping + selección
# ====================================

elif st.session_state.step == 2:
    st.header("Searching for job offers")

    # === Estilos base para las tarjetas ===
    st.markdown("""
    <style>
    .custom-card {
        border-radius: 10px;
        margin-bottom: 12px;
        padding: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: background-color 0.3s ease, border 0.3s ease;
    }
    .custom-title {
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 6px;
    }
    .custom-desc {
        font-size: 14px;
        color: #333333;
        margin-bottom: 6px;
    }
    .custom-meta {
        font-size: 13px;
        color: #555555;
    }
    </style>
    """, unsafe_allow_html=True)

    # ✅ 1. Construir URL personalizada si no existe
    if 'search_url' not in st.session_state:
        st.info("🔍 Analizando la conversación para construir la búsqueda personalizada...")
        agent = JobSearchAgent()
        history_text = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
        user_profile = agent.parse_profile_from_text(history_text)

        puesto = user_profile.get('job_title')
        modalidad = user_profile.get('work_modality')
        salario_min = user_profile.get('salary_expectation')
        localidades = user_profile.get('location')

        st.session_state.search_url = generar_url_infojobs(
            puesto=puesto,
            modalidad=modalidad,
            salario_minimo=salario_min,
            localidades=localidades
        )

    # ✅ 2. Scraping si no hay ofertas todavía
    if 'job_offers' not in st.session_state:
        st.info(f"🌐 Buscando ofertas en InfoJobs para: {st.session_state.search_url}")
        st.session_state.job_offers = scrape_jobs(st.session_state.search_url)

    offers = st.session_state.job_offers
    st.info(f"{len(offers)} ofertas encontradas")

    if not offers:
        st.warning("No se encontraron ofertas. Intenta con otros criterios.")
        if st.button("Reiniciar"):
            st.session_state.clear()
            st.rerun()

    st.write("Selecciona las ofertas que te interesen:")

    selected_indices = []

    # === Renderizado de tarjetas personalizadas ===
    for idx, offer in enumerate(offers):

        # Estado actual de la tarjeta
        is_checked = st.session_state.get(f"offer_{idx}", False)
        bg_color = "#ffcc9990" if is_checked else "#f8faff98"
        border_color = "#df0c9e85" if is_checked else "#ea833d"

        # Información de la oferta
        title = offer.get('title', 'Sin título')
        company = offer.get('company', 'Empresa')
        description = offer.get('description', '')[:250]
        location = ", ".join(offer.get('locations', []))
        tags = offer.get('tags', [])
        tags_text = ", ".join(reversed(tags)) if tags else "No especificado"

        salary = offer.get('salary_range')
        salary_text = "No especificado"
        if salary:
            min_salary = salary.get('min')
            max_salary = salary.get('max')
            currency = salary.get('currency', 'EUR')
            if min_salary and max_salary:
                salary_text = f"{min_salary} - {max_salary} {currency}"
            elif min_salary:
                salary_text = f"Desde {min_salary} {currency}"
            elif max_salary:
                salary_text = f"Hasta {max_salary} {currency}"

        url = offer.get('url')

        checked = st.checkbox("Seleccionar", key=f"offer_{idx}")

        # Tarjeta renderizada
        st.markdown(f"""
        <div class="custom-card" style="background-color:{bg_color}; border:2px solid {border_color};">
            <div class="custom-title">{title} – {company}</div>
            <div class="custom-desc">{description}...</div>
            <div class="custom-meta"><strong>Ubicación:</strong> {location}</div>
            <div class="custom-meta"><strong>Modalidad:</strong> {tags_text}</div>
            <div class="custom-meta"><strong>Salario:</strong> {salary_text}</div>
            {'<a href="'+url+'" target="_blank">🔗 Ver oferta original</a>' if url else ''}
        </div>
        """, unsafe_allow_html=True)

        if checked:
            selected_indices.append(idx)

    # === Botón procesar ofertas ===
    if selected_indices:
        if st.button("Procesar ofertas seleccionadas"):
            st.session_state.selected_offers = [offers[i] for i in selected_indices]
            st.session_state.exported_jobs = []

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
#   Paso 3: Procesar PDFs y Matching (con tarjetas)
# ====================================
elif st.session_state.step == 3:
    st.header("Matching Scores with your CV")

    def split_job_id(job_id: str):
        """Devuelve (source_id, external_id) a partir de job_id de Qdrant."""
        if "-i" in job_id:
            parts = job_id.split("-i")
            return parts[0], "i" + parts[1]
        return job_id, None


    # === Estilos de tarjetas ===
    st.markdown("""
    <style>
    .match-card {
        border-radius: 10px;
        margin-bottom: 12px;
        padding: 14px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        background-color: #f8faff98;
        transition: background-color 0.3s ease, border 0.3s ease;
        border: 2px solid #ea833d;
    }
    .match-card:hover {
        filter: brightness(0.97);
    }
    .match-title {
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 6px;
        color: #000000;
    }
    .match-score {
        font-size: 18px;
        color: #DE4A28;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .match-meta {
        font-size: 13px;
        color: #444444;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'matching_results' not in st.session_state:
        st.info("🔄 Procesando documentos y generando embeddings...")

        cv_path = st.session_state.cv_path
        cv_id = Path(cv_path).stem

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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, job_meta) for job_meta in st.session_state.exported_jobs]
            for future in concurrent.futures.as_completed(futures):
                job_id = future.result()
                if job_id:
                    newly_processed_job_ids.append(job_id)

        st.session_state.job_ids = newly_processed_job_ids

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

    if st.session_state.matching_results:
        st.subheader("Results")

        selected_offer_idx = None

        for idx, res in enumerate(st.session_state.matching_results):
            job_id = res['job_id']
            score = res['score']
            source_id, external_id = split_job_id(job_id)

            offer_data = next(
                (
                    o for o in st.session_state.selected_offers
                    if o.get('source_id') == source_id and o.get('external_id') == external_id
                ),
                None
            )
            if not offer_data:
                continue

            # === Tarjeta personalizada ===
            st.markdown(f"""
            <div class="match-card">
                <div class="match-score">Match Score: {score:.2f}</div>
                <div class="match-title">{offer_data.get('title', 'Sin título')} – {offer_data.get('company', 'Empresa')}</div>
                <div class="match-meta"><strong>Descripción:</strong> {offer_data.get('description', '')[:250]}...</div>
                <div class="match-meta"><strong>Ubicación:</strong> {', '.join(offer_data.get('locations', []))}</div>
                <div class="match-meta"><strong>Modalidad:</strong> {', '.join(reversed(offer_data.get('tags', [])))} </div>
                <div class="match-meta"><strong>Salario:</strong> {
                    f"{offer_data['salary_range'].get('min', '')} - {offer_data['salary_range'].get('max', '')} {offer_data['salary_range'].get('currency', 'EUR')}"
                    if offer_data.get('salary_range') else 'No especificado'
                }</div>
                {'<a href="'+offer_data.get("url", "")+'" target="_blank">🔗 Ver oferta original</a>' if offer_data.get("url") else ''}
            </div>
            """, unsafe_allow_html=True)

            # ✅ Radio debajo de la tarjeta para seleccionar
            if st.radio(
                "Seleccionar esta oferta",
                options=[False, True],
                index=0,
                key=f"match_radio_{idx}",
                horizontal=True,
                label_visibility="collapsed"
            ):
                selected_offer_idx = idx

        if selected_offer_idx is not None and st.button("Adaptar CV"):
            chosen_res = st.session_state.matching_results[selected_offer_idx]
            st.session_state.selected_job_for_tailoring = chosen_res['job_id']
            st.session_state.initial_score = chosen_res['score']
            st.session_state.step = 4
            st.rerun()
    else:
        st.error("❌ No se encontraron resultados de matching.")



# ====================================
#   Paso 4: Tailoring + Comparación
# ====================================
elif st.session_state.step == 4:
    st.header("Check your tailored CV")

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
        st.subheader("Preview")
        
        # Leer el contenido del CV adaptado
        try:
            with open(st.session_state.tailored_cv_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = "(No se pudo cargar el contenido)"
        
        # Mostrar el CV en formato HTML con estilo
        st.markdown("""
        <style>
        .cv-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 800px;
        }
        .cv-header {
            text-align: center;
            border-bottom: 2px solid #007bff;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .cv-section {
            margin-bottom: 20px;
        }
        .cv-section-title {
            color: #007bff;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        .cv-content {
            line-height: 1.6;
        }
        .download-btn {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 0;
        }
        .download-btn:hover {
            background-color: #0056b3;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Botón de descarga PDF
        if st.button("Descargar CV como PDF"):
            # Crear un nombre de archivo limpio
            filename = os.path.basename(str(st.session_state.tailored_cv_path)).replace('.md', '.pdf')
            
            # Convertir markdown a HTML básico para la descarga
            import markdown
            html_content = markdown.markdown(content)
            
            # Crear HTML completo para PDF
            html_for_pdf = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2, h3 {{ color: #333; }}
                    h1 {{ border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                    h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                    .section {{ margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Guardar HTML temporalmente
            temp_html_path = str(st.session_state.tailored_cv_path).replace('.md', '_temp.html')
            with open(temp_html_path, 'w', encoding='utf-8') as f:
                f.write(html_for_pdf)
            
            # Usar weasyprint para convertir a PDF
            try:
                import weasyprint
                pdf_path = str(st.session_state.tailored_cv_path).replace('.md', '.pdf')
                weasyprint.HTML(temp_html_path).write_pdf(pdf_path)
                
                # Ofrecer descarga
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=".Descargar PDF",
                        data=f,
                        file_name=filename,
                        mime="application/pdf"
                    )
                
                # Limpiar archivos temporales
                os.remove(temp_html_path)
                os.remove(pdf_path)
            except Exception as e:
                st.error(f"Error al generar PDF: {str(e)}")
                st.info("Asegúrate de tener instalado weasyprint: pip install weasyprint")

        # Añadir un modo de edición en el paso 4 del CV adaptado
        if 'cv_edit_mode' not in st.session_state:
            st.session_state.cv_edit_mode = "Vista"

        # Botones para cambiar entre vista y edición
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Vista previa", use_container_width=True, 
                         disabled=st.session_state.cv_edit_mode == "Vista"):
                st.session_state.cv_edit_mode = "Vista"
                st.rerun()
        with col2:
            if st.button("✏️ Editar CV", use_container_width=True,
                         disabled=st.session_state.cv_edit_mode == "Editar"):
                st.session_state.cv_edit_mode = "Editar"
                st.rerun()

        # Dependiendo del modo, mostrar vista o editor
        if st.session_state.cv_edit_mode == "Vista":
            # Código actual para mostrar el CV
            try:
                import markdown
                html_content = markdown.markdown(content)
                st.markdown(f"""
                <div class="cv-container">
                    <div class="cv-content">
                        {html_content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error al mostrar el CV: {str(e)}")
                st.text_area("Contenido CV Adaptado", content, height=300)
        else:
            # Modo de edición sencillo
            st.subheader("Editor de CV")
            
            # Editor visual sencillo estilo Word/Docs
            editor_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Editor Sencillo</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                    .editor-container { border: 1px solid #ccc; border-radius: 5px; margin-bottom: 10px; }
                    .toolbar { 
                        padding: 8px; 
                        background: #f5f5f5; 
                        border-bottom: 1px solid #ddd;
                        display: flex;
                        flex-wrap: wrap;
                        gap: 5px;
                    }
                    .toolbar button { 
                        padding: 5px 10px; 
                        background: #fff; 
                        border: 1px solid #ccc;
                        border-radius: 3px;
                        cursor: pointer;
                    }
                    .toolbar button:hover { background: #e9e9e9; }
                    .toolbar select { 
                        padding: 5px;
                        border: 1px solid #ccc;
                        border-radius: 3px;
                    }
                    #editor-content {
                        min-height: 400px;
                        padding: 15px;
                        overflow-y: auto;
                        background: white;
                        font-size: 14px;
                    }
                    .hidden { display: none; }
                </style>
            </head>
            <body>
                <div class="editor-container">
                    <div class="toolbar">
                        <select id="heading-select">
                            <option value="p">Párrafo</option>
                            <option value="h1">Título 1</option>
                            <option value="h2">Título 2</option>
                            <option value="h3">Título 3</option>
                        </select>
                        <button id="btn-bold" title="Negrita"><b>B</b></button>
                        <button id="btn-italic" title="Cursiva"><i>I</i></button>
                        <button id="btn-underline" title="Subrayado"><u>U</u></button>
                        <button id="btn-bullet" title="Lista con viñetas">• Lista</button>
                        <button id="btn-numbered" title="Lista numerada">1. Lista</button>
                        <button id="btn-image" title="Insertar imagen">🖼️ Imagen</button>
                        <input type="file" id="image-input" class="hidden" accept="image/*">
                    </div>
                    <div id="editor-content" contenteditable="true"></div>
                </div>
                
                <script>
                    // Inicializar el editor con el contenido actual
                    const editorContent = document.getElementById('editor-content');
                    const initialContent = `CONTENT_PLACEHOLDER`;
                    
                    // Convertir Markdown a HTML básico
                    function markdownToHtml(markdown) {
                        return markdown
                            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
                            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                            .replace(/\*(.*?)\*/g, '<em>$1</em>')
                            .replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1">')
                            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>')
                            .replace(/^- (.*$)/gm, '<ul><li>$1</li></ul>')
                            .replace(/^[0-9]+\\. (.*$)/gm, '<ol><li>$1</li></ol>');
                    }
                    
                    // Establecer contenido inicial
                    editorContent.innerHTML = markdownToHtml(initialContent);
                    
                    // Función para obtener el HTML del editor
                    function getEditorHtml() {
                        return editorContent.innerHTML;
                    }
                    
                    // Aplicar formato cuando se hace clic en los botones
                    document.getElementById('btn-bold').addEventListener('click', () => {
                        document.execCommand('bold', false, null);
                    });
                    
                    document.getElementById('btn-italic').addEventListener('click', () => {
                        document.execCommand('italic', false, null);
                    });
                    
                    document.getElementById('btn-underline').addEventListener('click', () => {
                        document.execCommand('underline', false, null);
                    });
                    
                    document.getElementById('btn-bullet').addEventListener('click', () => {
                        document.execCommand('insertUnorderedList', false, null);
                    });
                    
                    document.getElementById('btn-numbered').addEventListener('click', () => {
                        document.execCommand('insertOrderedList', false, null);
                    });
                    
                    // Manejar la selección de encabezados
                    document.getElementById('heading-select').addEventListener('change', function() {
                        const value = this.value;
                        if (value === 'p') {
                            document.execCommand('formatBlock', false, 'p');
                        } else {
                            document.execCommand('formatBlock', false, value);
                        }
                    });
                    
                    // Manejar la subida de imágenes
                    document.getElementById('btn-image').addEventListener('click', function() {
                        document.getElementById('image-input').click();
                    });
                    
                    document.getElementById('image-input').addEventListener('change', function(e) {
                        const file = e.target.files[0];
                        if (file) {
                            const reader = new FileReader();
                            reader.onload = function(event) {
                                document.execCommand('insertImage', false, event.target.result);
                            };
                            reader.readAsDataURL(file);
                        }
                    });
                    
                    // Enviar el contenido de vuelta a Streamlit cuando cambia
                    editorContent.addEventListener('input', function() {
                        // Intentar enviar el contenido a Streamlit
                        try {
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: getEditorHtml()
                            }, '*');
                        } catch(e) {
                            console.error('Error al enviar datos a Streamlit:', e);
                        }
                    });
                </script>
            </body>
            </html>
            """.replace('CONTENT_PLACEHOLDER', content.replace('`', '\\`').replace("'", "\\'").replace('"', '\\"'))
            
            # Mostrar el editor
            if 'cv_edited_html' not in st.session_state:
                st.session_state.cv_edited_html = ""
            
            editor_result = components.html(editor_html, height=500, scrolling=True)
            
            # Campo oculto para guardar el HTML resultante
            if editor_result:
                st.session_state.cv_edited_html = editor_result
            
            # Botones para guardar y cancelar
            col1, col2, col3 = st.columns([2,2,6])
            with col1:
                if st.button("💾 Guardar", use_container_width=True):
                    try:
                        # Convertir HTML a Markdown básico
                        html_content = st.session_state.cv_edited_html
                        if not html_content:
                            st.error("No hay contenido para guardar")
                        else:
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # Procesar las imágenes (base64)
                            for img in soup.find_all('img'):
                                if img.get('src', '').startswith('data:image'):
                                    # Extraer datos base64
                                    img_data = img['src'].split(',')[1]
                                    img_data = base64.b64decode(img_data)
                                    img_obj = Image.open(io.BytesIO(img_data))
                                    
                                    # Guardar imagen
                                    img_path = os.path.join(
                                        os.path.dirname(st.session_state.tailored_cv_path),
                                        f"{Path(st.session_state.tailored_cv_path).stem}_img_{len(os.listdir(os.path.dirname(st.session_state.tailored_cv_path)))}.png"
                                    )
                                    img_obj.save(img_path)
                                    
                                    # Actualizar la ruta en el HTML
                                    img['src'] = os.path.basename(img_path)
                            
                            # Guardar el HTML modificado
                            with open(st.session_state.tailored_cv_path.replace('.md', '.html'), 'w', encoding='utf-8') as f:
                                f.write(str(soup))
                            
                            # También guardar una versión markdown para compatibilidad
                            # Implementación simplificada
                            markdown_content = str(soup)
                            markdown_content = re.sub(r'<h1>(.*?)</h1>', r'# \1', markdown_content)
                            markdown_content = re.sub(r'<h2>(.*?)</h2>', r'## \1', markdown_content)
                            markdown_content = re.sub(r'<h3>(.*?)</h3>', r'### \1', markdown_content)
                            markdown_content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', markdown_content)
                            markdown_content = re.sub(r'<em>(.*?)</em>', r'*\1*', markdown_content)
                            markdown_content = re.sub(r'<img src="(.*?)" alt="(.*?)">', r'![\2](\1)', markdown_content)
                            
                            with open(st.session_state.tailored_cv_path, 'w', encoding='utf-8') as f:
                                f.write(markdown_content)
                            
                            st.success("✅ CV guardado correctamente")
                            st.session_state.cv_edit_mode = "Vista"
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error al guardar: {str(e)}")
            
            with col2:
                if st.button("❌ Cancelar", use_container_width=True):
                    st.session_state.cv_edit_mode = "Vista"
                    st.rerun()

        # Comparación de scores
        st.subheader("Compare your new match score")
        col1, col2 = st.columns(2)
        col1.metric("Score Original", f"{initial_score:.2f}")
        col2.metric("Score Adaptado", f"{new_score:.2f}")

        if st.button("Reiniciar"):
            st.session_state.clear()
            st.rerun()
