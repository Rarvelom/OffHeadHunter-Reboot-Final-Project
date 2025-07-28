import os
import argparse
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from pymongo import MongoClient
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Importar configuración y utilidades centralizadas
from qdrant_config import get_qdrant_client, CV_COLLECTION, JOB_COLLECTION
from src.qdrant_utils import get_all_chunks

# --- CONFIG ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MONGO_URI = os.getenv("MONGO_URI")

# --- HELPERS ---
# Función get_all_chunks movida a src/qdrant_utils.py para evitar duplicación

def extract_keywords(text: str):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {"the", "and", "or", "in", "on", "at", "to", "for", "with", "a", "de", "la", "el", "y", "en", "un", "una", "por", "con", "que", "los", "las"}
    return list({w for w in words if w not in stop_words and len(w) > 2})

def calculate_keyword_overlap(resume_keywords, job_keywords):
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    matching = list(resume_set & job_set)
    missing = list(job_set - resume_set)
    return matching, missing

def generate_tailored_resume(
    original_resume: str,
    job_description: str,
    matching_keywords,
    missing_keywords
) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
You are an expert resume writer. Please tailor the following resume to better match the job description.\n
Focus on emphasizing the matching skills and incorporating missing keywords naturally.\n
Job Description:\n{job_description}\n
Matching Keywords (emphasize these):\n{', '.join(matching_keywords)}\n
Missing Keywords (try to incorporate these naturally):\n{', '.join(missing_keywords)}\n
Original Resume:\n{original_resume}\n
Please return ONLY the improved resume text, with no additional commentary or explanations.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating tailored resume: {str(e)}")
        return original_resume

def get_job_details_by_id(job_id: str) -> str:
    """
    Fetches job details from MongoDB's job_offers collection using embedding_vector_id_qdrant.
    
    Args:
        job_id: The Qdrant vector ID to search for in the embedding_vector_id_qdrant field
        
    Returns:
        str: Combined text from requirements_text and description fields, or empty string if not found
    """
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client.get_database("offheadhunter_db")
        job_offers = db.get_collection("job_offers")
        
        # Find the job by embedding_vector_id_qdrant
        job_data = job_offers.find_one({"embedding_vector_id_qdrant": job_id})
        
        if not job_data:
            print(f"No se encontró ninguna oferta con embedding_vector_id_qdrant: {job_id}")
            return ""
            
        # Extract and combine the required fields
        requirements = job_data.get("requirements_text", "")  # Note: "tetx" appears to be a typo in the field name
        description = job_data.get("description", "")
        
        # Combine the fields with a space in between
        full_text = " ".join(filter(None, [requirements, description]))
        return full_text
        
    except Exception as e:
        print(f"Error al obtener detalles del trabajo: {e}")
        return ""

def save_text_as_pdf(text: str, output_path: Path):
    """Saves a string of text to a PDF file."""
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(line, styles['Normal']) for line in text.split('\n')]
        doc.build(story)
        logger.info(f"Tailored resume saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")

# --- CLI ---
def main(args_list=None):
    """Main function to run the resume tailoring script from the command line.
    
    Args:
        args_list: Optional list of command line arguments (for programmatic invocation)
    """
    parser = argparse.ArgumentParser(description="Resume Tailoring with Gemini")
    parser.add_argument('--cv_id', type=str, required=True, help='CV document ID')
    parser.add_argument('--job_id', type=str, required=True, help='Job document ID')
    parser.add_argument('--cv_collection', default=CV_COLLECTION, help='Qdrant collection for CVs')
    parser.add_argument('--job_collection', default=JOB_COLLECTION, help='Qdrant collection for jobs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the tailored resume')
    args = parser.parse_args(args_list)
    
    # Usar cliente centralizado con configuración optimizada
    client = get_qdrant_client(use_http=True, timeout=60.0)
    cv_chunks = get_all_chunks(client, args.cv_collection, args.cv_id)
    job_chunks = get_all_chunks(client, args.job_collection, args.job_id)
    cv_text = '\n'.join([c.payload['text'] for c in cv_chunks])
    job_text = '\n'.join([c.payload['text'] for c in job_chunks])
    cv_keywords = extract_keywords(cv_text)
    job_keywords = extract_keywords(job_text)
    matching, missing = calculate_keyword_overlap(cv_keywords, job_keywords)
    tailored = generate_tailored_resume(cv_text, job_text, matching, missing)

    # Guardar el resultado como PDF
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_filename = f"CV_{args.cv_id}_adaptado_para_{args.job_id}.pdf"
    output_path = output_dir / output_filename

    save_text_as_pdf(tailored, output_path)

    print("\n--- Proceso de Adaptación Finalizado ---")
    print(f"\nEl CV adaptado ha sido guardado en: {output_path}")

if __name__ == "__main__":
    main()
