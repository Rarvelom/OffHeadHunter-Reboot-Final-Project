import os
import argparse
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Importar configuración y utilidades centralizadas
from qdrant_config import get_qdrant_client, CV_COLLECTION, JOB_COLLECTION
from src.qdrant_utils import get_all_chunks
from mongodb_schema import db  # Importar la instancia de la base de datos
from datetime import datetime

# --- CONFIG ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- HELPERS ---

def extract_keywords(text: str):
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {"the", "and", "or", "in", "on", "at", "to", "for", "with", "a", "de", "la", "el", "y", "en", "un", "una", "por", "con", "que", "los", "las"}
    return list({w for w in words if w not in stop_words and len(w) > 2})

def generate_tailored_resume(
    raw_resume: str,
    raw_job_description: str,
    extracted_resume_keywords: list,
    extracted_job_keywords: list,
    current_cosine_similarity: float
) -> str:
    """Generates a tailored resume using the new detailed prompt."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
You are an expert multilingual resume editor and talent acquisition specialist with deep knowledge of ATS-optimized resume structures. Your task is to revise and restructure the following resume so it aligns as closely as possible with the provided job description and extracted keywords, while also improving its compatibility with Applicant Tracking Systems (ATS).

Instructions:
- **Detect the language** (English or Spanish) of the resume and job description.
- Rewrite the resume in the **same language** as the job description.
- Restructure the resume using a **universal ATS-friendly format** with the following sections (if applicable): 
  - Header (Name & Contact Info)
  - Professional Summary
  - Key Skills
  - Work Experience
  - Education
  - Certifications
  - Languages
  - Additional Information
- Carefully review the job description and extracted keywords.
- Update the resume by:
  - Naturally incorporating relevant skills and terminology from the job description and keyword list.
  - Rewriting, adding, or removing content to better match the role.
  - Using clear formatting, action verbs, and quantifiable achievements where possible.
  - Maintaining a **professional tone** and avoiding keyword stuffing.
- The current cosine similarity score is {current_cosine_similarity:.4f}. Revise the resume to further increase this score.

Output:
- **ONLY output the improved and reformatted resume**, using markdown.
- Do not include explanations, headers, or comments outside the resume.

Job Description:
```md
{raw_job_description}


Extracted Job Keywords:
```md
{', '.join(extracted_job_keywords)}
```

Original Resume:
```md
{raw_resume}
```

Extracted Resume Keywords:
```md
{', '.join(extracted_resume_keywords)}
```

NOTE: ONLY OUTPUT THE IMPROVED UPDATED RESUME In markdown format to be converted to PDF 
"""

    try:
        response = model.generate_content(prompt)
        # Limpiar cualquier texto introductorio o de cierre que el modelo pueda agregar por error
        cleaned_text = re.sub(r'^(.*?)```(pdf|text)?\n', '', response.text, flags=re.DOTALL)
        cleaned_text = re.sub(r'\n```$', '', cleaned_text).strip()
        return cleaned_text
    except Exception as e:
        logger.error(f"Error generating tailored resume: {str(e)}")
        return raw_resume

def save_rewritten_cv_to_db(cv_id: str, job_id: str, rewritten_text: str, original_cv_text: str):
    """Saves the rewritten CV to the 'cv_rewrites' collection in MongoDB."""
    try:
        cv_rewrites_collection = db['cv_rewrites']
        # Comprobar si ya existe una versión para este CV y trabajo
        latest_version = cv_rewrites_collection.find_one(
            {'original_cv_id': cv_id, 'job_offer_id': job_id},
            sort=[('version', -1)]
        )
        new_version = (latest_version['version'] + 1) if latest_version else 1

        rewrite_doc = {
            'original_cv_id': cv_id,
            'job_offer_id': job_id,
            'rewritten_text': rewritten_text,
            'original_cv_text': original_cv_text, # Guardar el CV original para referencia
            'version': new_version,
            'created_at': datetime.utcnow(),
            'model_used': 'gemini-1.5-flash',
        }
        result = cv_rewrites_collection.insert_one(rewrite_doc)
        logger.info(f"Rewritten CV saved to MongoDB with id: {result.inserted_id}, version: {new_version}")
        return result.inserted_id
    except Exception as e:
        logger.error(f"Failed to save rewritten CV to MongoDB: {e}")
        return None

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
    parser.add_argument('--initial_score', type=float, required=True, help='Initial matching score before tailoring')
    args = parser.parse_args(args_list)
    
    # Usar cliente centralizado con configuración optimizada
    client = get_qdrant_client(use_http=True, timeout=60.0)
    cv_chunks = get_all_chunks(client, args.cv_collection, args.cv_id)
    job_chunks = get_all_chunks(client, args.job_collection, args.job_id)
    cv_text = '\n'.join([c.payload['text'] for c in cv_chunks])
    job_text = '\n'.join([c.payload['text'] for c in job_chunks])
    cv_keywords = extract_keywords(cv_text)
    job_keywords = extract_keywords(job_text)
    
    tailored_text = generate_tailored_resume(
        raw_resume=cv_text,
        raw_job_description=job_text,
        extracted_resume_keywords=cv_keywords,
        extracted_job_keywords=job_keywords,
        current_cosine_similarity=args.initial_score
    )

    # Guardar en MongoDB
    save_rewritten_cv_to_db(
        cv_id=args.cv_id,
        job_id=args.job_id,
        rewritten_text=tailored_text,
        original_cv_text=cv_text
    )

    # Guardar el resultado como PDF
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_filename = f"CV_{args.cv_id}_adaptado_para_{args.job_id}.pdf"
    output_path = output_dir / output_filename

    save_text_as_pdf(tailored_text, output_path)

    print("\n--- Proceso de Adaptación Finalizado ---")
    print(f"\nEl CV adaptado ha sido guardado en: {output_path}")
    # Imprimir solo la ruta al final para que el pipeline la capture
    print(f"TAILORED_CV_PATH:{output_path}")

if __name__ == "__main__":
    main()
