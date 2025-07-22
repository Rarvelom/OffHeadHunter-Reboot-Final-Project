import sys
import logging
import random
import json
import uuid
from pathlib import Path
from job_matching import JobMatcher
sys.path.append(str(Path(__file__).parent / "src"))
from pdf_processor import PDFProcessor

# Configure logging for test output clarity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_job_matching")

UPLOADS_CVS_DIR = Path("uploads/cvs")
UPLOADS_JOBS_DIR = Path("uploads/job_descriptions")


def get_random_cv_pdf():
    pdfs = list(UPLOADS_CVS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF CVs found in uploads/cvs.")
        sys.exit(1)
    return random.choice(pdfs)

def extract_and_embed_job_pdfs(matcher):
    job_dir = Path("uploads/job_descriptions")
    job_files = list(job_dir.glob("*.pdf"))
    if not job_files:
        logger.error("No PDF job descriptions found in uploads/job_descriptions.")
        return []
    pdf_processor = PDFProcessor()
    job_ids = []
    for job_pdf in job_files:
        job_result = pdf_processor.process_pdf(job_pdf)
        job_text = job_result["text"]
        if not job_text.strip():
            logger.warning(f"No text extracted from {job_pdf.name}, skipping.")
            continue
        job_id = str(uuid.uuid4())
        matcher.add_job(job_id=job_id, job_description=job_text, metadata={"source_file": job_pdf.name})
        job_ids.append((job_id, job_text, job_pdf.name))
    return job_ids

def get_random_embedded_job(job_ids):
    if not job_ids:
        logger.error("No embedded job descriptions available for matching.")
        return None
    return random.choice(job_ids)

def main():
    matcher = JobMatcher()
    pdf_processor = PDFProcessor()

    # 1. Select a random CV PDF and extract text
    cv_pdf_path = get_random_cv_pdf()
    logger.info(f"Using CV PDF: {cv_pdf_path.name}")
    cv_result = pdf_processor.process_pdf(cv_pdf_path)
    resume_text = cv_result["text"]
    resume_id = str(uuid.uuid4())
    matcher.add_resume(
        resume_id=resume_id,
        resume_text=resume_text,
        metadata={"file_name": cv_pdf_path.name}
    )

    # 2. Extract and embed all PDF job descriptions
    job_ids = extract_and_embed_job_pdfs(matcher)
    if not job_ids:
        logger.error("No job descriptions found in uploads/job_descriptions.")
        return
    # 3. Randomly select one job for matching
    job_id, job_text, job_file = get_random_embedded_job(job_ids)
    logger.info(f"Selected job {job_file} for matching.")

    # 3. Run job matching using job_embeddings_BGE collection
    logger.info("Finding matching jobs for the CV...")
    matches = matcher.find_matching_jobs(resume_id, top_k=3)
    logger.info(f"Found {len(matches)} matching jobs")
    for idx, match in enumerate(matches):
        logger.info(f"Match #{idx+1}: {match['payload'].get('title', 'N/A')}")
        logger.info(f"Similarity score: {match['score']:.2f}")
        logger.info(f"Matching keywords: {', '.join(match['matching_keywords'])}")
        logger.info(f"Missing keywords: {', '.join(match['missing_keywords'])}")
    if matches:
        best_match = matches[0]
        logger.info("Generating tailored resume for the best match...")
        result = matcher.tailor_resume_for_job(resume_id, best_match['id'])
        if result:
            logger.info("\n=== Tailored Resume ===\n" + result["tailored_resume"])
            logger.info("\n=== Matching Keywords ===\n" + ", ".join(result["matching_keywords"]))
            logger.info("\n=== Missing Keywords ===\n" + ", ".join(result["missing_keywords"]))
        else:
            logger.error("Failed to tailor resume.")
    else:
        logger.warning("No matching jobs found.")

if __name__ == "__main__":
    main()
